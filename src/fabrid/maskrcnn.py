import argparse
import contextlib
import io
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fabrid import runio
from fabrid.geometry import rasterize
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import session_split
from fabrid.yolo import Deadline, arm_mapping, assert_unlocked, write_csv

INPUT_ROOT = Path("/kaggle/input")
LOSSES = ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg")
KINDS = (("box", "bbox"), ("mask", "segm"))
OVERALL = ("AP50", "AP50-95", "AR100")


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size:
            boxes[index] = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
    return boxes


def build_target(polygons, height: int, width: int) -> tuple[dict, int]:
    masks = np.zeros((len(polygons), height, width), dtype=np.uint8)
    for index, (_, points) in enumerate(polygons):
        masks[index] = rasterize(points * [width, height], height, width)
    labels = np.array([class_id + 1 for class_id, _ in polygons], dtype=np.int64)
    boxes = boxes_from_masks(masks)
    keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
    target = {"boxes": boxes[keep], "labels": labels[keep], "masks": masks[keep]}
    return target, int((~keep).sum())


def flip(pixels: np.ndarray, target: dict, horizontal: bool, vertical: bool):
    masks = target["masks"]
    if horizontal:
        pixels, masks = pixels[:, ::-1], masks[:, :, ::-1]
    if vertical:
        pixels, masks = pixels[::-1], masks[:, ::-1]
    return pixels, {**target, "masks": masks, "boxes": boxes_from_masks(masks)}


class Samples:
    def __init__(self, records, flips=(0.0, 0.0), seed: int = 0, prepare=None):
        self.records = records
        self.prepare = prepare
        self.flips = flips
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def load(self, index: int) -> tuple[np.ndarray, dict]:
        record = self.records[index]
        with Image.open(record.path) as image:
            pixels = np.array(image.convert("RGB"))
        if self.prepare is not None:
            pixels = self.prepare(pixels)
        target, _ = build_target(record.polygons, *pixels.shape[:2])
        if any(self.flips):
            rng = np.random.default_rng([self.seed, self.epoch, index])
            horizontal, vertical = (rng.random() < p for p in self.flips)
            pixels, target = flip(pixels, target, horizontal, vertical)
        return pixels, target

    def __getitem__(self, index: int):
        import torch

        pixels, target = self.load(index)
        image = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float() / 255
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in target.items()}
        return image, tensors, index


def collate(batch):
    return tuple(zip(*batch, strict=True))


def split_records(records, mapping: dict[str, str], locked: dict[str, str]):
    assert_unlocked(mapping, locked)
    chosen = {"train": [], "val": []}
    for record in records:
        if record.name in mapping:
            chosen[mapping[record.name]].append(record)
    return chosen["train"], chosen["val"]


def image_size(record) -> tuple[int, int]:
    with Image.open(record.path) as image:
        return image.height, image.width


def instance_counts(records) -> dict[str, int]:
    kept, dropped = 0, 0
    for record in records:
        target, lost = build_target(record.polygons, *image_size(record))
        kept += len(target["labels"])
        dropped += lost
    return {"images": len(records), "kept": kept, "dropped_under_1px": dropped}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_model(config: dict):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
        min_size=config["imgsz"],
        max_size=config["imgsz"],
    )
    n_classes = len(CLASSES) + 1
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, n_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, n_classes)
    return model


def make_optimizer(model, config: dict):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def to_device(images, targets, device):
    images = [image.to(device, non_blocking=True) for image in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def train_step(model, optimizer, scaler, images, targets, device, amp: bool):
    import torch

    with torch.autocast(device.type, enabled=amp):
        losses = model(images, targets)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss {[(k, float(v)) for k, v in losses.items()]}")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {k: float(v) for k, v in losses.items()}, float(loss)


def overfit_check(
    config: dict, train_records, device, amp: bool, builder=None, prepare=None
) -> dict:
    import torch

    settings = config["sanity"]
    seed_everything(config["seed"])
    picks = [record for record in train_records if record.polygons][: settings["images"]]
    data = Samples(picks, prepare=prepare)
    batches = []
    for start in range(0, len(data), config["batch"]):
        stop = min(start + config["batch"], len(data))
        images, targets, _ = collate([data[i] for i in range(start, stop)])
        batches.append(to_device(images, targets, device))
    model = (builder or build_model)(config).to(device).train()
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    totals = []
    for step in range(settings["iterations"]):
        images, targets = batches[step % len(batches)]
        totals.append(train_step(model, optimizer, scaler, images, targets, device, amp)[1])
    start, end = totals[0], float(np.mean(totals[-10:]))
    return {
        "images": [record.name for record in picks],
        "iterations": settings["iterations"],
        "start_loss": start,
        "end_loss_mean_last_10": end,
        "ratio": end / start,
        "limit": settings["ratio"],
        "losses_every_10": totals[::10],
        "passed": end < settings["ratio"] * start,
    }


def empty_check(config: dict, train_records, device, amp: bool, builder=None, prepare=None) -> dict:
    import torch

    full = next(record for record in train_records if record.polygons)
    empty = next((record for record in train_records if not record.polygons), None)
    data = Samples([full, empty or full], prepare=prepare)
    images, targets, _ = collate([data[0], data[1]])
    if empty is None:
        blank, _ = build_target([], *images[1].shape[1:])
        targets = (targets[0], {k: torch.from_numpy(v) for k, v in blank.items()})
    images, targets = to_device(images, targets, device)
    model = (builder or build_model)(config).to(device).train()
    result = {
        "images": [full.name, empty.name if empty else "synthetic empty target"],
        "empty_shapes": {k: list(v.shape) for k, v in targets[1].items()},
    }
    try:
        with torch.autocast(device.type, enabled=amp):
            losses = model(images, targets)
        loss = sum(losses.values())
        loss.backward()
        result.update(loss=float(loss), passed=bool(torch.isfinite(loss)))
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", passed=False)
    return result


def sanity_checks(
    config: dict, train_records, device, amp: bool, builder=None, prepare=None
) -> dict:
    import torch

    overfit = overfit_check(config, train_records, device, amp, builder, prepare)
    torch.cuda.empty_cache()
    empty = empty_check(config, train_records, device, amp, builder, prepare)
    torch.cuda.empty_cache()
    return {
        "overfit": overfit,
        "empty_image": empty,
        "passed": overfit["passed"] and empty["passed"],
    }


def encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_util

    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}


def xywh(box) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


def coco_truth(records) -> dict:
    images, annotations = [], []
    for image_id, record in enumerate(records, start=1):
        height, width = image_size(record)
        images.append({"id": image_id, "file_name": record.name, "width": width, "height": height})
        target, _ = build_target(record.polygons, height, width)
        for box, label, mask in zip(
            target["boxes"], target["labels"], target["masks"], strict=True
        ):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": encode(mask),
                    "area": int(mask.sum()),
                    "bbox": xywh(box),
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def predict(model, loader, device, amp: bool, threshold: float) -> list[dict]:
    import torch

    model.eval()
    detections = []
    with torch.no_grad():
        for images, _, indices in loader:
            with torch.autocast(device.type, enabled=amp):
                outputs = model([image.to(device) for image in images])
            for index, output in zip(indices, outputs, strict=True):
                masks = (output["masks"][:, 0] > threshold).cpu().numpy()
                boxes = output["boxes"].float().cpu().numpy()
                scores = output["scores"].float().cpu().tolist()
                labels = output["labels"].cpu().tolist()
                for box, score, label, mask in zip(boxes, scores, labels, masks, strict=True):
                    detections.append(
                        {
                            "image_id": index + 1,
                            "category_id": int(label),
                            "bbox": xywh(box),
                            "score": round(score, 5),
                            "segmentation": encode(mask),
                        }
                    )
    return detections


def mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def class_scores(precision: np.ndarray, recall: np.ndarray, k=slice(None)) -> dict:
    return {
        "AP50": mean_valid(precision[0, :, k, 0, -1]),
        "AP50-95": mean_valid(precision[:, :, k, 0, -1]),
        "AR100": mean_valid(recall[:, k, 0, -1]),
    }


def summarize(evaluator) -> dict:
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    per_class = {
        CLASSES[cat - 1]: class_scores(precision, recall, k)
        for k, cat in enumerate(evaluator.params.catIds)
    }
    return {"overall": class_scores(precision, recall), "per_class": per_class}


def no_detection_scores(truth: dict) -> dict:
    present = {annotation["category_id"] for annotation in truth["annotations"]}
    zero = dict.fromkeys(OVERALL, 0.0)
    per_class = {
        name: dict(zero) if i + 1 in present else dict.fromkeys(OVERALL)
        for i, name in enumerate(CLASSES)
    }
    return {"overall": dict(zero) if present else dict.fromkeys(OVERALL), "per_class": per_class}


def coco_scores(truth: dict, detections: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = {}
    with contextlib.redirect_stdout(io.StringIO()):
        ground = COCO()
        ground.dataset = truth
        ground.createIndex()
        for kind, iou_type in KINDS:
            drop = "segmentation" if kind == "box" else "bbox"
            listed = [{k: v for k, v in d.items() if k != drop} for d in detections]
            if not listed:
                results[kind] = no_detection_scores(truth)
                continue
            evaluator = COCOeval(ground, ground.loadRes(listed), iou_type)
            evaluator.evaluate()
            evaluator.accumulate()
            results[kind] = summarize(evaluator)
    return results


def epoch_row(epoch: int, lr: float, losses: dict, scores: dict, seconds: float) -> dict:
    row = {"epoch": epoch, "lr": lr, **losses}
    for kind, _ in KINDS:
        for metric in OVERALL:
            row[f"{kind}_{metric}"] = scores[kind]["overall"][metric]
    for kind, _ in KINDS:
        for name in CLASSES:
            for metric in ("AP50", "AP50-95"):
                row[f"{kind}_{name.replace(' ', '_')}_{metric}"] = scores[kind]["per_class"][name][
                    metric
                ]
    row["seconds"] = round(seconds, 1)
    return row


def save_checkpoint(state: dict, directory: Path) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / "last.pth.partial"
    torch.save(state, partial)
    partial.replace(directory / "last.pth")
    (directory / "last.json").write_text(json.dumps({"epoch": state["epoch"]}), encoding="utf-8")


def find_checkpoint(name: str, roots, step: str = "baselines") -> Path | None:
    found = []
    for root in roots:
        if not Path(root).is_dir():
            continue
        for marker in Path(root).glob(f"**/{step}/{name}/checkpoints/last.json"):
            weights = marker.with_name("last.pth")
            try:
                epoch = int(json.loads(marker.read_text(encoding="utf-8"))["epoch"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if weights.is_file():
                found.append((epoch, weights.stat().st_mtime, weights))
    return max(found, key=lambda item: item[:2])[2] if found else None


def adopt(source: Path, runs: Path) -> None:
    origin = source.parent.parent
    if origin.resolve() == runs.resolve():
        return
    for relative in ("checkpoints/best.pth", "val_predictions.json"):
        if (origin / relative).is_file():
            (runs / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin / relative, runs / relative)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train(
    config: dict,
    name: str,
    train_records,
    val_records,
    deadline,
    run_dir: Path,
    builder=None,
    prepare=None,
):
    import torch
    from torch.utils.data import DataLoader

    cuda = torch.cuda.is_available()
    device = torch.device(config["device"] if cuda else "cpu")
    if not cuda:
        print(f"{name}: no GPU found, running on CPU")
    amp = bool(config["amp"]) and device.type == "cuda"
    runs = runio.runs_dir(config["step"], name)
    checkpoints = runs / "checkpoints"
    source = find_checkpoint(name, [runs.parent.parent, INPUT_ROOT], config["step"])
    saved = torch.load(source, map_location="cpu", weights_only=False) if source else None
    if saved:
        sanity = saved["sanity"]
    else:
        sanity = sanity_checks(config, train_records, device, amp, builder, prepare)
    write_json(run_dir / "sanity.json", sanity)
    if not sanity["passed"]:
        raise SystemExit(f"{name}: sanity check failed, see {run_dir / 'sanity.json'}")
    print(f"{name}: sanity checks passed, wrote {run_dir / 'sanity.json'}")

    seed_everything(config["seed"])
    model = (builder or build_model)(config).to(device)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    flips = (config["hflip"], config["vflip"])
    train_data = Samples(train_records, flips, config["seed"], prepare)
    steps = math.ceil(len(train_data) / config["batch"])
    warmup, total = config["warmup_epochs"] * steps, config["epochs"] * steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total))
    state = {"epoch": 0, "best": None, "history": [], "sanity": sanity}
    if saved is not None:
        adopt(source, runs)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"])
        state = {key: saved[key] for key in state}
        print(f"{name}: resuming after epoch {state['epoch']} from {source}")
        del saved

    truth = coco_truth(val_records)
    write_json(runs / "val_ground_truth.json", truth)
    loader_args = {"collate_fn": collate, "num_workers": config["workers"], "pin_memory": cuda}
    val_data = Samples(val_records, prepare=prepare)
    val_loader = DataLoader(val_data, batch_size=config["batch"], **loader_args)
    for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
        started = time.time()
        train_data.epoch = epoch
        generator = torch.Generator().manual_seed(config["seed"] * 1000 + epoch)
        order = torch.randperm(len(train_data), generator=generator).tolist()
        loader = DataLoader(train_data, batch_size=config["batch"], sampler=order, **loader_args)
        sums = dict.fromkeys(("loss", *LOSSES), 0.0)
        model.train()
        for images, targets, _ in loader:
            images, targets = to_device(images, targets, device)
            losses, loss = train_step(model, optimizer, scaler, images, targets, device, amp)
            scheduler.step()
            for key, value in {"loss": loss, **losses}.items():
                sums[key] = sums.get(key, 0.0) + value
        means = {key: round(value / steps, 5) for key, value in sums.items()}
        detections = predict(model, val_loader, device, amp, config["mask_threshold"])
        scores = coco_scores(truth, detections)
        lr = optimizer.param_groups[0]["lr"]
        state["history"].append(epoch_row(epoch, lr, means, scores, time.time() - started))
        mask_ap50 = scores["mask"]["overall"]["AP50"] or 0.0
        if state["best"] is None or mask_ap50 > state["best"]["mask_AP50"]:
            state["best"] = {"epoch": epoch, "mask_AP50": mask_ap50, "scores": scores}
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoints / "best.pth")
            (runs / "val_predictions.json").write_text(json.dumps(detections), encoding="utf-8")
        state["epoch"] = epoch
        save_checkpoint(
            {
                **state,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoints,
        )
        history = state["history"]
        write_csv(run_dir / "curves.csv", list(history[0]), [list(r.values()) for r in history])
        box_ap50 = scores["box"]["overall"]["AP50"] or 0.0
        print(
            f"{name}: epoch {epoch}/{config['epochs']} loss {means['loss']:.4f} "
            f"val box AP50 {box_ap50:.4f} mask AP50 {mask_ap50:.4f}"
        )
        if deadline.passed() and epoch < config["epochs"]:
            print(f"{name}: wall clock limit reached after epoch {epoch}, state saved")
            return "paused", state
    return "finished", state


def finalize(
    config: dict, name: str, state: dict, mapping, counts, run_dir: Path, extra=None
) -> dict:
    runs = runio.runs_dir(config["step"], name)
    best = state["best"]
    summary = {
        "model": config["model"],
        "arm": config["arm"],
        "seed": config["seed"],
        "split": "val",
        "endpoint": "val mask AP50 at the best epoch by val mask AP50",
        "best_epoch": best["epoch"],
        "val_mask_AP50": best["mask_AP50"],
        "epochs_run": len(state["history"]),
        "n_train": counts["train"]["images"],
        "n_val": counts["val"]["images"],
        "instances": counts,
        **best["scores"],
        "predictions": str(runs / "val_predictions.json"),
        "ground_truth": str(runs / "val_ground_truth.json"),
        "weights": str(runs / "checkpoints" / "best.pth"),
        **(extra or {}),
    }
    write_json(run_dir / "metrics.json", summary)
    write_csv(run_dir / "split.csv", ["name", "split"], sorted(mapping.items()))
    return summary


def run(config: dict, builder=None, extra=None, prepare=None) -> None:
    deadline = Deadline(config["hours_limit"])
    name = f"{config['model']}_{config['arm']}_seed{config['seed']}"
    done = runio.repo_root() / "results" / config["step"] / name / "metrics.json"
    if done.is_file():
        print(f"{name}: already complete at {done}, skipping")
        return
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    locked = session_split(records, config["gap_seconds"])
    mapping = arm_mapping(records, config["arm"], config["seed"], config["gap_seconds"])
    train_records, val_records = split_records(records, mapping, locked)
    counts = {"train": instance_counts(train_records), "val": instance_counts(val_records)}
    for split, count in counts.items():
        print(
            f"{name}: {split} {count['images']} images, {count['kept']} instances, "
            f"{count['dropped_under_1px']} dropped with a box under 1 px"
        )
    run_dir = runio.start_run(config["step"], name, config)
    status, state = train(
        config, name, train_records, val_records, deadline, run_dir, builder, prepare
    )
    if status == "paused":
        print(f"{name}: paused, checkpoint in {runio.runs_dir(config['step'], name)}")
        return
    summary = finalize(config, name, state, mapping, counts, run_dir, extra)
    print(
        f"{name}: best epoch {summary['best_epoch']}, val mask AP50 {summary['val_mask_AP50']:.4f}"
    )
    print(f"{name}: wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(runio.load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
