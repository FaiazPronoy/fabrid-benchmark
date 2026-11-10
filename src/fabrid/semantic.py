import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fabrid import runio
from fabrid.geometry import rasterize
from fabrid.labels import CLASSES, load_index
from fabrid.maskrcnn import (
    find_checkpoint,
    lr_factor,
    save_checkpoint,
    seed_everything,
    split_records,
    write_json,
)
from fabrid.sessions import session_split
from fabrid.yolo import Deadline, arm_mapping, write_csv

INPUT_ROOT = Path("/kaggle/input")
IGNORE = 255
N_CLASSES = len(CLASSES) + 1
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAPS_NAME = "val_maps.npz"


def build_target(polygons, height: int, width: int) -> tuple[np.ndarray, int]:
    target = np.zeros((height, width), dtype=np.uint8)
    contested = np.zeros((height, width), dtype=bool)
    for class_id, points in polygons:
        mask = rasterize(points * [width, height], height, width)
        label = class_id + 1
        contested |= mask & (target > 0) & (target != label)
        target[mask] = label
    return target, int(contested.sum())


def pad(pixels: np.ndarray, target: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape
    if height > size or width > size:
        raise ValueError(f"image of {height}x{width} does not fit in {size}x{size}")
    padded_pixels = np.zeros((size, size, pixels.shape[2]), dtype=pixels.dtype)
    padded_pixels[:height, :width] = pixels
    padded_target = np.full((size, size), IGNORE, dtype=np.uint8)
    padded_target[:height, :width] = target
    return padded_pixels, padded_target


def normalize(pixels: np.ndarray) -> np.ndarray:
    return (pixels.astype(np.float32) / 255 - MEAN) / STD


class Samples:
    def __init__(self, records, size: int, flips=(0.0, 0.0), seed: int = 0):
        self.records = records
        self.size = size
        self.flips = flips
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        with Image.open(record.path) as image:
            pixels = np.array(image.convert("RGB"))
        target, _ = build_target(record.polygons, *pixels.shape[:2])
        if any(self.flips):
            rng = np.random.default_rng([self.seed, self.epoch, index])
            horizontal, vertical = (rng.random() < p for p in self.flips)
            if horizontal:
                pixels, target = pixels[:, ::-1], target[:, ::-1]
            if vertical:
                pixels, target = pixels[::-1], target[::-1]
        return pad(normalize(pixels), target, self.size)

    def __getitem__(self, index: int):
        import torch

        pixels, target = self.load(index)
        image = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        return image, torch.from_numpy(np.ascontiguousarray(target)).long(), index


def overlap_counts(records) -> dict[str, int]:
    pixels, images = 0, 0
    for record in records:
        with Image.open(record.path) as image:
            height, width = image.height, image.width
        _, contested = build_target(record.polygons, height, width)
        pixels += contested
        images += contested > 0
    return {"images": len(records), "contested_pixels": pixels, "contested_images": images}


def confusion(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE
    pred, target = pred[valid], target[valid]
    counts = np.zeros((N_CLASSES, 3), dtype=np.int64)
    for label in range(1, N_CLASSES):
        predicted, actual = pred == label, target == label
        counts[label] = [
            (predicted & actual).sum(),
            (predicted & ~actual).sum(),
            (~predicted & actual).sum(),
        ]
    return counts


def ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def class_scores(counts: np.ndarray) -> dict:
    per_class = {}
    for label, name in enumerate(CLASSES, start=1):
        tp, fp, fn = (int(v) for v in counts[label])
        per_class[name] = {
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "iou": ratio(tp, tp + fp + fn),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return per_class


def mean_dice(per_class: dict) -> float | None:
    values = [row["dice"] for row in per_class.values() if row["dice"] is not None]
    return float(np.mean(values)) if values else None


def is_alarm(pred: np.ndarray, target: np.ndarray, min_pixels: int) -> bool:
    valid = target != IGNORE
    counts = np.bincount(pred[valid].ravel(), minlength=N_CLASSES)
    return bool((counts[1:] >= min_pixels).any())


def dice_loss(logits, target, smooth: float = 1.0):
    import torch

    probs = torch.softmax(logits.float(), dim=1)
    valid = (target != IGNORE).unsqueeze(1)
    labels = torch.where(target == IGNORE, torch.zeros_like(target), target)
    onehot = torch.nn.functional.one_hot(labels, N_CLASSES).permute(0, 3, 1, 2).float()
    probs, onehot = probs * valid, onehot * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)[1:]
    denom = probs.sum(dims)[1:] + onehot.sum(dims)[1:]
    return 1 - ((2 * inter + smooth) / (denom + smooth)).mean()


def upsampled_logits(model, images):
    import torch

    logits = model(pixel_values=images).logits
    return torch.nn.functional.interpolate(
        logits, size=images.shape[-2:], mode="bilinear", align_corners=False
    )


def build_model(config: dict):
    from transformers import SegformerForSemanticSegmentation

    names = ("background", *CLASSES)
    return SegformerForSemanticSegmentation.from_pretrained(
        config["weights"],
        num_labels=N_CLASSES,
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
        ignore_mismatched_sizes=True,
    )


def make_optimizer(model, config: dict):
    import torch

    return torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )


def train_step(model, optimizer, scaler, images, targets, config: dict, amp: bool):
    import torch

    with torch.autocast(images.device.type, enabled=amp):
        logits = upsampled_logits(model, images)
    ce = torch.nn.functional.cross_entropy(logits.float(), targets, ignore_index=IGNORE)
    dice = dice_loss(logits, targets)
    loss = ce + config["dice_weight"] * dice
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss, ce {ce.item()} dice {dice.item()}")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {"loss": loss.item(), "ce": ce.item(), "dice_loss": dice.item()}


def stack(data: Samples, indices, device):
    import torch

    images, targets, _ = zip(*(data[i] for i in indices), strict=True)
    return torch.stack(images).to(device), torch.stack(targets).to(device)


def overfit_check(config: dict, train_records, device, amp: bool) -> dict:
    import torch

    settings = config["sanity"]
    seed_everything(config["seed"])
    picks = [record for record in train_records if record.polygons][: settings["images"]]
    data = Samples(picks, config["imgsz"])
    batches = [
        stack(data, range(start, min(start + config["batch"], len(data))), device)
        for start in range(0, len(data), config["batch"])
    ]
    model = build_model(config).to(device).train()
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    totals = []
    for step in range(settings["iterations"]):
        images, targets = batches[step % len(batches)]
        totals.append(train_step(model, optimizer, scaler, images, targets, config, amp)["loss"])
    start, end = totals[0], float(np.mean(totals[-10:]))
    return {
        "images": [record.name for record in picks],
        "iterations": settings["iterations"],
        "start_loss": start,
        "end_loss_mean_last_10": end,
        "ratio": end / start,
        "limit": settings["ratio"],
        "losses_every_10": totals[::10],
        "passed": len(picks) == settings["images"] and end < settings["ratio"] * start,
    }


def empty_check(config: dict, train_records, device, amp: bool) -> dict:
    import torch

    empty = [record for record in train_records if not record.polygons][:2]
    result = {"images": [record.name for record in empty]}
    if not empty:
        result.update(error="no defect-free train image", passed=False)
        return result
    images, targets = stack(Samples(empty, config["imgsz"]), range(len(empty)), device)
    model = build_model(config).to(device).train()
    with torch.autocast(device.type, enabled=amp):
        logits = upsampled_logits(model, images)
    loss = torch.nn.functional.cross_entropy(logits.float(), targets, ignore_index=IGNORE)
    loss = loss + config["dice_weight"] * dice_loss(logits, targets)
    loss.backward()
    result.update(loss=loss.item(), passed=bool(torch.isfinite(loss)))
    return result


def sanity_checks(config: dict, train_records, device, amp: bool) -> dict:
    import torch

    overfit = overfit_check(config, train_records, device, amp)
    torch.cuda.empty_cache()
    empty = empty_check(config, train_records, device, amp)
    torch.cuda.empty_cache()
    return {
        "overfit": overfit,
        "empty_image": empty,
        "passed": overfit["passed"] and empty["passed"],
    }


def evaluate(model, loader, records, device, amp: bool, min_pixels: int) -> tuple[dict, dict]:
    import torch

    model.eval()
    counts = np.zeros((N_CLASSES, 3), dtype=np.int64)
    alarms, clean, maps = 0, 0, {}
    with torch.no_grad():
        for images, targets, indices in loader:
            with torch.autocast(device.type, enabled=amp):
                logits = upsampled_logits(model, images.to(device))
            preds = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            for pred, target, index in zip(preds, targets.numpy(), indices.tolist(), strict=True):
                record = records[index]
                counts += confusion(pred, target)
                rows, cols = np.nonzero(target != IGNORE)
                maps[record.name] = pred[: rows.max() + 1, : cols.max() + 1]
                if not record.polygons:
                    clean += 1
                    alarms += is_alarm(pred, target, min_pixels)
    per_class = class_scores(counts)
    scores = {
        "mean_dice": mean_dice(per_class),
        "per_class": per_class,
        "false_alarm": {
            "defect_free_images": clean,
            "alarms": alarms,
            "rate": ratio(alarms, clean),
            "min_pixels": min_pixels,
        },
    }
    return scores, maps


def epoch_row(epoch: int, lr: float, losses: dict, scores: dict, seconds: float) -> dict:
    row = {"epoch": epoch, "lr": lr, **losses, "val_mean_dice": scores["mean_dice"]}
    for name, values in scores["per_class"].items():
        for metric in ("dice", "iou"):
            row[f"{name.replace(' ', '_')}_{metric}"] = values[metric]
    row["false_alarm_rate"] = scores["false_alarm"]["rate"]
    row["seconds"] = round(seconds, 1)
    return row


def adopt(source: Path, runs: Path) -> None:
    origin = source.parent.parent
    if origin.resolve() == runs.resolve():
        return
    for relative in ("checkpoints/best.pth", MAPS_NAME):
        if (origin / relative).is_file():
            (runs / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin / relative, runs / relative)


def train(config: dict, name: str, train_records, val_records, deadline, run_dir: Path):
    import torch
    from torch.utils.data import DataLoader

    cuda = torch.cuda.is_available()
    device = torch.device(config["device"] if cuda else "cpu")
    if not cuda:
        print(f"{name}: no GPU found, running on CPU")
    amp = bool(config["amp"]) and device.type == "cuda"
    runs = runio.runs_dir(config["step"], name)
    checkpoints = runs / "checkpoints"
    source = find_checkpoint(name, [runs.parent.parent, INPUT_ROOT])
    saved = torch.load(source, map_location="cpu", weights_only=False) if source else None
    sanity = saved["sanity"] if saved else sanity_checks(config, train_records, device, amp)
    write_json(run_dir / "sanity.json", sanity)
    if not sanity["passed"]:
        raise SystemExit(f"{name}: sanity check failed, see {run_dir / 'sanity.json'}")
    print(f"{name}: sanity checks passed, wrote {run_dir / 'sanity.json'}")

    seed_everything(config["seed"])
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    flips = (config["hflip"], config["vflip"])
    train_data = Samples(train_records, config["imgsz"], flips, config["seed"])
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

    loader_args = {"num_workers": config["workers"], "pin_memory": cuda}
    val_data = Samples(val_records, config["imgsz"])
    val_loader = DataLoader(val_data, batch_size=config["batch"], **loader_args)
    for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
        started = time.time()
        train_data.epoch = epoch
        generator = torch.Generator().manual_seed(config["seed"] * 1000 + epoch)
        order = torch.randperm(len(train_data), generator=generator).tolist()
        loader = DataLoader(train_data, batch_size=config["batch"], sampler=order, **loader_args)
        sums = dict.fromkeys(("loss", "ce", "dice_loss"), 0.0)
        model.train()
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            losses = train_step(model, optimizer, scaler, images, targets, config, amp)
            scheduler.step()
            for key, value in losses.items():
                sums[key] += value
        means = {key: round(value / steps, 5) for key, value in sums.items()}
        scores, maps = evaluate(model, val_loader, val_records, device, amp, config["alarm_pixels"])
        lr = optimizer.param_groups[0]["lr"]
        state["history"].append(epoch_row(epoch, lr, means, scores, time.time() - started))
        value = scores["mean_dice"] or 0.0
        if state["best"] is None or value > state["best"]["mean_dice"]:
            state["best"] = {"epoch": epoch, "mean_dice": value, "scores": scores}
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoints / "best.pth")
            np.savez_compressed(runs / MAPS_NAME, **maps)
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
        print(
            f"{name}: epoch {epoch}/{config['epochs']} loss {means['loss']:.4f} "
            f"val mean Dice {value:.4f}"
        )
        if deadline.passed() and epoch < config["epochs"]:
            print(f"{name}: wall clock limit reached after epoch {epoch}, state saved")
            return "paused", state
    return "finished", state


def finalize(config: dict, name: str, state: dict, mapping, overlaps, run_dir: Path) -> dict:
    runs = runio.runs_dir(config["step"], name)
    best = state["best"]
    summary = {
        "model": config["model"],
        "weights_source": config["weights"],
        "arm": config["arm"],
        "seed": config["seed"],
        "split": "val",
        "endpoint": "mean over 4 defect classes of pooled val Dice, best epoch by the same",
        "best_epoch": best["epoch"],
        "val_mean_dice": best["scores"]["mean_dice"],
        "epochs_run": len(state["history"]),
        "n_train": overlaps["train"]["images"],
        "n_val": overlaps["val"]["images"],
        "overlap": overlaps,
        **best["scores"],
        "class_maps": str(runs / MAPS_NAME),
        "weights": str(runs / "checkpoints" / "best.pth"),
    }
    write_json(run_dir / "metrics.json", summary)
    write_csv(run_dir / "split.csv", ["name", "split"], sorted(mapping.items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
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
    overlaps = {"train": overlap_counts(train_records), "val": overlap_counts(val_records)}
    for split, count in overlaps.items():
        print(
            f"{name}: {split} {count['images']} images, {count['contested_pixels']} pixels in "
            f"{count['contested_images']} images covered by more than one class"
        )
    run_dir = runio.start_run(config["step"], name, config)
    status, state = train(config, name, train_records, val_records, deadline, run_dir)
    if status == "paused":
        print(f"{name}: paused, checkpoint in {runio.runs_dir(config['step'], name)}")
        return
    summary = finalize(config, name, state, mapping, overlaps, run_dir)
    print(f"{name}: best epoch {summary['best_epoch']}, val mean Dice {summary['val_mean_dice']}")
    print(f"{name}: wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
