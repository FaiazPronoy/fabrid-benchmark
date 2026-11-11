import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fabrid import runio
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import session_split

KINDS = ("yolo", "hybrid")
EMPTY_PICKS = 2


def label_name(index: int, kind: str) -> str:
    position = int(index) - (1 if kind == "hybrid" else 0)
    if kind not in KINDS or not 0 <= position < len(CLASSES):
        raise ValueError(f"label {index} out of range for {kind}")
    return CLASSES[position]


def refuse_locked(names, locked: dict[str, str], listed: dict[str, str] | None = None) -> None:
    for name in names:
        if locked.get(name) != "val":
            raise AssertionError(f"{name} is not in the session split's val part")
        if listed is not None and listed.get(name) != "val":
            raise AssertionError(f"{name} is not listed as val in the run's split.csv")


def select_images(records, locked: dict[str, str], listed: dict[str, str] | None = None) -> list:
    val = sorted((r for r in records if locked.get(r.name) == "val"), key=lambda r: r.name)
    chosen, taken = [], set()
    for class_id, class_name in enumerate(CLASSES):
        pick = next(
            (r for r in val if r.name not in taken and class_id in {c for c, _ in r.polygons}),
            None,
        )
        if pick is None:
            raise ValueError(f"no val image holds the class {class_name}")
        chosen.append(pick)
        taken.add(pick.name)
    empty = [r for r in val if r.n_lines == 0 and r.name not in taken][:EMPTY_PICKS]
    if len(empty) < EMPTY_PICKS:
        raise ValueError(f"fewer than {EMPTY_PICKS} val images have an empty label file")
    chosen += empty
    refuse_locked([r.name for r in chosen], locked, listed)
    return chosen


def find(relative: str, roots) -> Path | None:
    for root in roots:
        if (Path(root) / relative).is_file():
            return Path(root) / relative
    for root in roots:
        if Path(root).is_dir():
            found = sorted(Path(root).glob(f"**/{relative}"))
            if found:
                return found[0]
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def detection(label: str, score: float, box, mask: np.ndarray) -> dict:
    return {
        "label": label,
        "score": round(float(score), 6),
        "box": [round(float(v), 2) for v in box],
        "mask_area": int(mask.sum()),
    }


def kept(found: list[dict], min_score: float) -> list[dict]:
    chosen = [d for d in found if d["score"] >= min_score]
    return sorted(chosen, key=lambda d: (-d["score"], d["label"], d["box"]))


def yolo_predictor(weights: Path, settings: dict):
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(path: Path) -> list[dict]:
        result = model.predict(str(path), verbose=False, **settings)[0]
        boxes = result.boxes
        masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else None
        found = []
        for index, (cls, score, box) in enumerate(
            zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True)
        ):
            mask = masks[index] if masks is not None else np.zeros((0,), dtype=bool)
            found.append(detection(label_name(int(cls), "yolo"), score, box, mask))
        return found

    return predict


def hybrid_predictor(weights: Path, run_config: dict):
    import torch

    from fabrid.hybrid import build_hybrid, preparer

    model = build_hybrid(run_config, pretrained=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.float().eval()
    prepare = preparer(run_config)
    threshold = run_config["mask_threshold"]

    def predict(path: Path) -> list[dict]:
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        if prepare is not None:
            pixels = prepare(pixels)
        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            output = model([tensor])[0]
        masks = (output["masks"][:, 0] > threshold).numpy()
        return [
            detection(label_name(int(label), "hybrid"), score, box, mask)
            for box, score, label, mask in zip(
                output["boxes"].float().numpy(),
                output["scores"].float().tolist(),
                output["labels"].tolist(),
                masks,
                strict=True,
            )
        ]

    return predict


def pack(model_key: str, weights_sha: str, min_score: float, chosen, predict) -> dict:
    images = []
    for record in chosen:
        with Image.open(record.path) as image:
            width, height = image.size
        found = kept(predict(record.path), min_score)
        images.append({"file": record.name, "width": width, "height": height, "detections": found})
    return {
        "model_key": model_key,
        "weights_sha256": weights_sha,
        "min_score": min_score,
        "images": images,
    }


def write_pack(out_dir: Path, data: dict, chosen) -> list[Path]:
    (out_dir / "detections.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    folder = out_dir / "images"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    copies = []
    for record in chosen:
        shutil.copy2(record.path, folder / record.name)
        copies.append(folder / record.name)
    return copies


def read_split(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row["split"] for row in csv.DictReader(handle)}


def run_config(run: str, roots) -> dict:
    local = runio.repo_root() / "results" / run / "config.yaml"
    path = local if local.is_file() else find(f"results/{run}/config.yaml", roots)
    if path is None:
        raise SystemExit(f"config of run {run} not found in results/ or the roots")
    return yaml.safe_load(path.read_text(encoding="utf-8"))["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    if config["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    roots = [Path(root) for root in config["roots"]]
    weights = find(f"{config['run']}/{config['weights']}", roots)
    if weights is None:
        raise SystemExit(f"weights {config['run']}/{config['weights']} not found under {roots}")
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    locked = session_split(records, config["gap_seconds"])
    listed = read_split(runio.repo_root() / "results" / config["run"] / "split.csv")
    chosen = select_images(records, locked, listed)
    print(f"weights {weights}, images {[r.name for r in chosen]}")
    if config["kind"] == "yolo":
        predict = yolo_predictor(weights, config["predict"])
    else:
        predict = hybrid_predictor(weights, run_config(config["run"], roots))
    data = pack(config["model_key"], sha256_of(weights), config["min_score"], chosen, predict)
    out_dir = runio.start_run(config["step"], config["model_key"], config)
    copies = write_pack(out_dir, data, chosen)
    for image in data["images"]:
        strong = sum(d["score"] >= config["report_score"] for d in image["detections"])
        print(
            f"{image['file']}: {strong} detections at score {config['report_score']} or more, "
            f"{len(image['detections'])} kept"
        )
    large = [p.name for p in copies if not runio.publishable(p)]
    if large:
        print(f"these images are too large to commit and will be left out: {large}")
    print(f"wrote {out_dir}")
    if config.get("publish", True):
        runio.publish(out_dir)


if __name__ == "__main__":
    main()
