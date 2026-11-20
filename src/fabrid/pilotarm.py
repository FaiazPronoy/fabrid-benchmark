import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fabrid import runio
from fabrid.evaluate import Scorer, save_scored, score_images
from fabrid.labels import CLASSES
from fabrid.pilot import (
    blocked_names,
    build,
    class_map,
    find_file,
    load_photos,
    views_dir,
)
from fabrid.reference import sha256_of
from fabrid.yolo import Deadline, curves, read_rows, train, write_csv

OFFSET = 1


def encode(mask: np.ndarray) -> str:
    flat = mask.astype(bool).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.concatenate([[False], flat, [False]]).astype(int)))
    edges = [0, *changes.tolist(), flat.size]
    counts = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    if counts and counts[-1] == 0:
        counts.pop()
    text = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = value != -1 if code & 0x10 else value != 0
            if more:
                code |= 0x20
            text.append(chr(code + 48))
    return "".join(text)


def class_lookup(names) -> dict[int, int]:
    lookup = class_map(names)
    if sorted(lookup.values()) != list(range(len(CLASSES))):
        raise ValueError(f"model classes {names} do not cover {list(CLASSES)}")
    return lookup


def entry(name: str, class_id: int, score: float, box, mask: np.ndarray) -> dict:
    x0, y0, x1, y1 = (float(v) for v in box)
    return {
        "file_name": name,
        "category_id": class_id + OFFSET,
        "score": round(float(score), 6),
        "bbox": [round(v, 2) for v in (x0, y0, x1 - x0, y1 - y0)],
        "segmentation": {"size": list(mask.shape), "counts": encode(mask)},
    }


def predict(weights: Path, photos, settings: dict, device) -> list[dict]:
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(weights))
    lookup = class_lookup(model.names)
    print(f"model classes {model.names} read as {[CLASSES[lookup[k]] for k in sorted(lookup)]}")
    found = []
    for photo in photos:
        with Image.open(photo.path) as image:
            width, height = image.size
        result = model.predict(str(photo.path), verbose=False, device=device, **settings)[0]
        boxes = result.boxes
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy() > 0.5
        if masks.shape[1:] != (height, width):
            raise ValueError(f"{photo.name}: mask shape {masks.shape[1:]} is not {height}x{width}")
        for index, (cls, score, box) in enumerate(
            zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True)
        ):
            found.append(entry(photo.name, lookup[int(cls)], score, box, masks[index]))
    return found


def rectangle(points: np.ndarray) -> np.ndarray:
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def box_view(photos, predictions: list[dict]) -> tuple[list, list[dict]]:
    boxed = [
        SimpleNamespace(
            name=p.name, path=p.path, polygons=[(c, rectangle(q)) for c, q in p.polygons]
        )
        for p in photos
    ]
    items = []
    for item in predictions:
        x, y, w, h = item["bbox"]
        items.append({**item, "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]})
    return boxed, items


def score(photos, predictions: list[dict], config: dict):
    names = sorted(p.name for p in photos)
    listed = {p.name for p in photos}
    outside = sorted({item["file_name"] for item in predictions} - listed)
    if outside:
        raise AssertionError(f"predictions for images outside the pilot test: {outside[:5]}")
    resolved = [item["file_name"] for item in predictions]
    mask, dropped = score_images(photos, predictions, resolved, names, config, OFFSET)
    boxed, items = box_view(photos, predictions)
    box, _ = score_images(boxed, items, resolved, names, config, OFFSET)
    return names, mask, box, dropped


def point(scored) -> dict:
    values = Scorer(scored).evaluate(np.ones(scored.n_images))
    clean = {k: None if np.isnan(v) else round(float(v), 6) for k, v in values.items()}
    return {"mean": clean["ap50"], "per_class": {c: clean[f"ap50/{c}"] for c in CLASSES}}


def trained_weights(config: dict, deadline: Deadline, run_dir: Path, data_yaml: Path):
    name = config["name"]
    if train(config, name, data_yaml, deadline) == "paused":
        return None
    train_dir = runio.runs_dir(config["step"], name) / "train"
    columns, table, best_epoch = curves(read_rows(train_dir / "results.csv"))
    write_csv(run_dir / "curves.csv", columns, table)
    print(f"{name}: best epoch {best_epoch} of {len(table)}")
    return train_dir / "weights" / "best.pt", {"best_epoch": best_epoch, "epochs_run": len(table)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    name = config["name"]
    done = runio.repo_root() / "results" / config["step"] / name / "metrics.json"
    if done.is_file():
        print(f"{name}: already complete at {done}, skipping")
        return
    deadline = Deadline(config.get("hours_limit", 12))
    trains = config.get("train_on")
    photos, fabrics = load_photos(config, with_old=trains == "pilot_merged")
    datasets = ("pilot_test", trains) if trains else ("pilot_test",)
    built = build(photos, config, views_dir(), datasets, blocked_names(fabrics))
    test = [p for p in photos if p.source == "new" and p.split == "test"]
    run_dir = runio.start_run(config["step"], name, config)
    extra = {}
    if trains:
        found = trained_weights(config, deadline, run_dir, built[trains])
        if found is None:
            print(f"{name}: paused, checkpoint in {runio.runs_dir(config['step'], name)}")
            return
        weights, extra = found
        shutil.copy2(built[trains].parent / "manifest.csv", run_dir / "manifest.csv")
    else:
        weights = find_file(config["weights_data"], config["weights_file"])
    print(f"{name}: predicting {len(test)} pilot test images with {weights}")
    predictions = predict(weights, test, config["predict"], config["device"])
    runs = runio.runs_dir(config["step"], name)
    (runs / "test_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    names, mask, box, dropped = score(test, predictions, config)
    save_scored(run_dir, mask, names)
    (run_dir / "box").mkdir(exist_ok=True)
    save_scored(run_dir / "box", box, names)
    summary = {
        "arm": name,
        "trained_on": trains,
        "weights": str(weights),
        "weights_sha256": sha256_of(weights),
        **extra,
        "n_test_images": len(names),
        "n_test_fabrics": len({p.fabric for p in test}),
        "n_predictions": len(predictions),
        "gt_instances": {c: int((mask.gt_class == i).sum()) for i, c in enumerate(CLASSES)},
        "gt_without_pixels": dropped,
        "mask_ap50": point(mask),
        "box_ap50": point(box),
        "note": "point values only; intervals are computed by the pilot evaluation step",
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{name}: pilot test mask AP50 {summary['mask_ap50']['mean']}, wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
