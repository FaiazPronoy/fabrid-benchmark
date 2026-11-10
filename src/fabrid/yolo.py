import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fabrid import runio
from fabrid.labels import CLASSES, load_index, polygon_area
from fabrid.sessions import assign_sessions, check_disjoint, session_split

ARMS = ("session", "random")
INPUT_ROOT = Path("/kaggle/input")
CURVE_PREFIXES = ("train/", "metrics/", "val/")
FITNESS_KEYS = (
    ("metrics/mAP50(B)", 0.1),
    ("metrics/mAP50-95(B)", 0.9),
    ("metrics/mAP50(M)", 0.1),
    ("metrics/mAP50-95(M)", 0.9),
)


class Deadline:
    def __init__(self, hours: float, start: float | None = None):
        if start is None:
            start = float(os.environ.get("FABRID_JOB_START", time.time()))
        self.start = start
        self.limit = hours * 3600

    def passed(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) - self.start >= self.limit


def assert_unlocked(mapping: dict[str, str], locked: dict[str, str]) -> None:
    leaked = sorted(name for name in mapping if locked.get(name) == "test")
    if leaked:
        raise AssertionError(f"locked test images reached a training view: {leaked[:5]}")
    unknown = set(mapping.values()) - {"train", "val"}
    if unknown:
        raise AssertionError(f"view may only hold train and val, found {sorted(unknown)}")


def arm_mapping(records, arm: str, seed: int = 0, gap_seconds: float = 30) -> dict[str, str]:
    locked = session_split(records, gap_seconds)
    check_disjoint(locked, assign_sessions(records, gap_seconds))
    pool = sorted(name for name, split in locked.items() if split != "test")
    if arm == "session":
        mapping = {name: locked[name] for name in pool}
    elif arm == "random":
        n_train = sum(locked[name] == "train" for name in pool)
        order = np.random.default_rng(seed).permutation(len(pool))
        mapping = {
            pool[index]: "train" if rank < n_train else "val" for rank, index in enumerate(order)
        }
    else:
        raise ValueError(f"unknown arm {arm}, expected one of {ARMS}")
    assert_unlocked(mapping, locked)
    return mapping


def link(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def build_view(records, mapping: dict[str, str], view_dir: Path) -> Path:
    if view_dir.exists():
        shutil.rmtree(view_dir)
    for split in ("train", "val"):
        (view_dir / split / "images").mkdir(parents=True)
        (view_dir / split / "labels").mkdir(parents=True)
    for record in records:
        split = mapping.get(record.name)
        if split is None:
            continue
        link(record.path, view_dir / split / "images" / record.name)
        label = view_dir / split / "labels" / f"{Path(record.name).stem}.txt"
        if record.label_path is not None:
            link(record.label_path, label)
        else:
            label.write_text("", encoding="utf-8")
    data = {
        "path": str(view_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(CLASSES)),
    }
    data_yaml = view_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def verify_view(view_dir: Path, mapping: dict[str, str], locked: dict[str, str]) -> None:
    data = yaml.safe_load((view_dir / "data.yaml").read_text(encoding="utf-8"))
    if "test" in data:
        raise AssertionError("data.yaml must not name a test split")
    found = {}
    for split in ("train", "val"):
        for path in (view_dir / split / "images").iterdir():
            found[path.name] = split
    if found != mapping:
        raise AssertionError("view contents differ from the arm mapping")
    assert_unlocked(found, locked)


def read_rows(results_csv: Path) -> list[dict[str, float]]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    by_epoch = {int(row["epoch"]): row for row in rows}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def fitness(row: dict[str, float]) -> float:
    return sum(weight * row[key] for key, weight in FITNESS_KEYS)


def curves(rows: list[dict[str, float]]) -> tuple[list[str], list[list[float]], int]:
    columns = ["epoch", *[k for k in rows[0] if k.startswith(CURVE_PREFIXES)], "fitness"]
    table = [[*[row[k] for k in columns[:-1]], fitness(row)] for row in rows]
    best = max(range(len(rows)), key=lambda i: (table[i][-1], -i))
    return columns, table, int(rows[best]["epoch"])


def resumable_epoch(path: Path) -> int | None:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    epoch = checkpoint.get("epoch", -1)
    if epoch is None or epoch < 0 or checkpoint.get("optimizer") is None:
        return None
    return int(epoch)


def find_checkpoint(name: str, roots, epoch_of=resumable_epoch) -> Path | None:
    found = []
    for root in roots:
        if not Path(root).is_dir():
            continue
        for path in Path(root).glob(f"**/baselines/{name}/train/weights/*.pt"):
            if path.name not in ("last.pt", "resume.pt"):
                continue
            epoch = epoch_of(path)
            if epoch is not None:
                found.append((epoch, path.stat().st_mtime, path))
    return max(found, key=lambda item: item[:2])[2] if found else None


def find_finished(name: str, roots) -> Path | None:
    for root in roots:
        if not Path(root).is_dir():
            continue
        for marker in sorted(Path(root).glob(f"**/baselines/{name}/train/finished.json")):
            if (marker.parent / "weights" / "best.pt").is_file():
                return marker.parent
    return None


def adopt(train_dir: Path, source_dir: Path) -> None:
    if source_dir.resolve() != train_dir.resolve():
        shutil.copytree(source_dir, train_dir, dirs_exist_ok=True)


def disable_clahe() -> None:
    from ultralytics.data import augment

    original = augment.Albumentations.__init__
    if getattr(original, "clahe_off", False):
        return

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        transform = getattr(self, "transform", None)
        for item in getattr(transform, "transforms", None) or []:
            if type(item).__name__ == "CLAHE":
                item.p = 0.0
                print("albumentations CLAHE switched off (p=0)")

    patched.clahe_off = True
    augment.Albumentations.__init__ = patched


def train(config: dict, name: str, data_yaml: Path, deadline: Deadline) -> str:
    from ultralytics import YOLO

    runs = runio.runs_dir(config["step"], name)
    train_dir = runs / "train"
    roots = [runs.parent.parent, INPUT_ROOT]
    finished = find_finished(name, roots)
    if finished is not None:
        adopt(train_dir, finished)
        print(f"{name}: training already finished in {finished}, reusing it")
        return "finished"
    checkpoint = find_checkpoint(name, roots)
    state = {"paused": False}

    def on_fit_epoch_end(trainer):
        if trainer.stop or not deadline.passed():
            return
        shutil.copy2(trainer.last, Path(trainer.last).with_name("resume.pt"))
        state["paused"] = True
        trainer.stop = True
        print(f"{name}: wall clock limit reached after epoch {trainer.epoch + 1}, state saved")

    disable_clahe()
    if checkpoint is not None:
        adopt(train_dir, checkpoint.parent.parent)
        last = train_dir / "weights" / "last.pt"
        local = train_dir / "weights" / checkpoint.name
        if local != last:
            shutil.copy2(local, last)
        print(f"{name}: resuming from {checkpoint}")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(resume=True, data=str(data_yaml))
    else:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        print(f"{name}: training from {config['weights']}")
        model = YOLO(config["weights"])
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=str(data_yaml),
            epochs=config["epochs"],
            patience=config["patience"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            seed=config["seed"],
            deterministic=True,
            pretrained=True,
            workers=config["workers"],
            device=config["device"],
            project=str(runs),
            name="train",
            exist_ok=True,
        )
    if state["paused"]:
        return "paused"
    (train_dir / "weights" / "resume.pt").unlink(missing_ok=True)
    (train_dir / "finished.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
    return "finished"


def ground_truth(records, mapping: dict[str, str], offset: int) -> dict:
    images, annotations = [], []
    for record in records:
        if mapping.get(record.name) != "val":
            continue
        with Image.open(record.path) as image:
            width, height = image.size
        image_id = Path(record.name).stem
        images.append({"id": image_id, "file_name": record.name, "width": width, "height": height})
        for class_id, points in record.polygons:
            pixels = points * [width, height]
            x0, y0 = pixels.min(axis=0)
            x1, y1 = pixels.max(axis=0)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": class_id + offset,
                    "segmentation": [pixels.round(2).ravel().tolist()],
                    "area": round(polygon_area(pixels), 2),
                    "bbox": [round(float(v), 2) for v in (x0, y0, x1 - x0, y1 - y0)],
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i + offset, "name": n} for i, n in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def summarize(metrics) -> dict:
    out = {}
    names = metrics.names
    for key, part in (("box", metrics.box), ("mask", metrics.seg)):
        overall = {"P": part.mp, "R": part.mr, "AP50": part.map50, "AP50-95": part.map}
        per_class = {}
        for index, class_id in enumerate(metrics.ap_class_index):
            p, r, ap50, ap = part.class_result(index)
            per_class[names[int(class_id)]] = {"P": p, "R": r, "AP50": ap50, "AP50-95": ap}
        out[key] = {
            "overall": {k: float(v) for k, v in overall.items()},
            "per_class": {c: {k: float(v) for k, v in m.items()} for c, m in per_class.items()},
        }
    return out


def write_csv(path: Path, columns: list[str], table) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(table)


def finalize(config, arm, name, records, mapping, data_yaml, run_dir) -> dict:
    from ultralytics import YOLO

    runs = runio.runs_dir(config["step"], name)
    train_dir = runs / "train"
    columns, table, best_epoch = curves(read_rows(train_dir / "results.csv"))
    write_csv(run_dir / "curves.csv", columns, table)
    split_rows = sorted(mapping.items())
    write_csv(run_dir / "split.csv", ["name", "split"], split_rows)
    predictions = runs / "val" / "predictions.json"
    predictions.unlink(missing_ok=True)
    model = YOLO(str(train_dir / "weights" / "best.pt"))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        save_json=True,
        plots=False,
        project=str(runs),
        name="val",
        exist_ok=True,
    )
    offset = 1
    if predictions.is_file():
        shutil.copy2(predictions, runs / "val_predictions.json")
        listed = json.loads(predictions.read_text(encoding="utf-8"))
        offset = 0 if any(item["category_id"] == 0 for item in listed) else 1
    truth = ground_truth(records, mapping, offset)
    (runs / "val_ground_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    best_row = table[[int(row[0]) for row in table].index(best_epoch)]
    summary = {
        "model": config["model"],
        "arm": arm,
        "seed": config["seed"],
        "split": "val",
        "best_epoch": best_epoch,
        "epochs_run": len(table),
        "n_train": sum(v == "train" for v in mapping.values()),
        "n_val": sum(v == "val" for v in mapping.values()),
        **summarize(metrics),
        "training_curve_at_best": dict(zip(columns, best_row, strict=True)),
        "predictions": str(runs / "val_predictions.json") if predictions.is_file() else None,
        "ground_truth": str(runs / "val_ground_truth.json"),
        "coco_category_offset": offset,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    deadline = Deadline(config["hours_limit"])
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    for arm in config["arms"]:
        name = f"{config['model']}_{arm}_seed{config['seed']}"
        done = runio.repo_root() / "results" / config["step"] / name / "metrics.json"
        if done.is_file():
            print(f"{name}: already complete at {done}, skipping")
            continue
        if deadline.passed():
            print(f"{name}: wall clock limit reached before start, rerun to continue")
            return
        mapping = arm_mapping(records, arm, config["seed"], config["gap_seconds"])
        locked = session_split(records, config["gap_seconds"])
        view_dir = Path(os.environ.get("FABRID_VIEWS", "/tmp/fabrid_views")) / name
        data_yaml = build_view(records, mapping, view_dir)
        verify_view(view_dir, mapping, locked)
        counts = {s: sum(v == s for v in mapping.values()) for s in ("train", "val")}
        print(f"{name}: view {view_dir} with {counts['train']} train, {counts['val']} val")
        run_dir = runio.start_run(config["step"], name, {**config, "arm": arm})
        if train(config, name, data_yaml, deadline) == "paused":
            print(f"{name}: paused, checkpoint in {runio.runs_dir(config['step'], name)}")
            return
        summary = finalize(config, arm, name, records, mapping, data_yaml, run_dir)
        mask = summary["mask"]["overall"]["AP50"]
        print(f"{name}: best epoch {summary['best_epoch']}, val mask AP50 {mask:.4f}")
        print(f"{name}: wrote {run_dir}")
        if config.get("publish", True):
            runio.publish(run_dir)


if __name__ == "__main__":
    main()
