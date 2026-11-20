import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from fabrid import runio
from fabrid.labels import (
    CLASSES,
    EXPORT_FOLDERS,
    IMAGE_SUFFIXES,
    data_root,
    load_index,
    read_labels,
)
from fabrid.needles import NEEDLE, Cluster, merge_needles
from fabrid.yolo import arm_mapping, link, write_csv

DEFAULT_INPUT = "/kaggle/input"
SPLITS = ("train", "test", "excluded")
PHOTO = re.compile(r"IMG_(\d+)")
SUFFIX = " defect"
DATASETS = ("pilot_test", "pilot_new", "pilot_merged")
FOLDERS = {"pilot_test": ("test",), "pilot_new": ("train", "val"), "pilot_merged": ("train", "val")}
MANIFEST = ("image", "source", "split", "test_fabric", "needle_before", "needle_after")
NEEDLE_COLUMNS = (
    "image",
    "split",
    "polygons_before",
    "polygons_after",
    "needle_before",
    "needle_after",
)
TILE = 320


@dataclass
class Photo:
    name: str
    path: Path
    polygons: list[tuple[int, np.ndarray]]
    split: str
    fabric: str = ""
    source: str = "new"
    before: int = 0
    after: int = 0
    lines_before: int = 0
    label_path: Path | None = None
    merged: list[Cluster] = field(default_factory=list)


def class_index(name: str) -> int:
    key = " ".join(str(name).lower().split())
    if key.endswith(SUFFIX):
        key = key[: -len(SUFFIX)].strip()
    if key not in CLASSES:
        raise ValueError(f"unknown class name {name!r}, expected one of {list(CLASSES)}")
    return CLASSES.index(key)


def class_map(names) -> dict[int, int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {int(index): class_index(name) for index, name in items}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"two class names map to the same class: {names}")
    return mapping


def input_dir(slug: str, base: str | Path | None = None) -> Path:
    base = Path(base or os.environ.get("FABRID_INPUT", DEFAULT_INPUT))
    if base.name == slug and base.is_dir():
        return base
    found = [Path(path) for path, dirs, _ in os.walk(base) for d in dirs if d == slug]
    found = [path / slug for path in found]
    if not found:
        raise FileNotFoundError(f"input {slug} not found under {base}")
    return min(found, key=lambda p: (len(p.parts), str(p)))


def find_file(slug: str, file_name: str, base: str | Path | None = None) -> Path:
    found = sorted(input_dir(slug, base).rglob(file_name))
    if not found:
        raise FileNotFoundError(f"{file_name} not found in input {slug}")
    return found[0]


def read_fabrics(path: str | Path) -> dict[str, tuple[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image", "split", "test_fabric"]:
            raise ValueError(f"{path}: columns must be image,split,test_fabric")
        rows = list(reader)
    table = {}
    for row in rows:
        name, split, fabric = row["image"].strip(), row["split"].strip(), row["test_fabric"].strip()
        if name in table:
            raise ValueError(f"{path}: {name} is listed twice")
        if split not in SPLITS:
            raise ValueError(f"{path}: {name} has split {split!r}, expected one of {SPLITS}")
        if split == "test" and not fabric:
            raise ValueError(f"{path}: test image {name} names no fabric")
        table[name] = (split, fabric if split == "test" else "")
    return table


def check_coverage(names, fabrics: dict) -> None:
    missing = sorted(set(names) - set(fabrics))
    extra = sorted(set(fabrics) - set(names))
    if missing or extra:
        raise ValueError(
            f"fabric list and export differ: {len(missing)} export images not listed "
            f"{missing[:5]}, {len(extra)} listed images not in the export {extra[:5]}"
        )


def export_files(root: Path) -> dict[str, tuple[Path, Path | None]]:
    files = {}
    for folder in EXPORT_FOLDERS:
        image_dir = root / folder / "images"
        if not image_dir.is_dir():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name in files:
                raise ValueError(f"{path.name} appears in more than one export folder")
            label = root / folder / "labels" / f"{path.stem}.txt"
            files[path.name] = (path, label if label.is_file() else None)
    return files


def export_names(root: Path) -> list[str]:
    for path in (root / "data.yaml", root.parent / "data.yaml"):
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    raise FileNotFoundError(f"data.yaml not found next to {root}")


def load_new(root: Path, fabrics: dict, config: dict) -> list[Photo]:
    mapping = class_map(export_names(root))
    files = export_files(root)
    check_coverage(files, fabrics)
    photos = []
    for name in sorted(files):
        split, fabric = fabrics[name]
        if split == "excluded":
            continue
        path, label = files[name]
        raw, _ = read_labels(label)
        polygons = [(mapping[class_id], points) for class_id, points in raw]
        with Image.open(path) as image:
            width, height = image.size
        merged, found = merge_needles(
            polygons, width, height, config["needle_link"], config["needle_min_width"]
        )
        photos.append(
            Photo(
                name=name,
                path=path,
                polygons=merged,
                split=split,
                fabric=fabric,
                before=sum(c == NEEDLE for c, _ in polygons),
                after=sum(c == NEEDLE for c, _ in merged),
                lines_before=len(polygons),
                merged=found,
            )
        )
    return photos


def photo_key(name: str) -> tuple:
    match = PHOTO.search(name)
    return (0, int(match.group(1)), name) if match else (1, 0, name)


def validation_blocks(names, share: float, block: int, seed: int) -> dict[str, str]:
    ordered = sorted(names, key=photo_key)
    blocks = [ordered[i : i + block] for i in range(0, len(ordered), block)]
    if not blocks:
        return {}
    count = max(1, round(len(blocks) * share))
    picked = set(np.random.default_rng(seed).choice(len(blocks), count, replace=False).tolist())
    return {
        name: "val" if index in picked else "train"
        for index, members in enumerate(blocks)
        for name in members
    }


def load_old(root: Path, config: dict) -> list[Photo]:
    records = load_index(root)
    mapping = arm_mapping(records, "session", 0, config["gap_seconds"])
    photos = []
    for record in records:
        split = mapping.get(record.name)
        if split is None:
            continue
        needles = sum(c == NEEDLE for c, _ in record.polygons)
        photos.append(
            Photo(
                name=record.name,
                path=record.path,
                polygons=record.polygons,
                split=split,
                source="old",
                before=needles,
                after=needles,
                lines_before=len(record.polygons),
                label_path=record.label_path,
            )
        )
    return photos


def label_text(polygons) -> str:
    lines = [f"{c} " + " ".join(f"{v:.6f}" for v in points.ravel()) for c, points in polygons]
    return "\n".join(lines) + ("\n" if lines else "")


def write_dataset(out_dir: Path, entries: list[tuple[Photo, str]]) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    names = [photo.name for photo, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"{out_dir.name}: image names are not unique")
    folders = sorted({split for _, split in entries})
    for split in folders:
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)
    rows = []
    for photo, split in sorted(entries, key=lambda item: item[0].name):
        link(photo.path, out_dir / split / "images" / photo.name)
        label = out_dir / split / "labels" / f"{Path(photo.name).stem}.txt"
        if photo.source == "old" and photo.label_path is not None:
            link(photo.label_path, label)
        else:
            label.write_text(label_text(photo.polygons), encoding="utf-8")
        rows.append([photo.name, photo.source, split, photo.fabric, photo.before, photo.after])
    data = {"path": str(out_dir.resolve())}
    for split in folders:
        data[split] = f"{split}/images"
    data["names"] = dict(enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    write_csv(out_dir / "manifest.csv", list(MANIFEST), rows)
    return out_dir / "data.yaml"


def folder_names(out_dir: Path, split: str) -> set[str]:
    folder = out_dir / split / "images"
    return {path.name for path in folder.iterdir()} if folder.is_dir() else set()


def check_built(out_dir: Path, forbidden: set[str]) -> None:
    for split in ("train", "val"):
        leaked = sorted(folder_names(out_dir, split) & forbidden)
        if leaked:
            raise AssertionError(f"{out_dir.name}: blocked images in {split}: {leaked}")
    data = yaml.safe_load((out_dir / "data.yaml").read_text(encoding="utf-8"))
    if "train" in data and "test" in data:
        raise AssertionError(f"{out_dir.name}: a training dataset must not name a test split")


def assignments(photos: list[Photo], config: dict) -> dict[str, list[tuple[Photo, str]]]:
    test = [(p, "test") for p in photos if p.source == "new" and p.split == "test"]
    pool = [p for p in photos if p.source == "new" and p.split == "train"]
    blocks = validation_blocks(
        [p.name for p in pool], config["val_share"], config["val_block"], config["split_seed"]
    )
    new = [(p, blocks[p.name]) for p in pool]
    old = [(p, p.split) for p in photos if p.source == "old"]
    return {"pilot_test": test, "pilot_new": new, "pilot_merged": new + old}


def blocked_names(fabrics: dict) -> set[str]:
    return {name for name, (split, _) in fabrics.items() if split != "train"}


def build(photos: list[Photo], config: dict, base: Path, datasets, blocked) -> dict[str, Path]:
    forbidden = set(blocked) | {p.name for p in photos if p.source == "new" and p.split != "train"}
    names = [p.name for p in photos]
    if len(set(names)) != len(names):
        raise ValueError("an old and a new image share a file name")
    plan = assignments(photos, config)
    built = {}
    for dataset in datasets:
        if dataset == "pilot_merged" and not any(p.source == "old" for p in photos):
            raise ValueError("pilot_merged needs the old data")
        built[dataset] = write_dataset(base / dataset, plan[dataset])
        if dataset == "pilot_test":
            if set().union(*(folder_names(base / dataset, s) for s in ("train", "val"))):
                raise AssertionError("pilot_test holds train or val folders")
        else:
            check_built(base / dataset, forbidden)
    return built


def counts(photos: list[Photo], config: dict) -> dict:
    plan = assignments(photos, config)
    new = [p for p in photos if p.source == "new"]
    fabrics = sorted({p.fabric for p in new if p.split == "test"})
    out = {
        "new_images_used": len(new),
        "pilot_test": len(plan["pilot_test"]),
        "pilot_test_fabrics": {f: sum(p.fabric == f for p in new) for f in fabrics},
        "training_pool": len(plan["pilot_new"]),
        "val_block": config["val_block"],
        "val_share": config["val_share"],
        "split_seed": config["split_seed"],
    }
    block = config["val_block"]
    pool = sorted((p.name for p, _ in plan["pilot_new"]), key=photo_key)
    split_of = {p.name: s for p, s in plan["pilot_new"]}
    starts = pool[::block]
    out["blocks"] = len(starts)
    out["val_blocks"] = sum(split_of[name] == "val" for name in starts)
    for dataset in ("pilot_new", "pilot_merged"):
        out[dataset] = {
            source: {
                split: sum(p.source == source and s == split for p, s in plan[dataset])
                for split in ("train", "val")
            }
            for source in ("new", "old")
        }
    out["needle"] = {
        "images_with_needle_marks": sum(p.before > 0 for p in new),
        "polygons_before": sum(p.before for p in new),
        "polygons_after": sum(p.after for p in new),
        "clusters_merged": sum(len(p.merged) for p in new),
        "images_changed": sum(bool(p.merged) for p in new),
    }
    out["instances_after_merge"] = {
        split: {
            name: sum(c == index for p in new if p.split == split for c, _ in p.polygons)
            for index, name in enumerate(CLASSES)
        }
        for split in ("train", "test")
    }
    return out


def needle_rows(photos: list[Photo]) -> list[list]:
    return [
        [p.name, p.split, p.lines_before, len(p.polygons), p.before, p.after]
        for p in sorted(photos, key=lambda photo: photo.name)
        if p.source == "new"
    ]


def contact_sheet(photos: list[Photo], path: Path, limit: int, columns: int = 6) -> int:
    chosen = [
        (photo, cluster)
        for photo in sorted(photos, key=lambda p: p.name)
        if photo.source == "new"
        for cluster in photo.merged
    ][:limit]
    if not chosen:
        return 0
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * TILE, rows * TILE), "white")
    for index, (photo, cluster) in enumerate(chosen):
        with Image.open(photo.path) as source:
            image = source.convert("RGB")
        scale = np.array(image.size, dtype=np.float64)
        corners = cluster.merged * scale
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        side = max(x1 - x0, y1 - y0) * 1.5 + 64
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        draw = ImageDraw.Draw(image)
        for member in cluster.members:
            draw.polygon([tuple(p) for p in member * scale], outline=(255, 220, 0))
        thick = max(2, int(side / 160))
        draw.polygon([tuple(p) for p in corners], outline=(0, 255, 255), width=thick)
        tile = image.crop(tuple(int(round(v)) for v in box)).resize((TILE, TILE))
        ImageDraw.Draw(tile).text((4, 4), photo.name[:40], fill=(255, 255, 255))
        sheet.paste(tile, ((index % columns) * TILE, (index // columns) * TILE))
    sheet.save(path, quality=90)
    return len(chosen)


def views_dir() -> Path:
    return Path(os.environ.get("FABRID_VIEWS", "/tmp/fabrid_views")) / "pilot"


def load_photos(config: dict, with_old: bool) -> tuple[list[Photo], dict]:
    fabrics = read_fabrics(runio.repo_root() / config["fabrics"])
    new_root = data_root(input_dir(config["new_data"]))
    photos = load_new(new_root, fabrics, config)
    print(f"new data {new_root}: {len(photos)} images used, {len(fabrics) - len(photos)} excluded")
    if with_old:
        old_root = data_root(input_dir(config["old_data"]))
        old = load_old(old_root, config)
        print(f"old data {old_root}: {len(old)} images from the session split train and val parts")
        photos += old
    return photos, fabrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    photos, fabrics = load_photos(config, with_old=True)
    built = build(photos, config, views_dir(), DATASETS, blocked_names(fabrics))
    run_dir = runio.start_run(config["step"], config["name"], config)
    summary = counts(photos, config)
    (run_dir / "counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(run_dir / "needles.csv", list(NEEDLE_COLUMNS), needle_rows(photos))
    for dataset, data_yaml in built.items():
        shutil.copy2(data_yaml.parent / "manifest.csv", run_dir / f"manifest_{dataset}.csv")
        print(f"{dataset}: {data_yaml}")
    sheet = runio.runs_dir(config["step"], config["name"]) / "needles.jpg"
    shown = contact_sheet(photos, sheet, config["sheet_limit"])
    print(f"needle contact sheet with {shown} merged clusters: {sheet}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
