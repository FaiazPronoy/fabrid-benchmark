import json
import random

import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import reference
from fabrid.labels import CLASSES, load_index


def line(class_id: int) -> str:
    return f"{class_id}{SQUARE[1:]}"


def synthetic(tmp_path):
    items = [
        ("train", "IMG_20260501_080000_jpg.rf.t.jpg", [line(c) for c in range(4)], 1),
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [line(0), line(1)], 2),
        ("train", "IMG_20260501_090100_jpg.rf.b.jpg", [line(1)], 3),
        ("train", "IMG_20260501_090200_jpg.rf.c.jpg", [line(2)], 4),
        ("train", "IMG_20260501_090300_jpg.rf.d.jpg", [line(3), line(3)], 5),
        ("train", "IMG_20260501_090400_jpg.rf.e.jpg", [], 6),
        ("train", "IMG_20260501_090500_jpg.rf.f.jpg", [], 7),
        ("train", "IMG_20260501_090600_jpg.rf.g.jpg", [], 8),
        ("train", "IMG_20260501_070000_jpg.rf.h.jpg", [line(2)], 9),
    ]
    records = load_index(make_dataset(tmp_path / "data", items))
    locked = {r.name: "val" for r in records}
    locked["IMG_20260501_080000_jpg.rf.t.jpg"] = "test"
    locked["IMG_20260501_070000_jpg.rf.h.jpg"] = "train"
    return records, locked


def test_selection_covers_classes_and_skips_locked(tmp_path):
    records, locked = synthetic(tmp_path)
    chosen = [r.name for r in reference.select_images(records, locked)]
    assert chosen == [
        "IMG_20260501_090000_jpg.rf.a.jpg",
        "IMG_20260501_090100_jpg.rf.b.jpg",
        "IMG_20260501_090200_jpg.rf.c.jpg",
        "IMG_20260501_090300_jpg.rf.d.jpg",
        "IMG_20260501_090400_jpg.rf.e.jpg",
        "IMG_20260501_090500_jpg.rf.f.jpg",
    ]
    shuffled = records[:]
    random.Random(3).shuffle(shuffled)
    assert [r.name for r in reference.select_images(shuffled, locked)] == chosen
    classes = [{c for c, _ in r.polygons} for r in reference.select_images(records, locked)[:4]]
    assert all(k in found for k, found in enumerate(classes))


def test_test_image_refused(tmp_path):
    records, locked = synthetic(tmp_path)
    with pytest.raises(AssertionError, match="val part"):
        reference.refuse_locked(["IMG_20260501_080000_jpg.rf.t.jpg"], locked)
    listed = {r.name: "val" for r in records}
    listed["IMG_20260501_090200_jpg.rf.c.jpg"] = "train"
    with pytest.raises(AssertionError, match="split.csv"):
        reference.select_images(records, locked, listed)
    no_stain = {**locked, "IMG_20260501_090300_jpg.rf.d.jpg": "test"}
    with pytest.raises(ValueError, match="stain"):
        reference.select_images(records, no_stain)


def test_label_mapping_for_both_kinds():
    assert [reference.label_name(i, "yolo") for i in range(4)] == list(CLASSES)
    assert [reference.label_name(i, "hybrid") for i in range(1, 5)] == list(CLASSES)
    for index, kind in ((4, "yolo"), (-1, "yolo"), (0, "hybrid"), (5, "hybrid"), (1, "other")):
        with pytest.raises(ValueError):
            reference.label_name(index, kind)


def test_json_writer(tmp_path):
    records, locked = synthetic(tmp_path)
    chosen = reference.select_images(records, locked)
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:6, 3:8] = True

    def predict(path):
        return [
            reference.detection("hole", 0.04, [1, 1, 2, 2], mask),
            reference.detection("stain", 0.9, [3.123, 2, 8, 6.456], mask),
            reference.detection("needle mark", 0.05, [0, 0, 4, 4], mask[:0]),
        ]

    weights = tmp_path / "best.pth"
    weights.write_bytes(b"weights")
    data = reference.pack("demo", reference.sha256_of(weights), 0.05, chosen, predict)
    out = tmp_path / "out"
    out.mkdir()
    copies = reference.write_pack(out, data, chosen)
    saved = json.loads((out / "detections.json").read_text(encoding="utf-8"))
    assert saved["model_key"] == "demo"
    assert saved["min_score"] == 0.05
    assert len(saved["weights_sha256"]) == 64
    assert [i["file"] for i in saved["images"]] == [r.name for r in chosen]
    first = saved["images"][0]
    assert (first["width"], first["height"]) == (32, 32)
    assert first["detections"] == [
        {"label": "stain", "score": 0.9, "box": [3.12, 2.0, 8.0, 6.46], "mask_area": 20},
        {"label": "needle mark", "score": 0.05, "box": [0.0, 0.0, 4.0, 4.0], "mask_area": 0},
    ]
    assert sorted(p.name for p in (out / "images").iterdir()) == sorted(r.name for r in chosen)
    assert all(p.read_bytes() == r.path.read_bytes() for p, r in zip(copies, chosen, strict=True))
