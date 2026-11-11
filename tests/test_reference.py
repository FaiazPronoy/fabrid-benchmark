import json
import random

import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import reference
from fabrid.labels import CLASSES, load_index


def line(class_id: int) -> str:
    return f"{class_id}{SQUARE[1:]}"


VAL = (
    ("a", [0, 1]),
    ("b", [0]),
    ("c", [0]),
    ("d", [1]),
    ("e", [1]),
    ("f", [1]),
    ("g", [2]),
    ("i", [2]),
    ("j", [2]),
    ("k", [3]),
    ("l", [3]),
    ("m", [3]),
    ("n", []),
    ("o", []),
    ("p", []),
    ("q", [0]),
)


def image_name(index: int, key: str) -> str:
    return f"IMG_20260501_09{index:02d}00_jpg.rf.{key}.jpg"


def synthetic(tmp_path):
    items = [
        ("train", image_name(i, key), [line(c) for c in classes], i + 1)
        for i, (key, classes) in enumerate(VAL)
    ]
    items.append(("train", "IMG_20260501_080000_jpg.rf.t.jpg", [line(c) for c in range(4)], 90))
    items.append(("train", "IMG_20260501_070000_jpg.rf.h.jpg", [line(2)], 91))
    records = load_index(make_dataset(tmp_path / "data", items))
    locked = {r.name: "val" for r in records}
    locked["IMG_20260501_080000_jpg.rf.t.jpg"] = "test"
    locked["IMG_20260501_070000_jpg.rf.h.jpg"] = "train"
    return records, locked


def keys(chosen) -> list[str]:
    return [r.name.split(".rf.")[1][0] for r in chosen]


def inflate(records, key: str) -> None:
    record = next(r for r in records if f".rf.{key}." in r.name)
    with record.path.open("ab") as handle:
        handle.write(b"0" * 50_000)


def test_selection_count_order_and_no_reuse(tmp_path):
    records, locked = synthetic(tmp_path)
    chosen, skipped = reference.select_images(records, locked)
    assert keys(chosen) == list("abcdefgijklmno")
    assert len(chosen) == 14
    assert len({r.name for r in chosen}) == 14
    assert skipped == []
    shuffled = records[:]
    random.Random(3).shuffle(shuffled)
    assert reference.select_images(shuffled, locked)[0] == chosen
    for position, record in enumerate(chosen[:12]):
        assert position // 3 in {c for c, _ in record.polygons}
    assert all(r.n_lines == 0 for r in chosen[12:])
    fewer, _ = reference.select_images(records, locked, per_class=2)
    assert keys(fewer) == list("abdegiklno")


def test_large_images_are_skipped(tmp_path):
    records, locked = synthetic(tmp_path)
    inflate(records, "b")
    inflate(records, "n")
    chosen, skipped = reference.select_images(records, locked, limit=20_000)
    assert keys(chosen) == list("acqdefgijklmop")
    assert [name.split(".rf.")[1][0] for name in skipped] == ["b", "n"]
    assert all(r.path.stat().st_size <= 20_000 for r in chosen)


def test_test_image_refused(tmp_path):
    records, locked = synthetic(tmp_path)
    chosen, _ = reference.select_images(records, locked)
    assert "IMG_20260501_080000_jpg.rf.t.jpg" not in {r.name for r in chosen}
    assert "IMG_20260501_070000_jpg.rf.h.jpg" not in {r.name for r in chosen}
    with pytest.raises(AssertionError, match="val part"):
        reference.refuse_locked(["IMG_20260501_080000_jpg.rf.t.jpg"], locked)
    listed = {r.name: "val" for r in records}
    listed[image_name(2, "c")] = "train"
    with pytest.raises(AssertionError, match="split.csv"):
        reference.select_images(records, locked, listed)
    no_stain = {**locked, **{image_name(i, k): "test" for i, k in ((9, "k"), (10, "l"))}}
    with pytest.raises(ValueError, match="stain"):
        reference.select_images(records, no_stain)
    with pytest.raises(ValueError, match="hole"):
        reference.select_images(records, locked, per_class=4)


def test_label_mapping_for_both_kinds():
    assert [reference.label_name(i, "yolo") for i in range(4)] == list(CLASSES)
    assert [reference.label_name(i, "hybrid") for i in range(1, 5)] == list(CLASSES)
    for index, kind in ((4, "yolo"), (-1, "yolo"), (0, "hybrid"), (5, "hybrid"), (1, "other")):
        with pytest.raises(ValueError):
            reference.label_name(index, kind)


def test_json_writer(tmp_path):
    records, locked = synthetic(tmp_path)
    chosen, _ = reference.select_images(records, locked)
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
