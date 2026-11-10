from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from conftest import make_dataset  # noqa: E402
from fabrid import augment, maskrcnn  # noqa: E402
from fabrid.labels import load_index  # noqa: E402

AUG = {"jitter": [0.6, 1.4], "hue": 0.05, "scale": [0.6, 1.4], "rotation": 15}
PASTE = {
    "probability": 1.0,
    "count": [1, 3],
    "scale": [0.75, 1.25],
    "flip": 0.5,
    "sigma": 1.5,
    "margin": 5,
}
SQUARE = "0 0.2 0.2 0.5 0.2 0.5 0.5 0.2 0.5"
BAR = "1 0.55 0.1 0.9 0.1 0.9 0.2 0.55 0.2"


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) / 64


def shapes_target():
    polygons = [(0, box(10, 10, 30, 30)), (2, box(36, 8, 60, 11)), (3, box(40, 40, 41, 41))]
    target, _ = maskrcnn.build_target(polygons, 64, 64)
    return target


def consistent(target: dict) -> None:
    masks = target["masks"]
    assert len(masks) == len(target["boxes"]) == len(target["labels"])
    assert (masks.sum(axis=(1, 2)) >= 4).all()
    np.testing.assert_array_equal(target["boxes"], maskrcnn.boxes_from_masks(masks))


def test_augment_keeps_masks_and_boxes_consistent():
    record = SimpleNamespace(name="IMG_20260501_090000.jpg")
    target = shapes_target()
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    for epoch in range(5):
        out, moved = augment.Augment(AUG, 0, 4)(pixels, target, record, epoch)
        assert out.shape == pixels.shape
        assert out.dtype == np.uint8
        consistent(moved)
        assert 4 not in moved["labels"].tolist()


def test_geometry_moves_pixels_and_masks_together():
    record = SimpleNamespace(name="IMG_20260501_090000.jpg")
    target = shapes_target()
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[target["masks"][0] > 0] = 255
    settings = {**AUG, "jitter": [1.0, 1.0], "hue": 0.0}
    for epoch in range(5):
        out, moved = augment.Augment(settings, 0, 4)(pixels, target, record, epoch)
        square = moved["masks"][moved["labels"] == 1][0] > 0
        assert out[..., 0][square].mean() > 200
        assert out[..., 0][~square].mean() < 60


def test_keep_instances_drops_masks_under_four_pixels():
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[1, 5, 5:8] = 1
    kept = augment.keep_instances(masks, np.array([1, 2]), 4)
    assert kept["labels"].tolist() == [1]
    assert kept["boxes"].tolist() == [[1, 1, 3, 3]]
    empty = augment.keep_instances(np.zeros((0, 8, 8), np.uint8), np.zeros(0, np.int64), 4)
    assert empty["boxes"].shape == (0, 4)


def test_pasted_instance_removes_covered_pixels():
    pixels = np.full((20, 20, 3), 100, dtype=np.uint8)
    below = np.zeros((20, 20), dtype=np.uint8)
    below[2:18, 2:18] = 1
    masks = [below]
    piece = np.zeros((6, 6), dtype=np.uint8)
    piece[1:5, 1:5] = 1
    patch = np.full((6, 6, 3), 200, dtype=np.float32)
    augment.paste_instance(pixels, masks, patch, piece, 4, 5, 1.5)
    assert len(masks) == 2
    assert masks[1][5:9, 6:10].all()
    assert masks[1].sum() == 16
    assert masks[0][5:9, 6:10].sum() == 0
    assert masks[0].sum() == 16 * 16 - 16
    assert (pixels[5:9, 6:10] == 100).all()


def sessions_dataset(tmp_path):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260501_090010_jpg.rf.b.jpg", [SQUARE], 2),
        ("train", "IMG_20260502_090000_jpg.rf.c.jpg", [BAR], 3),
        ("train", "IMG_20260503_090000_jpg.rf.d.jpg", [BAR], 4),
    ]
    records = load_index(make_dataset(tmp_path, items))
    sessions = {r.name: int(r.name[10:12]) for r in records}
    mapping = {r.name: "train" for r in records}
    mapping[records[3].name] = "val"
    locked = dict(mapping)
    return records, sessions, mapping, locked


def test_paste_donors_come_from_other_training_sessions(tmp_path):
    records, sessions, mapping, locked = sessions_dataset(tmp_path)
    train = records[:3]
    paste = augment.Paste(train, sessions, mapping, locked, PASTE, 0, 4)
    donors = {donor.name for donor, _ in paste.candidates(records[0].name)}
    assert donors == {records[2].name}
    assert {donor.name for donor, _ in paste.candidates(records[2].name)} == {
        records[0].name,
        records[1].name,
    }
    with pytest.raises(AssertionError):
        paste.candidates(records[3].name)
    with pytest.raises(AssertionError):
        augment.Paste(records, sessions, mapping, locked, PASTE, 0, 4)
    with pytest.raises(AssertionError):
        augment.Paste(train, sessions, mapping, {**locked, records[2].name: "test"}, PASTE, 0, 4)
    data = maskrcnn.Samples(train[:1])
    for epoch in range(4):
        pixels, target = data.load(0)
        pasted, result = paste(pixels, target, train[0], epoch)
        consistent(result)
        labels = result["labels"].tolist()
        assert 2 in labels
        assert set(labels) <= {1, 2}


def test_transforms_are_reproducible(tmp_path):
    records, sessions, mapping, locked = sessions_dataset(tmp_path)
    config = {
        "remedies": {"aug": True, "paste": True},
        "aug": AUG,
        "paste": PASTE,
        "seed": 0,
        "min_pixels": 4,
    }
    chain = augment.build_transform(config, records[:3], mapping, locked, sessions)
    pixels, target = maskrcnn.Samples(records[:1]).load(0)
    first = chain(pixels, target, records[0], 3)
    second = chain(pixels, target, records[0], 3)
    other = chain(pixels, target, records[0], 4)
    np.testing.assert_array_equal(first[0], second[0])
    for key in ("boxes", "labels", "masks"):
        np.testing.assert_array_equal(first[1][key], second[1][key])
    assert not np.array_equal(first[0], other[0])
    off = {**config, "remedies": {"aug": False, "paste": False}}
    assert augment.build_transform(off, records[:3], mapping, locked, sessions) is None


def mark_left_column(pixels, target, record, epoch):
    marked = pixels.copy()
    marked[:, 0] = 255
    return marked, target


def test_transform_runs_before_flips_and_only_where_given(tmp_path):
    items = [("train", "IMG_20260501_090000_jpg.rf.a.jpg", [], 1)]
    records = load_index(make_dataset(tmp_path, items))
    flipped = maskrcnn.Samples(records, (1.0, 0.0), transform=mark_left_column)
    assert (flipped.load(0)[0][:, -1] == 255).all()
    plain = maskrcnn.Samples(records)
    assert not (plain.load(0)[0][:, 0] == 255).all()


def test_last_epoch_summary():
    state = {
        "history": [
            {"epoch": 1, "loss": 1.2, "mask_AP50": 0.1},
            {"epoch": 2, "loss": 0.9, "mask_AP50": 0.2},
        ]
    }
    summary = augment.last_epoch(state, {"aug": True, "paste": False})
    assert summary["last_epoch"] == {"epoch": 2, "train_loss": 0.9, "val_mask_AP50": 0.2}
