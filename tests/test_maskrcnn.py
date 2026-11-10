import json
import os

import numpy as np
import pytest

from fabrid import maskrcnn


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) / 64


def test_target_keeps_one_pixel_line_and_drops_sliver():
    polygons = [(0, box(8, 8, 24, 24)), (2, box(4, 40, 60, 41)), (1, box(4, 50, 60, 50.2))]
    target, dropped = maskrcnn.build_target(polygons, 64, 64)
    assert dropped == 1
    assert target["labels"].tolist() == [1, 3]
    assert target["masks"].shape == (2, 64, 64)
    assert target["masks"][0].sum() == 256
    assert target["masks"][1].sum() == 56
    assert target["boxes"].tolist() == [[8, 8, 24, 24], [4, 40, 60, 41]]


def test_target_for_empty_image():
    target, dropped = maskrcnn.build_target([], 48, 64)
    assert dropped == 0
    assert target["masks"].shape == (0, 48, 64)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["boxes"].dtype == np.float32
    assert target["labels"].dtype == np.int64


def test_flip_moves_boxes_with_masks():
    target, _ = maskrcnn.build_target([(0, box(0, 0, 10, 20))], 64, 64)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    _, flipped = maskrcnn.flip(pixels, target, horizontal=True, vertical=True)
    assert flipped["boxes"].tolist() == [[54, 44, 64, 64]]
    assert flipped["masks"].sum() == target["masks"].sum()


def test_empty_image_gives_empty_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import make_dataset
    from fabrid.labels import load_index

    items = [("train", "IMG_20260501_090000_jpg.rf.a.jpg", [], 1)]
    records = load_index(make_dataset(tmp_path, items))
    image, target, index = maskrcnn.Samples(records, (1.0, 1.0))[0]
    assert index == 0
    assert image.shape == (3, 32, 32)
    assert target["masks"].shape == (0, 32, 32)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].dtype == torch.int64


def test_split_records_rejects_locked_test():
    records = [type("R", (), {"name": n})() for n in ("a", "b", "c")]
    locked = {"a": "train", "b": "val", "c": "test"}
    train, val = maskrcnn.split_records(records, {"a": "train", "b": "val"}, locked)
    assert [r.name for r in train] == ["a"]
    assert [r.name for r in val] == ["b"]
    with pytest.raises(AssertionError):
        maskrcnn.split_records(records, {"a": "train", "c": "val"}, locked)


def write_checkpoint(root, name, epoch, mtime):
    directory = root / "runs" / "baselines" / name / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "last.pth").write_bytes(b"0")
    (directory / "last.json").write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
    os.utime(directory / "last.pth", (mtime, mtime))
    return directory / "last.pth"


def test_resume_picks_newest_checkpoint(tmp_path):
    name = "maskrcnn_r50_session_seed0"
    write_checkpoint(tmp_path / "working", name, 7, 2000)
    newest = write_checkpoint(tmp_path / "input" / "a", name, 12, 1000)
    tied = write_checkpoint(tmp_path / "input" / "b", name, 12, 1500)
    broken = tmp_path / "input" / "c" / "runs" / "baselines" / name / "checkpoints"
    broken.mkdir(parents=True)
    (broken / "last.json").write_text(json.dumps({"epoch": 30}), encoding="utf-8")
    roots = [tmp_path / "working", tmp_path / "input", tmp_path / "missing"]
    assert maskrcnn.find_checkpoint(name, roots) == tied
    tied.unlink()
    assert maskrcnn.find_checkpoint(name, roots) == newest
    assert maskrcnn.find_checkpoint("other", roots) is None


def test_lr_schedule():
    assert maskrcnn.lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert maskrcnn.lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert maskrcnn.lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert maskrcnn.lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert maskrcnn.lr_factor(100, 10, 100) == pytest.approx(0.0)


def test_coco_scores_on_perfect_and_missing_predictions(tmp_path):
    pytest.importorskip("pycocotools")
    from conftest import make_dataset
    from fabrid.labels import load_index

    square = "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6"
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [square], 1),
        ("train", "IMG_20260501_090100_jpg.rf.b.jpg", [], 2),
    ]
    records = load_index(make_dataset(tmp_path, items))
    truth = maskrcnn.coco_truth(records)
    annotation = truth["annotations"][0]
    detection = {
        "image_id": annotation["image_id"],
        "category_id": 1,
        "bbox": annotation["bbox"],
        "score": 0.9,
        "segmentation": annotation["segmentation"],
    }
    scores = maskrcnn.coco_scores(truth, [detection])
    for kind in ("box", "mask"):
        assert scores[kind]["overall"]["AP50"] == pytest.approx(1.0)
        assert scores[kind]["per_class"]["needle mark"]["AP50-95"] == pytest.approx(1.0)
        assert scores[kind]["per_class"]["hole"]["AP50"] is None
    empty = maskrcnn.coco_scores(truth, [])
    assert empty["mask"]["overall"]["AP50"] == 0.0
    assert empty["mask"]["per_class"]["stain"]["AP50"] is None
