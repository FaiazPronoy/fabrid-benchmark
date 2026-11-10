import csv
from collections import Counter
from pathlib import Path

import pytest
import yaml

from conftest import SQUARE, make_dataset
from fabrid import yolo
from fabrid.labels import load_index
from fabrid.sessions import session_split


def test_arms_never_hold_locked_test(records):
    locked = session_split(records)
    session = yolo.arm_mapping(records, "session")
    random = yolo.arm_mapping(records, "random")
    assert set(session) == set(random)
    assert not any(locked[name] == "test" for name in session)
    assert Counter(session.values()) == Counter(random.values())
    assert session != random
    assert random == yolo.arm_mapping(records, "random")
    with pytest.raises(ValueError):
        yolo.arm_mapping(records, "other")


def test_assert_unlocked_rejects_test_images():
    locked = {"a": "train", "b": "test"}
    with pytest.raises(AssertionError):
        yolo.assert_unlocked({"a": "train", "b": "val"}, locked)
    with pytest.raises(AssertionError):
        yolo.assert_unlocked({"a": "test"}, {"a": "train"})
    yolo.assert_unlocked({"a": "train"}, locked)


def test_view_has_only_train_and_val(tmp_path):
    items = []
    for day in range(1, 8):
        for frame in range(3):
            name = f"IMG_202605{day:02d}_09{frame:02d}00_jpg.rf.v{day}{frame}.jpg"
            items.append(("train", name, [SQUARE] if frame else [], day * 10 + frame))
    records = load_index(make_dataset(tmp_path / "data", items))
    locked = session_split(records)
    mapping = {n: s for n, s in locked.items() if s != "test"}
    view = tmp_path / "view"
    data_yaml = yolo.build_view(records, mapping, view)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert "test" not in data
    assert data["val"] == "val/images"
    yolo.verify_view(view, mapping, locked)
    test_name = next(n for n, s in locked.items() if s == "test")
    record = next(r for r in records if r.name == test_name)
    yolo.link(record.path, view / "val" / "images" / test_name)
    with pytest.raises(AssertionError):
        yolo.verify_view(view, {**mapping, test_name: "val"}, locked)


def test_best_epoch_from_curves(tmp_path):
    header = [
        "                  epoch",
        "train/seg_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "lr/pg0",
    ]
    rows = [
        [1, 2.0, 0.5, 0.2, 0.4, 0.1, 0.01],
        [2, 1.5, 0.6, 0.4, 0.5, 0.3, 0.01],
        [3, 1.2, 0.9, 0.3, 0.9, 0.2, 0.01],
        [2, 1.5, 0.6, 0.4, 0.5, 0.3, 0.01],
    ]
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    columns, table, best = yolo.curves(yolo.read_rows(path))
    assert best == 2
    assert len(table) == 3
    assert "lr/pg0" not in columns
    assert columns[-1] == "fitness"


def test_find_checkpoint_prefers_latest_resumable(tmp_path):
    epochs = {}
    for root, epoch in (("working", 4), ("input", 9)):
        weights = tmp_path / root / "runs" / "baselines" / "m_session_seed0" / "train" / "weights"
        weights.mkdir(parents=True)
        for name, value in (("last.pt", epoch), ("best.pt", 99), ("resume.pt", None)):
            (weights / name).write_bytes(b"0")
            epochs[weights / name] = value
    newest = tmp_path / "input" / "runs" / "baselines" / "m_session_seed0" / "train"
    roots = [tmp_path / "working", tmp_path / "input"]
    assert yolo.find_checkpoint("m_session_seed0", roots, epochs.get) == newest / "weights/last.pt"
    assert yolo.find_checkpoint("other", [tmp_path], epochs.get) is None


def test_deadline():
    deadline = yolo.Deadline(10.5, start=0.0)
    assert not deadline.passed(now=10.5 * 3600 - 1)
    assert deadline.passed(now=10.5 * 3600)


def test_ground_truth_uses_val_only(tmp_path):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260501_100000_jpg.rf.b.jpg", ["2 0 0 0.5 0 0.5 0.5"], 2),
    ]
    records = load_index(make_dataset(tmp_path, items))
    mapping = {records[0].name: "train", records[1].name: "val"}
    truth = yolo.ground_truth(records, mapping, offset=1)
    assert [i["file_name"] for i in truth["images"]] == [records[1].name]
    annotation = truth["annotations"][0]
    assert annotation["category_id"] == 3
    assert annotation["bbox"] == [0.0, 0.0, 16.0, 16.0]
    assert annotation["area"] == pytest.approx(128.0)


def test_train_args_allow_list():
    config = {"train_args": {"degrees": 15, "copy_paste": 0.5, "flipud": 0.5}}
    assert yolo.train_overrides(config) == config["train_args"]
    assert yolo.train_overrides({}) == {}
    assert yolo.train_overrides({"train_args": None}) == {}
    for key in ("epochs", "mosaic", "lr0", "data"):
        with pytest.raises(ValueError, match=key):
            yolo.train_overrides({"train_args": {"degrees": 1, key: 1}})


def test_run_name_with_and_without_tag():
    config = {"model": "yolov8m", "seed": 0}
    assert yolo.run_name(config, "session") == "yolov8m_session_seed0"
    assert yolo.run_name({**config, "tag": "aug"}, "session") == "yolov8m_aug_session_seed0"
    assert yolo.run_name({**config, "tag": None, "seed": 2}, "random") == "yolov8m_random_seed2"


def test_aug_config_matches_base_except_augmentation():
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = yaml.safe_load((configs / "yolov8m_seg.yaml").read_text(encoding="utf-8"))
    aug = yaml.safe_load((configs / "yolov8m_seg_aug.yaml").read_text(encoding="utf-8"))
    changed = ("tag", "arms", "train_args")
    assert {k: v for k, v in aug.items() if k not in changed} == {
        k: v for k, v in base.items() if k != "arms"
    }
    assert aug["arms"] == ["session"]
    assert yolo.run_name(aug, "session") == "yolov8m_aug_session_seed0"
    assert yolo.train_overrides(aug) == {
        "degrees": 15,
        "scale": 0.5,
        "hsv_v": 0.6,
        "hsv_s": 0.8,
        "flipud": 0.5,
        "copy_paste": 0.5,
    }
