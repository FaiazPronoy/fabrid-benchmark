from datetime import datetime

import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import labels


def test_load_index_finds_nested_root_and_parses(tmp_path, monkeypatch):
    base = tmp_path / "mount" / "fabrid"
    make_dataset(
        base,
        [
            ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE, "3 0.5 0.5 0.6 0.5 0.6 0.6"], 1),
            ("valid", "IMG_20260501_090010_jpg.rf.b.jpg", [], 2),
            ("test", "IMG_20260502_100000_jpg.rf.c.jpg", ["1 0.1 0.1 0.2 0.2", "x y z"], 3),
            ("test", "IMG_20260502_100005_jpg.rf.d.jpg", ["2 0.1 0.1 0.2 0.2 0.3 0.3"], 4),
        ],
    )
    monkeypatch.setenv("FABRID_DATA", str(tmp_path))
    records = labels.load_index()
    by_name = {r.name: r for r in records}
    first = by_name["IMG_20260501_090000_jpg.rf.a.jpg"]
    assert first.stem == "IMG_20260501_090000"
    assert first.timestamp == datetime(2026, 5, 1, 9, 0, 0)
    assert first.folder == "train"
    assert (first.n_lines, first.n_valid) == (2, 2)
    assert [c for c, _ in first.polygons] == [0, 3]
    empty = by_name["IMG_20260501_090010_jpg.rf.b.jpg"]
    assert (empty.n_lines, empty.n_valid, empty.polygons) == (0, 0, [])
    bad = by_name["IMG_20260502_100000_jpg.rf.c.jpg"]
    assert (bad.n_lines, bad.n_valid) == (2, 0)
    collinear = by_name["IMG_20260502_100005_jpg.rf.d.jpg"]
    assert (collinear.n_lines, collinear.n_valid) == (1, 0)


def test_data_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        labels.data_root(tmp_path)


def test_bad_name_rejected():
    with pytest.raises(ValueError):
        labels.parse_timestamp("photo_1.jpg")


def test_parse_line_rules():
    assert labels.parse_line("4 0 0 1 0 1 1") is None
    assert labels.parse_line("0 0 0 1 0 1") is None
    assert labels.parse_line("0 0 0 1 nan 1 1") is None
    class_id, points = labels.parse_line(SQUARE)
    assert class_id == 0 and points.shape == (4, 2)
    assert labels.polygon_area(points) == pytest.approx(0.04)
    assert not labels.is_valid(np.array([[0, 0], [1, 1]]))
