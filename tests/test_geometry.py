import numpy as np
import pytest

from conftest import make_dataset
from fabrid import geometry
from fabrid.labels import load_index


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def test_rasterize_fills_pixel_centres():
    mask = geometry.rasterize(box(10, 10, 50, 50), 64, 64)
    rows, cols = np.nonzero(mask)
    assert mask.sum() == 1600
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (10, 49, 10, 49)
    assert not geometry.rasterize(box(5, 5, 40, 5.2), 64, 64).any()
    assert geometry.rasterize(box(-5, -5, 3, 3), 8, 8).sum() == 9


def test_three_pixel_line():
    row = geometry.describe(1, box(10, 20, 70, 23) / 100, 100, 100)
    assert row["area_px"] == 180
    assert (row["bbox_w"], row["bbox_h"]) == (60, 3)
    assert row["median_width"] == pytest.approx(3.0)
    assert row["rect_long"] == pytest.approx(60.0)
    assert row["aspect"] == pytest.approx(20.0)
    assert row["solidity"] == pytest.approx(1.0)
    assert row["width_stratum"] == "2_to_4"
    assert row["size_stratum"] == "under_32"


def test_filled_square():
    row = geometry.describe(3, box(10, 10, 50, 50) / 100, 100, 100)
    assert row["area_px"] == 1600
    assert (row["bbox_w"], row["bbox_h"]) == (40, 40)
    assert row["aspect"] == pytest.approx(1.0)
    assert row["solidity"] == pytest.approx(1.0)
    assert row["extent"] == pytest.approx(1.0)
    assert 30 < row["median_width"] <= 40
    assert row["width_stratum"] == "over_8"
    assert row["size_stratum"] == "32_to_96"
    large = geometry.describe(3, box(0, 0, 100, 100) / 100, 100, 100)
    assert large["size_stratum"] == "over_96"


def test_strata_edges():
    assert geometry.width_stratum(2.0) == "up_to_2"
    assert geometry.width_stratum(2.1) == "2_to_4"
    assert geometry.width_stratum(8.0) == "4_to_8"
    assert geometry.width_stratum(8.5) == "over_8"
    assert geometry.size_stratum(32 * 32 - 1) == "under_32"
    assert geometry.size_stratum(96 * 96) == "32_to_96"
    assert geometry.size_stratum(96 * 96 + 1) == "over_96"


def test_tables_skip_empty_rasters():
    shapes = [box(1, 1, 4, 30), box(1, 1, 9, 9), box(5, 5, 40, 5.2)]
    rows = [geometry.describe(0, shape / 64, 64, 64) for shape in shapes]
    assert rows[2]["area_px"] is None
    strata = geometry.strata_table(rows, "width_stratum", geometry.WIDTH_STRATA)
    table = {row[0]: row for row in strata}
    assert table["needle mark"][1:] == [2, 0.0, 0.5, 0.5, 0.0]
    assert table["hole"][1:] == [0, None, None, None, None]
    percentiles = geometry.percentile_table(rows)
    area = next(row for row in percentiles if row[:2] == ["needle mark", "area_px"])
    assert area[2] == 2
    assert "Rasterized to no pixel: 1" in geometry.summary_text(rows, 2)


def test_contact_sheets_never_use_test_images(tmp_path):
    names = [f"IMG_2026050{d}_090000_jpg.rf.s{d}.jpg" for d in range(1, 4)]
    square = "1 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6"
    items = [("train", name, [square], seed) for seed, name in enumerate(names)]
    records = load_index(make_dataset(tmp_path / "data", items))
    locked = dict(zip(names, ("train", "val", "test"), strict=True))
    rows = geometry.instance_rows(records, locked)
    sheets = geometry.sample_sheets(rows, locked, 1, sheets=4, per_sheet=24, seed=0)
    picked = {item["name"] for sheet in sheets for item in sheet}
    assert picked == set(names[:2])
    by_name = {record.name: record for record in records}
    geometry.draw_sheet(sheets[0], by_name, locked, 32, 6, tmp_path / "ok.jpg")
    assert (tmp_path / "ok.jpg").is_file()
    leaked = [row for row in rows if row["name"] == names[2]]
    with pytest.raises(AssertionError):
        geometry.draw_sheet(leaked, by_name, locked, 32, 6, tmp_path / "bad.jpg")
