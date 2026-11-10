import json
import shutil

import numpy as np
from PIL import Image

from conftest import SQUARE, make_dataset, write_image
from fabrid import integrity
from fabrid.labels import load_index

OUTPUTS = (
    "manifest.csv",
    "duplicates_exact.csv",
    "duplicates_near.csv",
    "stems_repeated.csv",
    "label_counts.json",
    "day_class_table.csv",
    "summary.md",
)


def test_hashes_and_pairs():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    images = [Image.fromarray(a) for a in (base, noisy, other)]
    p = np.array([integrity.phash(i) for i in images], dtype=np.uint64)
    d = np.array([integrity.dhash(i) for i in images], dtype=np.uint64)
    counts = integrity.popcount(np.array([0, 1, 2**64 - 1], dtype=np.uint64))
    assert counts.tolist() == [0, 1, 64]
    pairs = integrity.near_pairs(p, d, 8)
    assert (0, 1) in [(i, j) for i, j, _, _ in pairs]
    assert all(j != 2 for _, j, _, _ in pairs)


def build(tmp_path, extra_lines):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260501_090000_jpg.rf.b.jpg", [SQUARE], 2),
        ("valid", "IMG_20260501_090020_jpg.rf.c.jpg", [], 3),
        ("test", "IMG_20260502_100000_jpg.rf.d.jpg", extra_lines, 4),
    ]
    return make_dataset(tmp_path / "data", items)


def test_audit_passes_on_clean_set(tmp_path):
    base = build(tmp_path, ["1 0.1 0.1 0.2 0.2"])
    out = tmp_path / "out"
    out.mkdir()
    result = integrity.audit(load_index(base), 8, 30, out)
    assert result["passed"]
    assert result["counts"]["parsed_invalid_polygons"] == 1
    assert result["repeated_stems"] == 1
    assert result["repeated_stems_pixels_differ"] == 1
    counts = json.loads((out / "label_counts.json").read_text(encoding="utf-8"))
    assert counts["empty_images"] == 1
    assert counts["per_class"]["needle mark"] == {"lines": 2, "valid_polygons": 2, "images": 2}
    assert all((out / name).is_file() for name in OUTPUTS)
    assert "Verdict: PASS" in (out / "summary.md").read_text(encoding="utf-8")


def test_audit_fails_on_duplicate_and_bad_line(tmp_path):
    base = build(tmp_path, ["7 0.1 0.1 0.2 0.2 0.3 0.1", "stain"])
    source = base / "train" / "images" / "IMG_20260501_090000_jpg.rf.a.jpg"
    shutil.copy2(source, base / "valid" / "images" / "IMG_20260503_120000_jpg.rf.e.jpg")
    write_image(base / "valid" / "images" / "IMG_20260504_120000_jpg.rf.f.jpg", 1)
    out = tmp_path / "out"
    out.mkdir()
    result = integrity.audit(load_index(base), 8, 30, out)
    assert not result["passed"]
    assert result["checks"] == {"no_exact_duplicates": False, "all_lines_parse": False}
    assert result["exact_groups"] == 1
    assert result["exact_files"] == 3
    assert result["counts"]["unparsed_lines"] == 2
    assert result["counts"]["label_files_missing"] == 2
