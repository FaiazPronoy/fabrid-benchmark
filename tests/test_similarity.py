import csv
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from conftest import fake_record, write_image
from fabrid import similarity
from fabrid.sessions import assign_sessions, session_split

CONFIG = {
    "gap_seconds": 30,
    "pair_seconds": 5,
    "percentile": 5,
    "offset": 0.05,
    "random_seed": 0,
    "max_ratio": 0.5,
    "sheet_pairs": 6,
}


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)


def test_threshold_uses_close_pairs_on_one_day():
    start = datetime(2026, 5, 1, 23, 59, 57)
    stamps = [start + timedelta(seconds=s) for s in (0, 1, 3, 13, 4)]
    vectors = np.stack([unit(a) for a in (0.0, 0.1, 0.3, 1.2, 1.5)])
    pairs = similarity.consecutive_pairs(stamps, 3)
    assert pairs == [(0, 1), (2, 4)]
    threshold, values = similarity.consecutive_threshold(vectors, stamps, 3, 5)
    assert values == pytest.approx([np.cos(0.1), np.cos(1.2)])
    low, high = np.cos(1.2), np.cos(0.1)
    assert threshold == pytest.approx(low + 0.05 * (high - low))
    with pytest.raises(ValueError):
        similarity.consecutive_threshold(vectors, [start + timedelta(minutes=i) for i in range(5)])


def test_nearest_and_leak_share_known_answer():
    rng = np.random.default_rng(0)
    train = similarity.normalize(rng.normal(size=(20, 64)))
    copies = train[:6]
    others = similarity.normalize(rng.normal(size=(6, 64)))
    vectors = np.concatenate([train, copies, others])
    names = [f"n{i}" for i in range(len(vectors))]
    mapping = {n: "train" if i < 20 else "val" if i < 29 else "test" for i, n in enumerate(names)}
    scores = similarity.nearest_scores(vectors[20:], train)
    assert scores[:6] == pytest.approx(np.ones(6), abs=1e-5)
    assert (scores[6:] < 0.6).all()
    rows = similarity.leak_table(vectors, names, {"a": mapping}, 0.95, 0.05)
    val, test = rows
    assert (val["part"], val["n"], val["hits"]) == ("val", 9, 6)
    assert val["share_t"] == pytest.approx(6 / 9)
    assert test["share_t"] == 0
    with pytest.raises(ValueError):
        similarity.nearest_scores(vectors[:2], vectors[:0])


def test_session_max_groups_blocks():
    vectors = np.stack([unit(a) for a in (0.0, 0.05, 1.0, 1.02, 0.01)])
    groups = np.array([0, 0, 1, 1, 2])
    block, ids = similarity.session_max(vectors @ vectors.T, groups)
    assert ids.tolist() == [0, 1, 2]
    assert block[0, 2] == pytest.approx(np.cos(0.01))
    assert block[0, 1] == pytest.approx(np.cos(0.95))
    pairs = similarity.cross_pairs(vectors @ vectors.T, groups)
    assert pairs[0][:2] == (0, 2)
    assert similarity.best_pair(vectors @ vectors.T, [0, 1], [4]) == (0, 4)


def test_assert_aggregate_blocks_locked_names():
    locked = {"a": "train", "b": "test"}
    similarity.assert_aggregate(["a"], locked)
    with pytest.raises(AssertionError):
        similarity.assert_aggregate(["a", "b"], locked)


def test_fill_defects_uses_image_mean():
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[:, 5:] = 200
    square = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]])
    filled = np.asarray(similarity.fill_defects(Image.fromarray(pixels), [(1, square)]))
    assert (filled[1:3, 1:3] == 100).all()
    assert (filled[8, 8] == 200).all()
    assert (pixels[1:3, 1:3] == 0).all()


def test_cache_roundtrip_and_key(tmp_path):
    cache = {"b": np.ones(3, dtype=np.float32), "a": np.zeros(3, dtype=np.float32)}
    path = tmp_path / "embeddings.npz"
    similarity.write_cache(path, cache, "key one")
    found = similarity.read_caches([path, tmp_path / "missing.npz"], "key one")
    assert sorted(found) == ["a", "b"]
    assert found["b"] == pytest.approx(np.ones(3))
    assert similarity.read_caches([path], "key two") == {}


def synthetic(records):
    rng = np.random.default_rng(5)
    sessions = assign_sessions(records)
    directions = similarity.normalize(rng.normal(size=(max(sessions.values()) + 1, 128)))
    directions[1] = directions[0]
    vectors = []
    for index, record in enumerate(records):
        record.folder = ("train", "valid", "test")[index % 3]
        noise = rng.normal(scale=0.01, size=128)
        vectors.append(directions[sessions[record.name]] + noise)
    return similarity.normalize(np.array(vectors))


def test_audit_on_synthetic_sessions(records, tmp_path):
    vectors = synthetic(records)
    result = similarity.audit(records, vectors, CONFIG, tmp_path)
    names = {"threshold.json", "leak_share.csv", "session_links.csv", "session_sizes.csv"}
    assert names | {"summary.md"} <= {p.name for p in tmp_path.iterdir()}
    assert result["threshold"]["threshold"] > 0.9
    assert result["val_share"]["random"] > 0.5
    assert result["val_share"]["export"] > 0.5
    assert result["passed"]
    locked = session_split(records)
    sessions = assign_sessions(records)
    with (tmp_path / "session_links.csv").open(encoding="utf-8") as handle:
        links = list(csv.DictReader(handle))
    assert result["links_all"] == 1
    assert result["links"] + result["links_locked"] == 1
    assert all(locked[row[k]] != "test" for row in links for k in ("name_a", "name_b"))
    split_of = {sessions[name]: split for name, split in locked.items()}
    if "test" not in (split_of[0], split_of[1]):
        assert [(row["session_a"], row["session_b"]) for row in links] == [("0", "1")]


def test_audit_fails_when_sessions_leak(records, tmp_path):
    rng = np.random.default_rng(9)
    vectors = similarity.normalize(np.tile(rng.normal(size=(1, 32)), (len(records), 1)))
    vectors = similarity.normalize(vectors + rng.normal(scale=1e-3, size=vectors.shape))
    for index, record in enumerate(records):
        record.folder = ("train", "valid", "test")[index % 3]
    result = similarity.audit(records, vectors, CONFIG, tmp_path)
    assert not result["passed"]


def test_contact_sheets(tmp_path):
    stamp = datetime(2026, 5, 1, 9)
    by_name = {}
    for i in range(3):
        record = fake_record(f"IMG_{i}.jpg", stamp, [])
        record.path = tmp_path / record.name
        write_image(record.path, i, size=40)
        by_name[record.name] = record
    pairs = [("IMG_0.jpg", "IMG_1.jpg", 0, 1, 0.99), ("IMG_1.jpg", "IMG_2.jpg", 1, 2, 0.98)]
    written = similarity.contact_sheets(pairs, by_name, tmp_path / "sheets", per_sheet=1)
    assert [p.name for p in written] == ["pairs_01.jpg", "pairs_02.jpg"]
    with Image.open(written[0]) as sheet:
        assert sheet.size == (2 * 192 + 8, 212)
