from collections import Counter
from datetime import datetime

import pytest

from conftest import fake_record
from fabrid import sessions


def test_assign_sessions_gap_and_day():
    stamps = [
        datetime(2026, 5, 1, 23, 59, 50),
        datetime(2026, 5, 1, 23, 59, 59),
        datetime(2026, 5, 2, 0, 0, 5),
        datetime(2026, 5, 2, 0, 0, 35),
        datetime(2026, 5, 2, 0, 1, 6),
    ]
    records = [fake_record(f"IMG_{i}", s, []) for i, s in enumerate(stamps)]
    assigned = sessions.assign_sessions(records, gap_seconds=30)
    assert [assigned[f"IMG_{i}"] for i in range(5)] == [0, 0, 1, 1, 2]


def test_session_split_disjoint_covers_classes_and_repeats(records):
    groups = sessions.assign_sessions(records)
    first = sessions.session_split(records)
    second = sessions.session_split(list(reversed(records)))
    assert first == second
    sessions.check_disjoint(first, groups)
    for split in sessions.SPLITS:
        members = [r for r in records if first[r.name] == split]
        present = {c for r in members for c, _ in r.polygons}
        assert present == {0, 1, 2, 3}
        assert any(not r.polygons for r in members)
    shares = Counter(first.values())
    assert shares["train"] > shares["val"] and shares["train"] > shares["test"]


def test_check_disjoint_raises_on_spanning_session():
    groups = {"a": 0, "b": 0, "c": 1}
    with pytest.raises(ValueError):
        sessions.check_disjoint({"a": "train", "b": "test", "c": "val"}, groups)
    with pytest.raises(KeyError):
        sessions.check_disjoint({"a": "train", "b": "train"}, groups)


def test_random_split_ratios_and_seed(records):
    first = sessions.random_split(records, seed=0)
    assert first == sessions.random_split(records, seed=0)
    assert first != sessions.random_split(records, seed=1)
    counts = Counter(first.values())
    assert counts["train"] == round(len(records) * 0.7)
    assert counts["val"] == round(len(records) * 0.15)
    assert sum(counts.values()) == len(records)
