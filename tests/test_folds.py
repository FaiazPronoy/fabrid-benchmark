import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from conftest import SQUARE, fake_record, make_dataset
from fabrid import folds, maskrcnn, runio
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import assign_sessions, session_split
from fabrid.yolo import arm_mapping, write_csv

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_folds_cover_pool_once_and_keep_sessions_whole(records):
    assigned = folds.assign_folds(records)
    locked = session_split(records)
    sessions = assign_sessions(records)
    pool = {name for name, part in locked.items() if part != "test"}
    assert any(part == "test" for part in locked.values())
    assert set(assigned) == pool
    assert sorted(set(assigned.values())) == list(range(folds.FOLDS))
    by_session = {}
    for name, fold in assigned.items():
        assert by_session.setdefault(sessions[name], fold) == fold
    shuffled = records[:]
    random.Random(5).shuffle(shuffled)
    assert folds.assign_folds(shuffled) == assigned
    rows = folds.fold_table(records, assigned, sessions)
    assert [row[-1] for row in rows] == [""] * folds.FOLDS
    assert sum(row[2] for row in rows) == len(pool)
    assert all(row[2] + row[-2] == len(pool) for row in rows)
    assert all(0.1 * len(pool) <= row[2] <= 0.3 * len(pool) for row in rows)


def rare_records(seed):
    rng = random.Random(seed)
    clock = datetime(2026, 5, 1, 9, 0, 0)
    chosen = set(rng.sample(range(40), rng.choice((1, 2, 3))))
    out = []
    for session in range(40):
        clock += timedelta(minutes=10)
        for frame in range(rng.randint(1, 6)):
            stamp = clock + timedelta(seconds=5 * frame)
            classes = [rng.randrange(3)] + ([3] if session in chosen and frame == 0 else [])
            name = f"IMG_{stamp:%Y%m%d_%H%M%S}_jpg.rf.r{session}.jpg"
            out.append(fake_record(name, stamp, classes))
    return out


@pytest.mark.parametrize("seed", range(8))
def test_classes_reach_every_training_side_where_possible(seed):
    records = rare_records(seed)
    assigned = folds.assign_folds(records)
    sessions = assign_sessions(records)
    rows = folds.fold_table(records, assigned, sessions)
    for class_id, class_name in enumerate(CLASSES):
        holding = {
            sessions[r.name]
            for r in records
            if r.name in assigned and class_id in {c for c, _ in r.polygons}
        }
        lacking = [row[0] for row in rows if class_name in row[-1].split(";")]
        if len(holding) >= 2:
            assert lacking == []
        elif len(holding) == 1:
            assert lacking == [next(assigned[n] for n in assigned if sessions[n] in holding)]


def test_fold_view_guards(records):
    assigned = folds.assign_folds(records)
    locked = session_split(records)
    sessions = assign_sessions(records)
    view = folds.fold_view(assigned, 1, locked, sessions)
    assert {n for n, part in view.items() if part == "val"} == {
        n for n, k in assigned.items() if k == 1
    }
    test_name = next(n for n, part in locked.items() if part == "test")
    with pytest.raises(AssertionError, match="test"):
        folds.check_fold_view({**view, test_name: "train"}, locked, sessions)
    with pytest.raises(AssertionError, match="outside"):
        folds.check_fold_view({**view, "IMG_unknown.jpg": "train"}, locked, sessions)
    counts = {}
    for name in view:
        counts.setdefault(sessions[name], []).append(name)
    split_me = next(names for names in counts.values() if len(names) > 1)
    broken = {**view, split_me[0]: "train" if view[split_me[0]] == "val" else "val"}
    with pytest.raises(ValueError, match="spans"):
        folds.check_fold_view(broken, locked, sessions)
    with pytest.raises(ValueError):
        folds.fold_view(assigned, 7, locked, sessions)


def test_trainer_without_fold_is_unchanged(records, monkeypatch, tmp_path):
    monkeypatch.setenv("FABRID_REPO", str(tmp_path))
    config = runio.load_config(CONFIGS / "maskrcnn_r50.yaml")
    assert maskrcnn.with_fold(config) is config
    assert maskrcnn.run_name(config) == "maskrcnn_r50_session_seed0"
    assert maskrcnn.view_of(records, config) == arm_mapping(records, "session", 0, 30)
    aug = runio.load_config(CONFIGS / "maskrcnn_r50_aug.yaml")
    assert maskrcnn.with_fold(aug)["step"] == "generalization"
    assert maskrcnn.run_name(aug) == "maskrcnn_r50_aug_session_seed0"


def test_trainer_with_fold(records, monkeypatch, tmp_path):
    monkeypatch.setenv("FABRID_REPO", str(tmp_path))
    config = runio.load_config(CONFIGS / "crossval" / "maskrcnn_r50_aug_fold2.yaml")
    ready = maskrcnn.with_fold(config)
    assert ready["step"] == "crossval"
    assert maskrcnn.run_name(ready) == "maskrcnn_r50_aug_session_seed0_fold2"
    view = maskrcnn.view_of(records, ready)
    assigned = folds.assign_folds(records)
    assert view == {n: "val" if k == 2 else "train" for n, k in assigned.items()}
    locked = session_split(records)
    train, val = maskrcnn.split_records(records, view, locked)
    assert len(train) + len(val) == sum(part != "test" for part in locked.values())
    for bad in ({**config, "fold": 5}, {**config, "arm": "random"}):
        with pytest.raises(ValueError):
            maskrcnn.with_fold(bad)
    stored = tmp_path / folds.FOLDS_FILE
    stored.parent.mkdir(parents=True)
    rows = [[n, 0, k] for n, k in assigned.items()]
    write_csv(stored, ["name", "session", "fold"], rows)
    assert maskrcnn.view_of(records, ready) == view
    rows[0][2] = (rows[0][2] + 1) % folds.FOLDS
    write_csv(stored, ["name", "session", "fold"], rows)
    with pytest.raises(ValueError, match="differs"):
        maskrcnn.view_of(records, ready)


def test_fold_configs_match_parents():
    for parent in ("maskrcnn_r50", "maskrcnn_r50_aug"):
        base = runio.load_config(CONFIGS / f"{parent}.yaml")
        for fold in range(folds.FOLDS):
            found = runio.load_config(CONFIGS / "crossval" / f"{parent}_fold{fold}.yaml")
            assert found == {**base, "step": "crossval", "fold": fold}


def test_folds_main_writes_tables(tmp_path, monkeypatch):
    start = datetime(2026, 5, 1, 9, 0, 0)
    items = []
    for index in range(24):
        stamp = start + timedelta(minutes=10 * index)
        line = f"{index % 4}{SQUARE[1:]}"
        name = f"IMG_{stamp:%Y%m%d_%H%M%S}_jpg.rf.s{index}.jpg"
        items.append(("train", name, [line] if index % 6 else [], index))
    data = make_dataset(tmp_path / "data", items)
    config = {"step": "crossval", "name": "folds", "gap_seconds": 30, "publish": False}
    path = tmp_path / "folds.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("FABRID_DATA", str(data))
    monkeypatch.setenv("FABRID_REPO", str(tmp_path / "repo"))
    monkeypatch.setattr("sys.argv", ["folds", "--config", str(path)])
    folds.main()
    out = tmp_path / "repo" / "results" / "crossval" / "folds"
    assert all((out / name).is_file() for name in ("config.yaml", "folds.csv", "sizes.csv"))
    written = folds.read_folds(out / "folds.csv")
    locked = session_split(load_index(data))
    assert set(written) == {n for n, part in locked.items() if part != "test"}
    sizes = (out / "sizes.csv").read_text(encoding="utf-8").splitlines()
    assert len(sizes) == 1 + folds.FOLDS
