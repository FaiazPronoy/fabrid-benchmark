import csv
import json

import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import triage
from fabrid.labels import load_index

CONFIG = {
    "step": "audit",
    "name": "triage",
    "seed": 0,
    "per_sheet": 12,
    "columns": 4,
    "tile": 64,
}


def dataset(tmp_path):
    items = []
    for day in range(1, 8):
        for frame in range(4):
            name = f"IMG_202605{day:02d}_09{frame:02d}00_jpg.rf.t{day}{frame}.jpg"
            items.append(("train", name, [SQUARE] if frame == 3 else [], day * 10 + frame))
    records = load_index(make_dataset(tmp_path / "data", items))
    locked = {}
    for record in records:
        locked[record.name] = {1: "test", 2: "val"}.get(record.timestamp.day, "train")
    return records, locked


def fake_predictor(calls):
    def predict(paths):
        calls.extend(p.name for p in paths)
        rng = np.random.default_rng(len(paths))
        return {p.name: [(1, float(rng.random()), [2.0, 2.0, 20.0, 20.0])] for p in paths}

    return predict


def test_ordering_by_score():
    detections = {
        "a": [(0, 0.2, [0, 0, 1, 1]), (1, 0.9, [0, 0, 1, 1])],
        "b": [(2, 0.95, [0, 0, 1, 1])],
        "c": [],
        "d": [(3, 0.9, [0, 0, 1, 1])],
    }
    assert triage.by_score(["c", "d", "a", "b", "e"], detections) == ["b", "a", "d", "c", "e"]
    assert triage.shuffled(["x", "y", "z", "w"], 0) == triage.shuffled(["w", "z", "y", "x"], 0)


def test_test_images_never_reach_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRID_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("FABRID_RUNS", str(tmp_path / "runs"))
    records, locked = dataset(tmp_path)
    calls = []
    run_dir = triage.sheets(CONFIG, records, locked, fake_predictor(calls))
    tests = {n for n, s in locked.items() if s == "test"}
    assert calls
    assert not tests & set(calls)
    with pytest.raises(AssertionError):
        triage.detect(records, locked, fake_predictor([]))
    with (run_dir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sum(r.n_lines == 0 for r in records)
    for row in rows:
        if row["split"] == "test":
            assert row["top_score"] == row["top_class"] == row["detections"] == ""
    scores = [float(r["top_score"]) for r in rows if r["split"] != "test"]
    assert scores == sorted(scores, reverse=True)
    assert rows[-1]["split"] == "test"
    sheets = sorted((tmp_path / "runs" / "audit" / "triage" / "sheets").iterdir())
    assert [p.name for p in sheets] == ["test_001.jpg", "trainval_001.jpg", "trainval_002.jpg"]
    form = (run_dir / "review_form.csv").read_text(encoding="utf-8").splitlines()
    assert form[0] == "name,reviewer_a,reviewer_b,note"
    assert all(line.endswith(",,,") for line in form[1:])


def test_kappa_hand_built_table():
    table = np.array([[20, 5, 0], [10, 15, 0], [0, 0, 0]])
    assert triage.kappa(table) == pytest.approx(0.4)
    assert triage.kappa(np.diag([3, 4, 5])) == pytest.approx(1.0)
    assert triage.kappa(np.array([[6, 0, 0], [0, 0, 0], [0, 0, 0]])) is None


def write_form(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(triage.FORM_COLUMNS)
        writer.writerows(rows)


def test_unknown_or_blank_answer_raises(tmp_path):
    form = tmp_path / "form.csv"
    write_form(form, [["a", "clean", "Defect ", ""], ["b", "clean", "maybe", ""]])
    with pytest.raises(ValueError, match="maybe"):
        triage.read_form(form, ["a", "b"])
    write_form(form, [["a", "clean", "", ""], ["b", "clean", "clean", ""]])
    with pytest.raises(ValueError):
        triage.read_form(form, ["a", "b"])
    write_form(form, [["a", "clean", "clean", ""]])
    with pytest.raises(ValueError, match="missing"):
        triage.read_form(form, ["a", "b"])


def test_collect_writes_agreement(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRID_REPO", str(tmp_path))
    base = tmp_path / "results" / "audit" / "triage"
    base.mkdir(parents=True)
    splits = {"a": "train", "b": "train", "c": "val", "d": "test"}
    with (base / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(triage.CANDIDATE_COLUMNS)
        writer.writerows([[n, s, "x.jpg", 1, "", "", ""] for n, s in splits.items()])
    write_form(
        base / "review_form.csv",
        [
            ["a", "clean", "clean", ""],
            ["b", "clean", "defect", "line at left"],
            ["c", "unsure", "clean", ""],
            ["d", "clean", "clean", ""],
        ],
    )
    run_dir = triage.collect({**CONFIG, "form": None})
    result = json.loads((run_dir / "agreement.json").read_text(encoding="utf-8"))
    assert result["images"] == 4
    assert result["both_clean"] == 2
    assert result["both_clean_share"] == 0.5
    assert result["any_defect_share"] == 0.25
    assert result["pairs"]["clean|defect"] == 1
    assert result["per_split"]["train"]["images"] == 2
    clean = (run_dir / "verified_clean.csv").read_text(encoding="utf-8").splitlines()
    assert clean == ["name,split", "a,train", "d,test"]
