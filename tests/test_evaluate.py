import json

import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import evaluate
from fabrid.labels import load_index
from fabrid.metrics import match_instances

CONFIG = {"iou": 0.5, "tolerance_px": 2, "cldice_tol_px": 1}
HAND_AP = (51 + 50 * 2 / 3) / 101


def rle_string(mask: np.ndarray) -> str:
    flat = mask.astype(bool).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.concatenate([[False], flat, [False]]).astype(int)))
    edges = [0, *changes.tolist(), flat.size]
    counts = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    if counts and counts[-1] == 0:
        counts.pop()
    text = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = value != -1 if code & 0x10 else value != 0
            if more:
                code |= 0x20
            text.append(chr(code + 48))
    return "".join(text)


def prediction(name, class_id, score, mask):
    size = list(mask.shape)
    segmentation = {"size": size, "counts": rle_string(mask)}
    return {
        "file_name": name,
        "category_id": class_id + 1,
        "score": score,
        "segmentation": segmentation,
    }


def scored_set(tmp_path, items):
    records = load_index(make_dataset(tmp_path / "data", items))
    names = [r.name for r in records]
    return records, names


def score(records, predictions, names):
    resolved = [item["file_name"] for item in predictions]
    return evaluate.score_images(records, predictions, resolved, names, CONFIG, 1)


def gt_mask(record, index=0):
    instances, _ = evaluate.ground_truth(record, 32, 32)
    return instances[index].mask


def test_rle_round_trip():
    rng = np.random.default_rng(3)
    for shape in ((32, 32), (17, 40)):
        mask = rng.random(shape) > 0.6
        mask[:, :3] = True
        decoded = evaluate.decode({"size": list(shape), "counts": rle_string(mask)}, *shape)
        assert (decoded == mask).all()
    with pytest.raises(ValueError):
        evaluate.decode({"size": [8, 8], "counts": rle_string(mask)}, 8, 8)


def test_greedy_agrees_with_metrics():
    rng = np.random.default_rng(5)
    preds = rng.random((6, 12, 12)) > 0.5
    gts = preds[:3] | (rng.random((3, 12, 12)) > 0.8)
    scores = rng.random(6)
    expected = match_instances(preds, gts, 0.5, scores)
    instances = [evaluate.Instance(0, m, evaluate.bounding_box(m)) for m in preds]
    targets = [evaluate.Instance(0, m, evaluate.bounding_box(m)) for m in gts]
    matrix = evaluate.overlap_matrix(instances, targets, evaluate.plain_iou, 0)
    matched = evaluate.greedy(matrix, scores, 0.5)
    assert sorted((i, j) for i, j in enumerate(matched) if j is not None) == sorted(expected.tp)


def test_known_ap_for_hand_built_case(tmp_path):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260502_090000_jpg.rf.b.jpg", [SQUARE], 2),
    ]
    records, names = scored_set(tmp_path, items)
    exact = gt_mask(records[0])
    elsewhere = np.zeros_like(exact)
    elsewhere[20:28, 20:28] = True
    predictions = [
        prediction(names[0], 0, 0.9, exact),
        prediction(names[0], 0, 0.8, elsewhere),
        prediction(names[1], 0, 0.7, exact),
    ]
    scored, empty = score(records, predictions, names)
    result = evaluate.Scorer(scored).evaluate(np.ones(2))
    assert empty == 0
    assert result["ap50/needle mark"] == pytest.approx(HAND_AP)
    assert result["ap50"] == pytest.approx(HAND_AP)
    assert np.isnan(result["ap50/hole"])
    assert result["f1_point/needle mark/threshold"] == pytest.approx(0.7)
    assert result["f1_point/needle mark/recall"] == pytest.approx(1.0)
    assert result["f1_point/needle mark/precision"] == pytest.approx(2 / 3)
    assert result["size_ap50/small"] == pytest.approx(HAND_AP)
    assert np.isnan(result["size_ap50/large"])


def test_average_precision_directly():
    tp = np.array([True, False, True])
    assert evaluate.average_precision(tp, np.ones(3), 2) == pytest.approx(HAND_AP)
    assert evaluate.average_precision(tp, np.array([1, 0, 1]), 2) == pytest.approx(1.0)
    assert evaluate.average_precision(tp[:0], np.ones(0), 2) == 0.0
    assert np.isnan(evaluate.average_precision(tp, np.ones(3), 0))


def test_tolerant_ap_above_plain_for_shifted_thin_line(tmp_path):
    line = "2 0.3 0.1 0.33 0.1 0.33 0.9 0.3 0.9"
    items = [("train", "IMG_20260501_090000_jpg.rf.a.jpg", [line], 1)]
    records, names = scored_set(tmp_path, items)
    truth = gt_mask(records[0])
    assert truth.sum() > 0
    assert truth.any(axis=0).sum() == 1
    shifted = np.roll(truth, 2, axis=1)
    predictions = [prediction(names[0], 2, 0.9, shifted)]
    scored, _ = score(records, predictions, names)
    result = evaluate.Scorer(scored).evaluate(np.ones(1))
    assert result["ap50/missing yarn"] == 0.0
    assert result["tolerant_ap50/missing yarn"] == pytest.approx(1.0)
    assert result["tolerant_ap50"] > result["ap50"]
    assert result["missing_yarn/matched_share"] == 0.0


def test_bootstrap_reproducible(tmp_path):
    items = [
        ("train", f"IMG_202605{day:02d}_090000_jpg.rf.{day}.jpg", [SQUARE], day)
        for day in range(1, 6)
    ]
    records, names = scored_set(tmp_path, items)
    exact = gt_mask(records[0])
    predictions = [prediction(n, 0, 0.5 + i / 10, exact) for i, n in enumerate(names[:3])]
    scored, _ = score(records, predictions, names)
    sessions = np.arange(len(names))
    first = evaluate.draws(sessions, 200, 0, "m_session_seed0")
    second = evaluate.draws(sessions, 200, 0, "m_session_seed0")
    other = evaluate.draws(sessions, 200, 0, "m_random_seed0")
    assert all((a == b).all() for a, b in zip(first, second, strict=True))
    assert any((a != b).any() for a, b in zip(first, other, strict=True))
    assert all(w.sum() == len(names) for w in first)
    scorer = evaluate.Scorer(scored)
    one, draws_one = evaluate.bootstrap(scorer, first)
    two, draws_two = evaluate.bootstrap(scorer, second)
    assert one == two
    assert np.array_equal(draws_one["ap50"], draws_two["ap50"], equal_nan=True)
    assert one["ap50"]["low"] <= one["ap50"]["value"] <= one["ap50"]["high"]


def test_prediction_outside_train_and_val_rejected(tmp_path):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260502_090000_jpg.rf.b.jpg", [SQUARE], 2),
        ("test", "IMG_20260503_090000_jpg.rf.c.jpg", [SQUARE], 3),
    ]
    records, names = scored_set(tmp_path, items)
    lookup = {n: n for n in names} | {n.rsplit(".", 1)[0]: n for n in names}
    mapping = {names[0]: "train", names[1]: "val"}
    locked = {names[0]: "train", names[1]: "val", names[2]: "test"}
    predictions = [{"image_id": names[1].rsplit(".", 1)[0]}, {"file_name": names[0]}]
    resolved = evaluate.resolve(predictions, lookup, {})
    evaluate.guard(resolved, mapping, locked)
    stray = evaluate.resolve([{"image_id": 7}], lookup, {"7": names[2]})
    with pytest.raises(AssertionError):
        evaluate.guard(stray, mapping, locked)
    with pytest.raises(AssertionError):
        evaluate.guard(resolved, {**mapping, names[2]: "val"}, locked)
    with pytest.raises(ValueError):
        evaluate.resolve([{"image_id": 8}], lookup, {})
    with pytest.raises(ValueError):
        evaluate.resolve([{"image_id": "IMG_unknown"}], lookup, {})


def test_leakage_claims_only_when_interval_excludes_zero():
    def run(arm, value, draws):
        metrics = {"primary": {"value": value}}
        return {"run": f"m_{arm}", "model": "m", "arm": arm, "metrics": metrics, "draws": draws}

    flat = np.full(100, 0.5)
    clear = evaluate.leakage([run("session", 0.5, flat), run("random", 0.7, flat + 0.2)])
    assert clear[0][-1] is True
    noisy = np.linspace(0.3, 0.9, 100)
    unclear = evaluate.leakage([run("session", 0.5, flat), run("random", 0.6, noisy)])
    assert unclear[0][-1] is False
    assert evaluate.leakage([run("session", 0.5, flat)]) == []


def hand_scored(tp: np.ndarray, scores: np.ndarray) -> evaluate.Scored:
    n = len(tp)
    image = np.arange(n)
    zeros = np.zeros(n, dtype=int)
    return evaluate.Scored(
        pred_image=image,
        pred_class=zeros,
        pred_score=scores,
        pred_order=zeros,
        pred_tp=tp,
        pred_tp_tolerant=tp,
        pred_bucket=zeros,
        pred_matched_bucket=np.where(tp, 0, -1),
        gt_image=image,
        gt_class=zeros,
        gt_bucket=zeros,
        gt_matched=tp,
        gt_cldice=np.full(n, np.nan),
        n_images=n,
    )


def test_paired_interval_narrower_than_unpaired():
    rng = np.random.default_rng(11)
    n = 80
    scores = rng.random(n)
    base_tp = rng.random(n) < 0.6
    cand_tp = base_tp.copy()
    cand_tp[np.flatnonzero(~base_tp)[:4]] = True
    baseline = evaluate.Scorer(hand_scored(base_tp, scores))
    candidate = evaluate.Scorer(hand_scored(cand_tp, scores))
    sessions = np.repeat(np.arange(n // 2), 2)
    shared = evaluate.draws(sessions, 500, 0, "pair")
    table = evaluate.paired(baseline, candidate, shared)
    entry = table["ap50"]
    assert entry["difference"] > 0
    first = evaluate.draws(sessions, 500, 0, "first")
    second = evaluate.draws(sessions, 500, 0, "second")
    a = np.array([baseline.evaluate(w)["ap50"] for w in first])
    b = np.array([candidate.evaluate(w)["ap50"] for w in second])
    low, high = np.percentile(b - a, [2.5, 97.5])
    assert entry["high"] - entry["low"] < (high - low) / 3
    again = evaluate.paired(baseline, candidate, shared)
    assert again == table
    assert set(table) == {
        f"{m}{s}"
        for m in ("ap50", "tolerant_ap50")
        for s in ("", "/needle mark", "/hole", "/missing yarn", "/stain")
    }


def test_differing_validation_sets_raise():
    scorer = evaluate.Scorer(hand_scored(np.array([True, False]), np.array([0.9, 0.5])))
    run = {"scorer": scorer, "val_names": ["a", "b"], "sessions": np.array([0, 1])}
    found = {"x": run, "y": {**run, "val_names": ["a", "c"]}}
    config = {"resamples": 20, "seed": 0}
    with pytest.raises(ValueError, match="differ"):
        evaluate.compare({"name": "t", "baseline": "x", "candidate": "y"}, found, config)
    result = evaluate.compare({"name": "t", "baseline": "x", "candidate": "x"}, found, config)
    assert result["mask_ap50"]["mean"]["difference"] == 0
    assert result["verdict"] == "not shown"
    rows = evaluate.comparison_rows(result)
    assert len(rows) == 2 * 5
    assert rows[0][:2] == ["mask_ap50", "mean"]


def test_verdict_logic():
    assert evaluate.verdict(0.05, 0.01, 0.09) == "higher"
    assert evaluate.verdict(-0.05, -0.09, -0.01) == "lower"
    assert evaluate.verdict(0.05, -0.01, 0.09) == "not shown"
    assert evaluate.verdict(-0.05, -0.09, 0.01) == "not shown"
    assert evaluate.verdict(0.05, 0.0, 0.09) == "not shown"
    assert evaluate.verdict(None, 0.01, 0.09) == "not shown"
    assert evaluate.verdict(0.05, None, None) == "not shown"


def test_string_image_ids_from_ground_truth_file(tmp_path):
    name = "IMG_20260501_090000_jpg.rf.a.jpg"
    stem = name.rsplit(".", 1)[0]
    truth = {"images": [{"id": stem, "file_name": name}, {"id": 4, "file_name": "other.jpg"}]}
    (tmp_path / "val_ground_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    ids = evaluate.image_ids(tmp_path / "val_predictions.json")
    lookup = {name: name, stem: name, "other.jpg": "other.jpg"}
    resolved = evaluate.resolve([{"image_id": stem}, {"image_id": 4}], lookup, ids)
    assert resolved == [name, "other.jpg"]


def test_stored_scores_round_trip_and_guards(tmp_path):
    items = [
        ("train", "IMG_20260501_090000_jpg.rf.a.jpg", [SQUARE], 1),
        ("train", "IMG_20260502_090000_jpg.rf.b.jpg", [SQUARE, SQUARE], 2),
    ]
    records, names = scored_set(tmp_path, items)
    hit = gt_mask(records[0])
    predictions = [
        prediction(names[0], 0, 0.9, hit),
        prediction(names[1], 0, 0.123456789, np.roll(hit, 12, axis=1)),
    ]
    scored, _ = score(records, predictions, names)
    run_dir = tmp_path / "evaluation" / "demo"
    run_dir.mkdir(parents=True)
    evaluate.save_scored(run_dir, scored, names)
    loaded, loaded_names = evaluate.load_scored(run_dir)
    assert loaded_names == names
    ones = np.ones(len(names))
    first = evaluate.Scorer(scored).evaluate(ones)
    second = evaluate.Scorer(loaded).evaluate(ones)
    assert first.keys() == second.keys()
    for key, value in first.items():
        assert value == second[key] or (np.isnan(value) and np.isnan(second[key]))
    run = {"name": "demo"}
    base = tmp_path / "evaluation"
    locked = {names[0]: "val", names[1]: "val"}
    assert evaluate.cached_run(run, None, locked, base) is None
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assert evaluate.cached_run(run, None, locked, base) is not None
    with pytest.raises(AssertionError):
        evaluate.cached_run(run, None, {**locked, names[1]: "test"}, base)
    with pytest.raises(ValueError):
        evaluate.cached_run(run, {names[0]: "val", names[1]: "train"}, locked, base)
    assert evaluate.load_scored(tmp_path) is None
