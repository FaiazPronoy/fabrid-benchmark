import numpy as np
import pytest

from conftest import SQUARE, make_dataset
from fabrid import evaluate
from fabrid.folds import FOLDS
from fabrid.labels import load_index

CONFIG = {"iou": 0.5, "tolerance_px": 2, "cldice_tol_px": 1, "resamples": 300, "seed": 0}


def prediction(name, class_id, score, mask):
    flat = mask.astype(bool).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.concatenate([[False], flat, [False]]).astype(int)))
    edges = [0, *changes.tolist(), flat.size]
    counts = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    segmentation = {"size": list(mask.shape), "counts": counts}
    return {
        "file_name": name,
        "category_id": class_id + 1,
        "score": score,
        "segmentation": segmentation,
    }


def same(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for key, value in first.items():
        assert value == second[key] or (np.isnan(value) and np.isnan(second[key])), key


def test_pooled_scoring_equals_scoring_the_union(tmp_path):
    items = [
        ("train", f"IMG_20260501_09{i:02d}00_jpg.rf.p{i}.jpg", [f"{i % 4}{SQUARE[1:]}"], i)
        for i in range(10)
    ]
    records = load_index(make_dataset(tmp_path / "data", items))
    names = [r.name for r in records]
    predictions = []
    for index, record in enumerate(records):
        truth, _ = evaluate.ground_truth(record, 32, 32)
        mask = truth[0].mask
        shifted = np.roll(mask, 3 * (index % 3), axis=1)
        predictions.append(prediction(record.name, truth[0].class_id, 0.5, shifted))
        predictions.append(prediction(record.name, truth[0].class_id, 0.9 - index / 50, mask))
        predictions.append(prediction(record.name, (index + 1) % 4, 0.5, mask))
    resolved = [item["file_name"] for item in predictions]
    union, _ = evaluate.score_images(records, predictions, resolved, names, CONFIG, 1)
    parts = [names[k::3] for k in range(3)]
    scored = []
    for part in parts:
        chosen = sorted(part)
        part_scored, _ = evaluate.score_images(records, predictions, resolved, chosen, CONFIG, 1)
        scored.append((part_scored, chosen))
    merged, merged_names = evaluate.merge_scored(scored)
    assert merged_names == names
    ones = np.ones(len(names))
    same(evaluate.Scorer(merged).evaluate(ones), evaluate.Scorer(union).evaluate(ones))
    weights = np.arange(len(names)) % 3
    same(evaluate.Scorer(merged).evaluate(weights), evaluate.Scorer(union).evaluate(weights))


def fold_scored(tp: np.ndarray, scores: np.ndarray) -> evaluate.Scored:
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


def synthetic_folds(per_fold=40, gain=3, seed=11):
    rng = np.random.default_rng(seed)
    found, sessions, expected = {}, {}, []
    for k in range(FOLDS):
        names = [f"img_{k}_{i:03d}" for i in range(per_fold)]
        expected.append(names)
        for i, name in enumerate(names):
            sessions[name] = k * 1000 + i // 2
        scores = rng.random(per_fold)
        base_tp = rng.random(per_fold) < 0.6
        cand_tp = base_tp.copy()
        cand_tp[np.flatnonzero(~base_tp)[:gain]] = True
        image_sessions = np.array([sessions[n] for n in names])
        for run, tp in (("base", base_tp), ("cand", cand_tp)):
            found[f"{run}_fold{k}"] = {
                "run": f"{run}_fold{k}",
                "scorer": evaluate.Scorer(fold_scored(tp, scores)),
                "val_names": names,
                "sessions": image_sessions,
            }
    locked = dict.fromkeys(sessions, "val")
    return found, sessions, locked, expected


COMPARISON = {"name": "demo", "baseline": "base", "candidate": "cand"}


def test_pooled_interval_narrower_than_one_fold():
    found, sessions, locked, expected = synthetic_folds()
    result = evaluate.compare_pooled(COMPARISON, found, locked, sessions, CONFIG, expected)
    pooled = result["mask_ap50"]["mean"]
    assert result["n_val_images"] == FOLDS * 40
    assert pooled["difference"] > 0
    assert len(result["per_fold_mask_ap50"]) == FOLDS
    first, second = found["base_fold0"], found["cand_fold0"]
    weights = evaluate.draws(first["sessions"], CONFIG["resamples"], 0, "fold0")
    single = evaluate.paired(first["scorer"], second["scorer"], weights)["ap50"]
    assert pooled["high"] - pooled["low"] < single["high"] - single["low"]
    again = evaluate.compare_pooled(COMPARISON, found, locked, sessions, CONFIG, expected)
    assert again == result
    rows = evaluate.pooled_rows(result)
    assert rows[-1][0] == "fold_mask_ap50" and len(rows) == 2 * 5 + FOLDS


def test_pooled_refusals():
    found, sessions, locked, expected = synthetic_folds(per_fold=10)
    missing = {k: v for k, v in found.items() if k != "cand_fold3"}
    with pytest.raises(ValueError, match="missing"):
        evaluate.compare_pooled(COMPARISON, missing, locked, sessions, CONFIG, expected)
    twice = dict(found)
    for run in ("base_fold1", "cand_fold1"):
        names = sorted([*found[run]["val_names"][1:], found["base_fold0"]["val_names"][0]])
        twice[run] = {**found[run], "val_names": names}
    with pytest.raises(ValueError, match="more than one fold"):
        evaluate.compare_pooled(COMPARISON, twice, locked, sessions, CONFIG)
    with pytest.raises(ValueError, match="fold assignment"):
        evaluate.compare_pooled(COMPARISON, twice, locked, sessions, CONFIG, expected)
    other = {**found, "cand_fold2": {**found["cand_fold2"], "val_names": ["x"]}}
    with pytest.raises(ValueError, match="different images"):
        evaluate.compare_pooled(COMPARISON, other, locked, sessions, CONFIG, expected)
    larger = {**locked, "img_9_000": "train"}
    with pytest.raises(ValueError, match="do not cover"):
        evaluate.compare_pooled(COMPARISON, found, larger, sessions, CONFIG, expected)


def test_interval_verdict_and_fold_runs():
    assert evaluate.interval_verdict(0.01, 0.09) == "higher"
    assert evaluate.interval_verdict(-0.09, -0.01) == "lower"
    assert evaluate.interval_verdict(-0.01, 0.09) == "not shown"
    assert evaluate.interval_verdict(0.0, 0.09) == "not shown"
    assert evaluate.interval_verdict(None, None) == "not shown"
    config = {"runs": [{"name": "a_fold0", "step": "crossval"}], "crossval": [COMPARISON]}
    runs = evaluate.with_fold_runs(config)
    assert [r["name"] for r in runs] == [
        "a_fold0",
        *[f"base_fold{k}" for k in range(FOLDS)],
        *[f"cand_fold{k}" for k in range(FOLDS)],
    ]
    assert all(r["step"] == "crossval" for r in runs)
    assert evaluate.with_fold_runs({"runs": [{"name": "x"}]}) == [{"name": "x"}]
