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
    stray = evaluate.resolve([{"image_id": 7}], lookup, {7: names[2]})
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
