import csv
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from fabrid import figures  # noqa: E402
from fabrid.labels import CLASSES  # noqa: E402


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def context(tmp_path):
    return figures.Context(
        root=tmp_path,
        results=tmp_path / "results",
        figures=tmp_path / "figures",
        tables=tmp_path / "tables",
        dpi=50,
    )


def estimate(value, low, high):
    return {"value": value, "low": low, "high": high}


def run_metrics(per_class, mean, images=12, sessions=4):
    return {
        "n_val_images": images,
        "n_val_sessions": sessions,
        "primary": {"metric": "mask AP50", **mean},
        "mask_ap50": {"mean": mean, "per_class": per_class},
        "tolerant_ap50": {"mean": estimate(0.5, 0.4, 0.6), "per_class": per_class},
    }


def leakage_inputs(tmp_path, session_value=0.15):
    results = tmp_path / "results" / "evaluation"
    write_csv(
        results / "summary_toy" / "leakage.csv",
        [
            "model",
            "session_run",
            "random_run",
            "session_ap50",
            "random_ap50",
            "difference",
            "low",
            "high",
            "inflation_claimed",
        ],
        [
            [
                "toy",
                "toy_session_seed0",
                "toy_random_seed0",
                "0.15",
                "0.35",
                "0.2",
                "0.05",
                "0.31",
                "True",
            ]
        ],
    )
    empty = dict.fromkeys(CLASSES, estimate(None, None, None))
    write_json(
        results / "toy_session_seed0" / "metrics.json",
        run_metrics(empty, estimate(session_value, 0.091, 0.23)),
    )
    write_json(
        results / "toy_random_seed0" / "metrics.json",
        run_metrics(empty, estimate(0.35, 0.27, 0.43), sessions=9),
    )


def test_missing_input_skips_without_files(tmp_path, capsys):
    ctx = context(tmp_path)
    specs = [
        {"name": "leakage", "kind": "leakage", "models": ["toy"]},
        {"name": "ablation", "kind": "forest", "comparisons": ["first"]},
        {"name": "per_class", "kind": "per_class", "runs": ["toy_session_seed0"]},
        {"name": "pr", "kind": "pr", "runs": ["toy_session_seed0"]},
        {"name": "dataset", "kind": "dataset", "tables": ["day_class"]},
    ]
    report = figures.build(ctx, specs)
    assert report["written"] == []
    skipped = dict(report["skipped"])
    assert "results/evaluation/summary_toy/leakage.csv" in skipped["leakage"]
    assert "results/evaluation/compare_first/comparison.csv" in skipped["ablation"]
    assert "results/audit/integrity/day_class_table.csv" in skipped["dataset_day_class"]
    assert "skipped" in capsys.readouterr().out
    assert not ctx.figures.exists() and not ctx.tables.exists()


def test_figure_skipped_when_one_of_its_runs_is_missing(tmp_path):
    ctx = context(tmp_path)
    curves = tmp_path / "results" / "baselines" / "parent_seed0" / "curves.csv"
    write_csv(curves, ["epoch", "loss", "mask_AP50"], [[1, 1.0, 0.1], [2, 0.5, 0.2]])
    spec = {
        "name": "curves",
        "kind": "curves",
        "loss_columns": ["loss"],
        "ap_columns": ["mask_AP50"],
        "runs": [
            {"step": "baselines", "run": "parent_seed0"},
            {"step": "ablation", "run": "arm_seed0"},
        ],
    }
    report = figures.build(ctx, [spec])
    assert report["written"] == []
    assert "results/ablation/arm_seed0/curves.csv" in report["skipped"][0][1]
    assert not ctx.figures.exists()


def test_curves_read_columns_from_file(tmp_path):
    ctx = context(tmp_path)
    curves = tmp_path / "results" / "baselines" / "parent_seed0" / "curves.csv"
    write_csv(
        curves, ["epoch", "train/seg_loss", "metrics/mAP50(M)"], [[1, 1.0, 0.1], [2, 0.5, 0.2]]
    )
    spec = {
        "name": "curves",
        "kind": "curves",
        "loss_columns": ["loss", "train/seg_loss"],
        "ap_columns": ["mask_AP50", "metrics/mAP50(M)"],
        "runs": [{"step": "baselines", "run": "parent_seed0", "label": "parent"}],
    }
    assert figures.build(ctx, [spec])["written"] == ["curves"]
    caption = (ctx.figures / "curves.txt").read_text(encoding="utf-8")
    assert "columns train/seg_loss, metrics/mAP50(M)" in caption
    spec["ap_columns"] = ["mask_AP50"]
    report = figures.build(ctx, [{**spec, "name": "other"}])
    assert "none of the columns" in report["skipped"][0][1]


def test_leakage_numbers_equal_inputs(tmp_path):
    leakage_inputs(tmp_path)
    ctx = context(tmp_path)
    report = figures.build(ctx, [{"name": "leakage", "kind": "leakage", "models": ["toy"]}])
    assert report["written"] == ["leakage"]
    for suffix in ("pdf", "png", "txt"):
        assert (ctx.figures / f"leakage.{suffix}").is_file()
    (row,) = read_csv(ctx.tables / "leakage.csv")
    assert row["session_ap50"] == "0.15" and row["random_ap50"] == "0.35"
    assert (row["session_low"], row["session_high"]) == ("0.091", "0.23")
    assert (row["random_low"], row["random_high"]) == ("0.27", "0.43")
    assert (row["difference"], row["low"], row["high"]) == ("0.2", "0.05", "0.31")
    caption = (ctx.figures / "leakage.txt").read_text(encoding="utf-8")
    assert "toy_random_seed0 (seed 0): 12 validation images in 9 sessions" in caption
    assert "read: results/evaluation/summary_toy/leakage.csv" in caption


def test_leakage_stops_when_files_disagree(tmp_path):
    leakage_inputs(tmp_path, session_value=0.16)
    with pytest.raises(figures.Mismatch):
        figures.build(
            context(tmp_path), [{"name": "leakage", "kind": "leakage", "models": ["toy"]}]
        )


def test_forest_and_per_class_numbers_equal_inputs(tmp_path):
    ctx = context(tmp_path)
    folder = tmp_path / "results" / "evaluation" / "compare_first"
    header = ["metric", "class", "baseline", "candidate", "difference", "low", "high", "verdict"]
    rows = []
    for metric in ("mask_ap50", "tolerant_ap50"):
        rows.append([metric, "mean", "0.1", "0.2", "0.1", "0.02", "0.18", "higher"])
        for k, cls in enumerate(CLASSES):
            rows.append([metric, cls, "0.3", "0.25", "-0.05", f"-0.0{k + 6}", "0.01", "not shown"])
    write_csv(folder / "comparison.csv", header, rows)
    write_json(
        folder / "comparison.json",
        {
            "baseline": "parent_seed0",
            "candidate": "arm_seed0",
            "n_val_images": 12,
            "n_val_sessions": 4,
            "resamples": 1000,
        },
    )
    per_class_header = ["class", *figures.PER_CLASS_COLUMNS, "f1"]
    per_class_rows = [
        [cls, 7, 0.1 * k, 0.01, 0.4, 0.2, 0.05, 0.5, 0.3] for k, cls in enumerate(CLASSES)
    ]
    run = tmp_path / "results" / "evaluation" / "arm_seed0"
    write_csv(run / "per_class.csv", per_class_header, per_class_rows)
    write_json(run / "metrics.json", run_metrics({}, estimate(0.123456, 0.01, 0.3)))
    specs = [
        {"name": "ablation", "kind": "forest", "comparisons": ["first"]},
        {"name": "classes", "kind": "per_class", "runs": ["arm_seed0"]},
    ]
    assert figures.build(ctx, specs)["written"] == ["ablation", "classes"]
    table = read_csv(ctx.tables / "ablation.csv")
    assert [[r[c] for c in header] for r in table] == rows
    assert {r["candidate_run"] for r in table} == {"arm_seed0"}
    classes = read_csv(ctx.tables / "classes.csv")
    source = read_csv(run / "per_class.csv")
    for got, expected in zip(classes[:-1], source, strict=True):
        assert all(got[c] == expected[c] for c in figures.PER_CLASS_COLUMNS)
    assert (classes[-1]["class"], classes[-1]["ap50"], classes[-1]["ap50_high"]) == (
        "mean",
        "0.123456",
        "0.3",
    )


def test_dataset_tables_copy_inputs(tmp_path):
    ctx = context(tmp_path)
    audit = tmp_path / "results" / "audit"
    write_csv(
        audit / "geometry" / "size_strata.csv", ["class", "count", "under_32"], [["hole", 5, 0.4]]
    )
    counts = {
        "images": 10,
        "empty_images": 3,
        "valid_polygons": 9,
        "per_class": {cls: {"valid_polygons": k + 1, "images": k} for k, cls in enumerate(CLASSES)},
    }
    write_json(audit / "integrity" / "label_counts.json", counts)
    spec = {"name": "dataset", "kind": "dataset", "tables": ["class_counts", "size_strata"]}
    assert figures.build(ctx, [spec])["written"] == ["dataset_class_counts", "dataset_size_strata"]
    assert read_csv(ctx.tables / "dataset_size_strata.csv") == read_csv(
        audit / "geometry" / "size_strata.csv"
    )
    rows = {r["class"]: r for r in read_csv(ctx.tables / "dataset_class_counts.csv")}
    assert rows["stain"]["instances"] == "4" and rows["stain"]["images"] == "3"
    assert rows["empty label file"]["images"] == "3" and rows["all images"]["images"] == "10"


PRED_HEADER = [
    "pred_image",
    "pred_class",
    "pred_score",
    "pred_order",
    "pred_tp",
    "pred_tp_tolerant",
    "pred_bucket",
    "pred_matched_bucket",
]
EXPECTED_AP = round((51 * 1.0 + 50 * 2 / 3) / 101, 6)


def scored_run(tmp_path, preds):
    folder = tmp_path / "results" / "evaluation" / "toy_session_seed0"
    write_csv(folder / "scored_images.csv", ["image", "name"], [[0, "a.jpg"], [1, "b.jpg"]])
    write_csv(folder / "scored_predictions.csv", PRED_HEADER, preds)
    write_csv(
        folder / "scored_truth.csv",
        ["gt_image", "gt_class", "gt_bucket", "gt_matched", "gt_cldice"],
        [[0, 0, 1, 1, ""], [1, 0, 1, 1, ""]],
    )
    per_class = dict.fromkeys(CLASSES, estimate(None, None, None))
    per_class[CLASSES[0]] = estimate(EXPECTED_AP, 0.5, 1.0)
    write_json(folder / "metrics.json", run_metrics(per_class, estimate(EXPECTED_AP, 0.5, 1.0)))
    return folder


GOOD = [
    [1, 0, 0.7, 0, 1, 1, 1, 1],
    [0, 0, 0.9, 0, 1, 1, 1, 1],
    [0, 0, 0.8, 1, 0, 0, 1, -1],
]


def test_pr_curve_matches_hand_computed_ap(tmp_path):
    scored_run(tmp_path, GOOD)
    ctx = context(tmp_path)
    report = figures.build(ctx, [{"name": "pr", "kind": "pr", "runs": ["toy_session_seed0"]}])
    assert report["written"] == ["pr_toy_session_seed0"]
    caption = (ctx.figures / "pr_toy_session_seed0.txt").read_text(encoding="utf-8")
    assert f"({EXPECTED_AP:.6f})" in caption
    assert "12 validation images in 4 sessions" in caption


@pytest.mark.parametrize(
    "tampered",
    [
        [GOOD[0], GOOD[1], [0, 0, 0.8, 1, 1, 1, 1, 1]],
        [[1, 0, 0.85, 0, 1, 1, 1, 1], GOOD[1], GOOD[2]],
        GOOD[1:],
    ],
)
def test_pr_stops_when_scores_are_tampered(tmp_path, tampered):
    scored_run(tmp_path, tampered)
    ctx = context(tmp_path)
    with pytest.raises(figures.Mismatch):
        figures.build(ctx, [{"name": "pr", "kind": "pr", "runs": ["toy_session_seed0"]}])
    assert not (ctx.figures / "pr_toy_session_seed0.png").exists()


def test_pr_curve_ordering_breaks_ties_by_image_then_order():
    scores = np.array([0.5, 0.5, 0.5])
    images = np.array([1, 0, 0])
    orders = np.array([0, 1, 0])
    hits = np.array([0, 0, 1])
    recall, precision, _, ap = figures.pr_curve(scores, images, orders, hits, 1)
    assert precision.tolist() == [1.0, 0.5, 1 / 3]
    assert ap == 1.0
