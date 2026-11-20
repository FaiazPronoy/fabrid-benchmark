import argparse
import json
from pathlib import Path

import numpy as np

from fabrid import runio
from fabrid.evaluate import Scorer, bootstrap, clean, draws, find_file, load_scored, paired
from fabrid.labels import CLASSES
from fabrid.pilot import read_fabrics
from fabrid.yolo import write_csv

KINDS = ("mask", "box")
BETTER = "better on new fabrics"
NOT_SHOWN = "not shown"
CAVEAT = (
    "Intervals are 95% bootstraps over the pilot test images, which come from only 5 fabrics; "
    "they do not describe variation between fabrics."
)


def locate(arm: str, roots) -> Path:
    found = find_file(f"results/pilot/{arm}/scored_images.csv", roots)
    if found is None:
        raise SystemExit(f"stored scores of {arm} not found under {roots}")
    return found.parent


def load_arm(arm: str, roots, test_names: list[str]) -> dict:
    folder = locate(arm, roots)
    scorers = {}
    for kind, path in zip(KINDS, (folder, folder / "box"), strict=True):
        loaded = load_scored(path)
        if loaded is None:
            raise SystemExit(f"{arm}: scored tables missing in {path}")
        scored, names = loaded
        if names != test_names:
            raise ValueError(f"{arm}: scored images differ from the pilot test list")
        scorers[kind] = Scorer(scored)
    stored = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    value = clean(scorers["mask"].evaluate(np.ones(len(test_names)))["ap50"])
    if stored["mask_ap50"]["mean"] != value:
        written = stored["mask_ap50"]["mean"]
        raise ValueError(f"{arm}: stored scores give {value}, metrics.json says {written}")
    print(f"{arm}: read {folder}")
    return {"folder": str(folder), "scorers": scorers, "metrics": stored}


def pass_verdict(low) -> str:
    return BETTER if low is not None and low > 0 else NOT_SHOWN


def fabric_rows(arm: str, scorer: Scorer, fabric_of: np.ndarray, weights) -> list[list]:
    rows = []
    for fabric in sorted(set(fabric_of.tolist())):
        inside = (fabric_of == fabric).astype(float)
        value = scorer.evaluate(inside)["ap50"]
        samples = np.array([scorer.evaluate(w * inside)["ap50"] for w in weights])
        samples = samples[~np.isnan(samples)]
        low, high = np.percentile(samples, [2.5, 97.5]) if samples.size else (np.nan, np.nan)
        truth = scorer.scored.gt_image
        gt = int(np.isin(truth, np.flatnonzero(inside)).sum())
        rows.append([arm, fabric, int(inside.sum()), gt, clean(value), clean(low), clean(high)])
    return rows


def class_rows(arm: str, estimates: dict, truth: np.ndarray) -> list[list]:
    rows = []
    for label in ("mean", *CLASSES):
        key = "ap50" if label == "mean" else f"ap50/{label}"
        gt = truth.size if label == "mean" else int((truth == CLASSES.index(label)).sum())
        values = [estimates[kind][key][k] for kind in KINDS for k in ("value", "low", "high")]
        rows.append([arm, label, gt, *values])
    return rows


def comparison(entry: dict, found: dict, weights) -> dict:
    out = {"name": entry["name"], "baseline": entry["baseline"], "candidate": entry["candidate"]}
    for kind in KINDS:
        table = paired(
            found[entry["baseline"]]["scorers"][kind],
            found[entry["candidate"]]["scorers"][kind],
            weights,
        )
        out[f"{kind}_ap50"] = {
            "mean": table["ap50"],
            "per_class": {c: table[f"ap50/{c}"] for c in CLASSES},
        }
    for kind in KINDS:
        for item in (out[f"{kind}_ap50"]["mean"], *out[f"{kind}_ap50"]["per_class"].values()):
            item["verdict"] = pass_verdict(item["low"])
    out["verdict"] = out["mask_ap50"]["mean"]["verdict"]
    return out


def show(value, low, high) -> str:
    if value is None:
        return "n/a"
    if low is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def summary_text(arms, results: dict, comparisons: list[dict], n_images: int, fabrics) -> str:
    lines = [
        "# Pilot on new fabrics",
        "",
        f"Pilot test: {n_images} images from {len(fabrics)} fabrics ({', '.join(fabrics)}).",
        "Single endpoint: mask AP50 over the four classes. " + CAVEAT,
        "",
        "| arm | mask AP50 | box AP50 |",
        "|---|---|---|",
    ]
    for arm in arms:
        mask, box = (results[arm][kind]["ap50"] for kind in KINDS)
        lines.append(
            f"| {arm} | {show(mask['value'], mask['low'], mask['high'])} "
            f"| {show(box['value'], box['low'], box['high'])} |"
        )
    lines += ["", "| comparison | mask AP50 difference | verdict |", "|---|---|---|"]
    for item in comparisons:
        mean = item["mask_ap50"]["mean"]
        label = f"{item['candidate']} minus {item['baseline']}"
        interval = show(mean["difference"], mean["low"], mean["high"])
        lines.append(f"| {label} | {interval} | {item['verdict']} |")
    lines += [
        "",
        "Rule: a candidate counts as better than its baseline on new fabrics only if the lower "
        "end of the paired 95% interval of its mask AP50 difference is above 0. Anything else is "
        "reported as not shown.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    fabrics = read_fabrics(runio.repo_root() / config["fabrics"])
    test_names = sorted(n for n, (split, _) in fabrics.items() if split == "test")
    roots = [Path(root) for root in config["roots"]]
    found = {arm: load_arm(arm, roots, test_names) for arm in config["arms"]}
    weights = draws(np.arange(len(test_names)), config["resamples"], config["seed"], "pilot")
    fabric_of = np.array([fabrics[n][1] for n in test_names])
    names = sorted(set(fabric_of.tolist()))
    results, metrics, per_class, per_fabric = {}, {}, [], []
    for arm in config["arms"]:
        scorers = found[arm]["scorers"]
        results[arm] = {kind: bootstrap(scorers[kind], weights)[0] for kind in KINDS}
        metrics[arm] = {
            "source": found[arm]["folder"],
            **{
                f"{kind}_ap50": {
                    "mean": results[arm][kind]["ap50"],
                    "per_class": {c: results[arm][kind][f"ap50/{c}"] for c in CLASSES},
                }
                for kind in KINDS
            },
        }
        per_class += class_rows(arm, results[arm], scorers["mask"].scored.gt_class)
        per_fabric += fabric_rows(arm, scorers["mask"], fabric_of, weights)
    comparisons = [comparison(entry, found, weights) for entry in config["comparisons"]]
    run_dir = runio.start_run(config["step"], config["name"], config)
    header = {
        "n_test_images": len(test_names),
        "test_fabrics": {f: int((fabric_of == f).sum()) for f in names},
        "resamples": config["resamples"],
        "seed": config["seed"],
        "interval": CAVEAT,
        "endpoint": "mask AP50 over the four classes on the pilot test",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps({**header, "arms": metrics}, indent=2), encoding="utf-8"
    )
    write_csv(
        run_dir / "per_class.csv",
        ["arm", "class", "gt", "mask_ap50", "mask_low", "mask_high", "box_ap50", "box_low"]
        + ["box_high"],
        per_class,
    )
    write_csv(
        run_dir / "per_fabric.csv",
        ["arm", "fabric", "images", "gt", "mask_ap50", "mask_low", "mask_high"],
        per_fabric,
    )
    rows = []
    for item in comparisons:
        for kind in KINDS:
            part = item[f"{kind}_ap50"]
            for label, values in (("mean", part["mean"]), *part["per_class"].items()):
                rows.append(
                    [item["name"], item["candidate"], item["baseline"], f"{kind}_ap50", label]
                    + [values[k] for k in ("baseline", "candidate", "difference", "low", "high")]
                    + [values["verdict"]]
                )
    write_csv(
        run_dir / "comparisons.csv",
        ["comparison", "candidate", "baseline", "metric", "class", "baseline_value"]
        + ["candidate_value", "difference", "low", "high", "verdict"],
        rows,
    )
    (run_dir / "comparisons.json").write_text(
        json.dumps({**header, "comparisons": comparisons}, indent=2), encoding="utf-8"
    )
    text = summary_text(config["arms"], results, comparisons, len(test_names), names)
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
