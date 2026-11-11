import argparse
import csv
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from fabrid import runio
from fabrid.labels import CLASSES

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
VERDICT_STYLE = {
    "higher": ("#0072B2", "^"),
    "lower": ("#D55E00", "v"),
    "not shown": ("#666666", "o"),
}
RECALL_LEVELS = np.linspace(0, 1, 101)
PR_COLUMNS = ("pred_class", "pred_score", "pred_image", "pred_order", "pred_tp")
STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


class Skip(Exception):
    pass


class Missing(Skip):
    def __init__(self, paths):
        self.paths = list(paths)
        super().__init__("missing " + ", ".join(p.as_posix() for p in self.paths))


class Mismatch(Exception):
    pass


@dataclass
class Context:
    root: Path
    results: Path
    figures: Path
    tables: Path
    dpi: int = 200

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


def need(ctx: Context, *paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise Missing(Path(ctx.rel(p)) for p in missing)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_of(run: str) -> str:
    match = re.search(r"_seed(\d+)$", run)
    return f"seed {match.group(1)}" if match else "seed not recorded"


def number(text) -> float:
    return float(text) if text not in ("", None) else math.nan


def text_of(value) -> str:
    return "" if value is None else str(value)


def interval(value: dict) -> tuple[float, float, float]:
    return number(value["value"]), number(value["low"]), number(value["high"])


def write_table(ctx: Context, name: str, header, rows, sources) -> list[Path]:
    ctx.tables.mkdir(parents=True, exist_ok=True)
    csv_path = ctx.tables / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    md_path = ctx.tables / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    txt_path = ctx.tables / f"{name}.txt"
    txt_path.write_text("\n".join(sources) + "\n", encoding="utf-8")
    return [csv_path, md_path, txt_path]


def save_figure(ctx: Context, name: str, figure: Figure, sources) -> list[Path]:
    ctx.figures.mkdir(parents=True, exist_ok=True)
    pdf = ctx.figures / f"{name}.pdf"
    png = ctx.figures / f"{name}.png"
    figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png, dpi=ctx.dpi)
    figure.clear()
    txt = ctx.figures / f"{name}.txt"
    txt.write_text("\n".join(sources) + "\n", encoding="utf-8")
    return [pdf, png, txt]


def validation_line(metrics: dict, run: str) -> str:
    return (
        f"{run} ({seed_of(run)}): {metrics['n_val_images']} validation images in "
        f"{metrics['n_val_sessions']} sessions"
    )


def leakage(ctx: Context, spec: dict) -> list[Path]:
    entries = []
    for model in spec["models"]:
        path = ctx.results / "evaluation" / f"summary_{model}" / "leakage.csv"
        need(ctx, path)
        rows = [row for row in read_rows(path) if row["model"] == model]
        if len(rows) != 1:
            raise Skip(f"{ctx.rel(path)} has no single row for model {model}")
        row = rows[0]
        runs = {arm: row[f"{arm}_run"] for arm in ("session", "random")}
        paths = {
            arm: ctx.results / "evaluation" / run / "metrics.json" for arm, run in runs.items()
        }
        need(ctx, *paths.values())
        metrics = {arm: read_json(p) for arm, p in paths.items()}
        for arm in ("session", "random"):
            stored = number(metrics[arm]["primary"]["value"])
            if stored != number(row[f"{arm}_ap50"]):
                raise Mismatch(
                    f"{ctx.rel(path)} {arm}_ap50 {row[f'{arm}_ap50']} differs from "
                    f"{ctx.rel(paths[arm])} primary value {stored}"
                )
        entries.append((model, row, runs, paths, metrics))

    with rc_context(STYLE):
        figure = Figure(figsize=(6.6, 2.8), layout="constrained")
        left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        positions = np.arange(len(entries))
        for offset, arm, color, marker in (
            (-0.12, "session", PALETTE[0], "o"),
            (0.12, "random", PALETTE[1], "s"),
        ):
            values = np.array([interval(e[4][arm]["primary"]) for e in entries])
            left.errorbar(
                positions + offset,
                values[:, 0],
                yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]],
                fmt=marker,
                color=color,
                capsize=3,
                label=f"{arm} split, seed 0",
            )
        left.set_xticks(positions, [e[0] for e in entries])
        left.set_xlabel("Model")
        left.set_ylabel("Validation mask AP50 (fraction, 0 to 1)")
        figure.legend(loc="outside upper center", ncols=2, frameon=False)
        diffs = np.array(
            [[number(e[1][k]) for k in ("difference", "low", "high")] for e in entries]
        )
        right.errorbar(
            positions,
            diffs[:, 0],
            yerr=[diffs[:, 0] - diffs[:, 1], diffs[:, 2] - diffs[:, 0]],
            fmt="D",
            color=PALETTE[2],
            capsize=3,
        )
        right.axhline(0, color="#999999", linewidth=0.8)
        right.set_xticks(positions, [e[0] for e in entries])
        right.set_xlabel("Model")
        right.set_ylabel("Random minus session AP50 (fraction)")
        right.set_xlim(-0.5, len(entries) - 0.5)
        left.set_xlim(-0.5, len(entries) - 0.5)

    header = [
        "model",
        "session_run",
        "session_ap50",
        "session_low",
        "session_high",
        "random_run",
        "random_ap50",
        "random_low",
        "random_high",
        "difference",
        "low",
        "high",
        "inflation_claimed",
    ]
    rows = []
    for model, row, runs, _, metrics in entries:
        session, random = metrics["session"]["primary"], metrics["random"]["primary"]
        rows.append(
            [
                model,
                runs["session"],
                row["session_ap50"],
                text_of(session["low"]),
                text_of(session["high"]),
                runs["random"],
                row["random_ap50"],
                text_of(random["low"]),
                text_of(random["high"]),
                row["difference"],
                row["low"],
                row["high"],
                row["inflation_claimed"],
            ]
        )
    sources = ["Validation mask AP50, session against random split, 95% session bootstrap."]
    for model, _, runs, paths, metrics in entries:
        sources.append(
            f"read: {ctx.rel(ctx.results / 'evaluation' / f'summary_{model}' / 'leakage.csv')}"
        )
        for arm in ("session", "random"):
            sources.append(f"read: {ctx.rel(paths[arm])}")
            sources.append(validation_line(metrics[arm], runs[arm]))
    name = spec["name"]
    return save_figure(ctx, name, figure, sources) + write_table(ctx, name, header, rows, sources)


def forest(ctx: Context, spec: dict) -> list[Path]:
    found = []
    for comparison in spec["comparisons"]:
        folder = ctx.results / "evaluation" / f"compare_{comparison}"
        need(ctx, folder / "comparison.csv", folder / "comparison.json")
        found.append(
            (
                comparison,
                read_rows(folder / "comparison.csv"),
                read_json(folder / "comparison.json"),
                folder,
            )
        )
    metrics = spec.get("metrics", ["mask_ap50", "tolerant_ap50"])
    order = ["mean", *CLASSES]
    labels = []
    for comparison, _, _, _ in found:
        labels += [f"{comparison}: {item}" for item in order]
    seeds = sorted(
        {seed_of(info[key]) for _, _, info, _ in found for key in ("baseline", "candidate")}
    )

    with rc_context(STYLE):
        height = 0.9 + 0.17 * len(labels)
        figure = Figure(figsize=(3.3 * len(metrics) + 1.2, height), layout="constrained")
        axes = np.atleast_1d(figure.subplots(1, len(metrics), sharey=True))
        positions = np.arange(len(labels))[::-1]
        for ax, metric in zip(axes, metrics, strict=True):
            y = 0
            for _, rows, _, _ in found:
                by_class = {row["class"]: row for row in rows if row["metric"] == metric}
                for item in order:
                    row = by_class.get(item)
                    if row is not None:
                        value, low, high = (number(row[k]) for k in ("difference", "low", "high"))
                        color, marker = VERDICT_STYLE[row["verdict"]]
                        size = 6 if item == "mean" else 4
                        ax.errorbar(
                            value,
                            positions[y],
                            xerr=[[value - low], [high - value]],
                            fmt=marker,
                            color=color,
                            markersize=size,
                            capsize=2,
                        )
                    y += 1
            for boundary in range(len(order), len(labels), len(order)):
                ax.axhline(positions[boundary] + 0.5, color="#dddddd", linewidth=0.6)
            ax.axvline(0, color="#999999", linewidth=0.8)
            name = "Mask AP50" if metric == "mask_ap50" else "Tolerant AP50"
            ax.set_xlabel(f"{name} difference (fraction)")
        axes[0].set_yticks(positions, labels)
        handles = [
            Line2D([], [], color=c, marker=m, linestyle="", label=f"verdict: {v}")
            for v, (c, m) in VERDICT_STYLE.items()
        ]
        figure.supxlabel("Validation, candidate minus baseline, 95% paired session bootstrap")
        figure.legend(
            handles=handles,
            title=", ".join(seeds),
            loc="outside upper center",
            ncols=len(handles),
            frameon=False,
        )

    header = [
        "comparison",
        "baseline_run",
        "candidate_run",
        "metric",
        "class",
        "baseline",
        "candidate",
        "difference",
        "low",
        "high",
        "verdict",
    ]
    rows = []
    sources = ["Paired differences, candidate minus baseline, 95% paired session bootstrap."]
    for comparison, table, info, folder in found:
        for row in table:
            rows.append(
                [
                    comparison,
                    info["baseline"],
                    info["candidate"],
                    row["metric"],
                    row["class"],
                    row["baseline"],
                    row["candidate"],
                    row["difference"],
                    row["low"],
                    row["high"],
                    row["verdict"],
                ]
            )
        sources.append(f"read: {ctx.rel(folder / 'comparison.csv')}")
        sources.append(f"read: {ctx.rel(folder / 'comparison.json')}")
        sources.append(
            f"{comparison}: {info['candidate']} against {info['baseline']}, "
            f"{info['n_val_images']} validation images in {info['n_val_sessions']} sessions, "
            f"{info['resamples']} resamples"
        )
    name = spec["name"]
    return save_figure(ctx, name, figure, sources) + write_table(ctx, name, header, rows, sources)


def pick(header: list[str], candidates: list[str], path: str) -> str:
    for column in candidates:
        if column in header:
            return column
    raise Skip(f"{path} has none of the columns {candidates}")


def validation_count(ctx: Context, step: str, run: str) -> str:
    split = ctx.results / step / run / "split.csv"
    images = sum(row["split"] == "val" for row in read_rows(split)) if split.is_file() else None
    evaluated = ctx.results / "evaluation" / run / "metrics.json"
    sessions = read_json(evaluated)["n_val_sessions"] if evaluated.is_file() else None
    image_text = (
        f"{images} validation images" if images is not None else "validation images not recorded"
    )
    session_text = (
        f"{sessions} sessions (from {ctx.rel(evaluated)})"
        if sessions is not None
        else "sessions not recorded in the files read"
    )
    return f"{run} ({seed_of(run)}): {image_text}, {session_text}"


def curves(ctx: Context, spec: dict) -> list[Path]:
    paths = [ctx.results / r["step"] / r["run"] / "curves.csv" for r in spec["runs"]]
    need(ctx, *paths)
    series = []
    for entry, path in zip(spec["runs"], paths, strict=True):
        rows = read_rows(path)
        header = list(rows[0]) if rows else []
        epoch = pick(header, spec.get("epoch_columns", ["epoch"]), ctx.rel(path))
        loss = pick(header, spec["loss_columns"], ctx.rel(path))
        ap = pick(header, spec["ap_columns"], ctx.rel(path))
        values = {
            k: np.array([number(row[c]) for row in rows])
            for k, c in (("epoch", epoch), ("loss", loss), ("ap", ap))
        }
        series.append((entry, values, loss, ap))

    with rc_context(STYLE):
        figure = Figure(figsize=(6.6, 3.1), layout="constrained")
        left, right = figure.subplots(1, 2)
        for index, (entry, values, _, _) in enumerate(series):
            color = PALETTE[index % len(PALETTE)]
            label = f"{entry.get('label', entry['run'])}, {seed_of(entry['run'])}"
            left.plot(values["epoch"], values["loss"], color=color, linewidth=1.1)
            right.plot(values["epoch"], values["ap"], color=color, linewidth=1.1, label=label)
        left.set_xlabel("Epoch")
        left.set_ylabel("Mean training loss per epoch")
        right.set_xlabel("Epoch")
        right.set_ylabel("Validation mask AP50 (fraction)")
        figure.legend(loc="outside upper center", ncols=3, frameon=False)

    sources = [
        "Training loss and validation mask AP50 per epoch, as logged by each trainer.",
        "Per epoch values from one run each; the files hold no interval for them.",
        "The loss column sums the loss terms each arm trains with, so an arm that adds terms "
        "sums more of them. Validation mask AP50 here is the trainer's own per epoch log.",
    ]
    for (entry, _, loss, ap), path in zip(series, paths, strict=True):
        sources.append(f"read: {ctx.rel(path)} (columns {loss}, {ap})")
        sources.append(validation_count(ctx, entry["step"], entry["run"]))
    return save_figure(ctx, spec["name"], figure, sources)


PER_CLASS_COLUMNS = (
    "gt",
    "ap50",
    "ap50_low",
    "ap50_high",
    "tolerant_ap50",
    "tolerant_ap50_low",
    "tolerant_ap50_high",
)


def per_class(ctx: Context, spec: dict) -> list[Path]:
    folders = [ctx.results / "evaluation" / run for run in spec["runs"]]
    need(ctx, *[f / name for f in folders for name in ("per_class.csv", "metrics.json")])
    rows = []
    sources = ["Validation mask AP50 and tolerant AP50 (2 px), 95% session bootstrap."]
    for run, folder in zip(spec["runs"], folders, strict=True):
        metrics = read_json(folder / "metrics.json")
        for row in read_rows(folder / "per_class.csv"):
            rows.append([run, row["class"], *(row[c] for c in PER_CLASS_COLUMNS)])
        mask, tolerant = metrics["mask_ap50"]["mean"], metrics["tolerant_ap50"]["mean"]
        rows.append(
            [
                run,
                "mean",
                "",
                *(text_of(mask[k]) for k in ("value", "low", "high")),
                *(text_of(tolerant[k]) for k in ("value", "low", "high")),
            ]
        )
        sources.append(f"read: {ctx.rel(folder / 'per_class.csv')}")
        sources.append(f"read: {ctx.rel(folder / 'metrics.json')}")
        sources.append(validation_line(metrics, run))
    return write_table(ctx, spec["name"], ["run", "class", *PER_CLASS_COLUMNS], rows, sources)


def copy_table(ctx: Context, name: str, path: Path, note: str) -> list[Path]:
    need(ctx, path)
    rows = read_rows(path)
    header = list(rows[0]) if rows else []
    body = [[row[c] for c in header] for row in rows]
    return write_table(ctx, name, header, body, [note, f"read: {ctx.rel(path)}"])


def class_counts(ctx: Context, name: str) -> list[Path]:
    path = ctx.results / "audit" / "integrity" / "label_counts.json"
    need(ctx, path)
    counts = read_json(path)
    rows = [
        [cls, counts["per_class"][cls]["valid_polygons"], counts["per_class"][cls]["images"]]
        for cls in CLASSES
    ]
    rows.append(["all classes", counts["valid_polygons"], ""])
    rows.append(["empty label file", "", counts["empty_images"]])
    rows.append(["all images", "", counts["images"]])
    sources = [
        "Instances (valid polygons) and images per class, all pooled images, no split.",
        f"read: {ctx.rel(path)}",
    ]
    return write_table(ctx, name, ["class", "instances", "images"], rows, sources)


COPIED_TABLES = {
    "day_class": (
        ("integrity", "day_class_table.csv"),
        "Images, sessions, empty images and instances per capture day, all pooled images.",
    ),
    "size_strata": (
        ("geometry", "size_strata.csv"),
        "Share of instances by bounding box area in pixels (under 32x32, 32x32 to 96x96, "
        "over 96x96), per class.",
    ),
    "width_strata": (
        ("geometry", "width_strata.csv"),
        "Share of instances by median width in pixels (up to 2, over 2 to 4, over 4 to 8, "
        "over 8), per class.",
    ),
}
DATASET_TABLES = ("class_counts", *COPIED_TABLES)


def dataset_unit(table: str):
    def run(ctx: Context, spec: dict) -> list[Path]:
        name = f"{spec['name']}_{table}"
        if table == "class_counts":
            return class_counts(ctx, name)
        (folder, file_name), note = COPIED_TABLES[table]
        return copy_table(ctx, name, ctx.results / "audit" / folder / file_name, note)

    return run


def pr_curve(scores, images, orders, hits, n_gt: int):
    order = np.lexsort((orders, images, -scores))
    tp = hits[order].astype(float)
    if n_gt == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), math.nan
    if tp.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), 0.0
    found = np.cumsum(tp)
    recall = found / n_gt
    precision = found / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_LEVELS, side="left")
    sampled = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0.0)
    return recall, precision, envelope, float(sampled.mean())


def same(computed: float, stored) -> bool:
    if stored is None:
        return math.isnan(computed)
    return not math.isnan(computed) and round(computed, 6) == float(stored)


def pr_unit(run: str):
    def draw(ctx: Context, spec: dict) -> list[Path]:
        folder = ctx.results / "evaluation" / run
        files = ("scored_predictions.csv", "scored_truth.csv", "scored_images.csv", "metrics.json")
        need(ctx, *(folder / f for f in files))
        preds = read_rows(folder / "scored_predictions.csv")
        truth = read_rows(folder / "scored_truth.csv")
        images = read_rows(folder / "scored_images.csv")
        metrics = read_json(folder / "metrics.json")
        column = {k: np.array([number(row[k]) for row in preds], dtype=float) for k in PR_COLUMNS}
        gt_class = np.array([int(row["gt_class"]) for row in truth], dtype=int)
        curves_by_class, computed = {}, []
        for class_id, cls in enumerate(CLASSES):
            members = column["pred_class"] == class_id
            n_gt = int((gt_class == class_id).sum())
            recall, precision, envelope, ap = pr_curve(
                *(column[k][members] for k in PR_COLUMNS[1:]), n_gt
            )
            stored = metrics["mask_ap50"]["per_class"][cls]
            if not same(ap, stored["value"]):
                raise Mismatch(
                    f"{run} {cls}: AP50 from the stored scores is {ap:.6f}, "
                    f"metrics.json has {stored['value']}"
                )
            computed.append(ap)
            curves_by_class[cls] = (recall, precision, envelope, stored, n_gt, int(members.sum()))
        finite = [v for v in computed if not math.isnan(v)]
        mean = float(np.mean(finite)) if finite else math.nan
        if not same(mean, metrics["mask_ap50"]["mean"]["value"]):
            raise Mismatch(
                f"{run}: mean AP50 from the stored scores is {mean:.6f}, "
                f"metrics.json has {metrics['mask_ap50']['mean']['value']}"
            )

        with rc_context(STYLE):
            figure = Figure(figsize=(6.6, 5.0), layout="constrained")
            axes = figure.subplots(2, 2, sharex=True, sharey=True).ravel()
            for ax, (cls, (recall, precision, envelope, stored, n_gt, _)) in zip(
                axes, curves_by_class.items(), strict=True
            ):
                if len(recall):
                    ax.plot(
                        recall,
                        precision,
                        color=PALETTE[5],
                        linewidth=0.6,
                        label="precision at each prediction",
                    )
                    value, low, high = interval(stored)
                    ax.step(
                        np.r_[0.0, recall],
                        np.r_[envelope[0], envelope],
                        where="pre",
                        color=PALETTE[0],
                        linewidth=1.2,
                        label=f"AP50 {value:.3f} [{low:.3f}, {high:.3f}], {seed_of(run)}",
                    )
                    ax.legend(frameon=False, loc="upper right")
                else:
                    ax.text(0.5, 0.5, "no predictions or no ground truth", ha="center", va="center")
                ax.set_title(f"{cls} ({n_gt} ground truth instances)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.05)
            for ax in axes[2:]:
                ax.set_xlabel("Recall (fraction of ground truth instances)")
            for ax in axes[::2]:
                ax.set_ylabel("Precision (fraction of predictions)")

        sources = [
            "Precision and recall per class at mask IoU 0.5, from the evaluator's stored "
            "match flags, ordered by score, then image, then file order. The step line is the "
            "running maximum precision used for AP50; the interval is the 95% session bootstrap.",
            f"Recomputed AP50 equals metrics.json for every class and the mean ({mean:.6f}).",
        ]
        sources += [f"read: {ctx.rel(folder / f)}" for f in files]
        sources.append(validation_line(metrics, run))
        sources.append(
            f"scored images: {len(images)}, predictions: {len(preds)}, ground truth: {len(truth)}"
        )
        return save_figure(ctx, f"{spec['name']}_{run}", figure, sources)

    return draw


def units(spec: dict) -> list[tuple[str, object]]:
    kind = spec["kind"]
    if kind == "dataset":
        return [
            (f"{spec['name']}_{t}", dataset_unit(t)) for t in spec.get("tables", DATASET_TABLES)
        ]
    if kind == "pr":
        return [(f"{spec['name']}_{run}", pr_unit(run)) for run in spec["runs"]]
    single = {"leakage": leakage, "forest": forest, "curves": curves, "per_class": per_class}
    if kind not in single:
        raise ValueError(f"unknown figure kind {kind}")
    return [(spec["name"], single[kind])]


def build(ctx: Context, specs: list[dict]) -> dict[str, list]:
    report = {"written": [], "skipped": []}
    for spec in specs:
        for name, make in units(spec):
            try:
                paths = make(ctx, spec)
            except Skip as reason:
                print(f"{name}: skipped, {reason}")
                report["skipped"].append((name, str(reason)))
                continue
            for path in paths:
                print(f"{name}: wrote {ctx.rel(path)}")
            report["written"].append(name)
            gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    root = runio.repo_root()
    ctx = Context(
        root=root,
        results=root / config["results"],
        figures=root / config["figures_dir"],
        tables=root / config["tables_dir"],
        dpi=config.get("dpi", 200),
    )
    try:
        report = build(ctx, config["figures"])
    except Mismatch as error:
        raise SystemExit(f"stopped: {error}") from error
    print(f"produced {len(report['written'])}, skipped {len(report['skipped'])}")
    for name, reason in report["skipped"]:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
