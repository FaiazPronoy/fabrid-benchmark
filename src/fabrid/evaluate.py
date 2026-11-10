import argparse
import csv
import json
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fabrid import runio
from fabrid.geometry import rasterize
from fabrid.labels import CLASSES, load_index
from fabrid.metrics import cldice, mask_iou, size_bucket, tolerant_iou
from fabrid.sessions import assign_sessions, session_split
from fabrid.yolo import assert_unlocked, write_csv

BUCKETS = ("small", "medium", "large")
RECALL_LEVELS = np.linspace(0, 1, 101)
YARN = CLASSES.index("missing yarn")
RUN_NAME = re.compile(r"^(?P<model>.+)_(?P<arm>session|random)_seed\d+$")
DEFAULT_ROOTS = ("/kaggle/working/runs", "/kaggle/input")
COMPARED = {"ap50": "mask_ap50", "tolerant_ap50": "tolerant_ap50"}
COMPARISON_FIELDS = ("baseline", "candidate", "difference", "low", "high", "verdict")


@dataclass
class Instance:
    class_id: int
    mask: np.ndarray
    box: tuple[int, int, int, int] | None
    score: float = 0.0


@dataclass
class Scored:
    pred_image: np.ndarray
    pred_class: np.ndarray
    pred_score: np.ndarray
    pred_order: np.ndarray
    pred_tp: np.ndarray
    pred_tp_tolerant: np.ndarray
    pred_bucket: np.ndarray
    pred_matched_bucket: np.ndarray
    gt_image: np.ndarray
    gt_class: np.ndarray
    gt_bucket: np.ndarray
    gt_matched: np.ndarray
    gt_cldice: np.ndarray
    n_images: int


def rle_counts(text: str) -> list[int]:
    counts = []
    position = 0
    while position < len(text):
        value, shift, more = 0, 0, True
        while more:
            code = ord(text[position]) - 48
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            position += 1
            shift += 1
            if not more and code & 0x10:
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=bool)
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if len(points) >= 3:
                mask |= rasterize(points, height, width)
        return mask
    if [int(v) for v in segmentation["size"]] != [height, width]:
        raise ValueError(f"mask size {segmentation['size']} differs from image {height}x{width}")
    counts = segmentation["counts"]
    counts = rle_counts(counts) if isinstance(counts, str) else [int(c) for c in counts]
    if sum(counts) != height * width:
        raise ValueError("run lengths do not cover the image")
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[position : position + count] = True
        position += count
    return np.ascontiguousarray(flat.reshape(width, height).T)


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def bucket_of(box) -> int:
    area = 0 if box is None else (box[1] - box[0]) * (box[3] - box[2])
    return BUCKETS.index(size_bucket(area))


def crop(first, second, margin: int, shape) -> tuple[slice, slice] | None:
    if first is None or second is None:
        return None
    top, bottom = max(first[0], second[0]), min(first[1], second[1])
    left, right = max(first[2], second[2]), min(first[3], second[3])
    if top - bottom >= 2 * margin or left - right >= 2 * margin:
        return None
    rows = slice(
        max(0, min(first[0], second[0]) - margin), min(shape[0], max(first[1], second[1]) + margin)
    )
    cols = slice(
        max(0, min(first[2], second[2]) - margin), min(shape[1], max(first[3], second[3]) + margin)
    )
    return rows, cols


def plain_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(mask_iou(pred, gt)[0, 0])


def overlap_matrix(preds, gts, score, margin: int) -> np.ndarray:
    matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            window = crop(pred.box, gt.box, margin, gt.mask.shape)
            if window is not None:
                matrix[i, j] = score(pred.mask[window], gt.mask[window])
    return matrix


def greedy(iou: np.ndarray, scores, threshold: float) -> list[int | None]:
    n_pred, n_gt = iou.shape
    taken = np.zeros(n_gt, dtype=bool)
    matched: list[int | None] = [None] * n_pred
    for i in sorted(range(n_pred), key=lambda k: (-scores[k], k)):
        if n_gt == 0:
            continue
        candidates = np.where(taken, -1.0, iou[i])
        j = int(np.argmax(candidates))
        if candidates[j] >= threshold and candidates[j] > 0:
            taken[j] = True
            matched[i] = j
    return matched


def find_file(path: str | Path, roots) -> Path | None:
    path = Path(path)
    if path.is_absolute():
        return path if path.is_file() else None
    for root in roots:
        candidate = Path(root) / path
        if candidate.is_file():
            return candidate
    for root in roots:
        if Path(root).is_dir():
            found = sorted(Path(root).glob(f"**/{path.as_posix()}"))
            if found:
                return found[0]
    return None


def read_split(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row["split"] for row in csv.DictReader(handle)}


def image_ids(predictions_path: Path) -> dict[str, str]:
    truth = predictions_path.with_name("val_ground_truth.json")
    if not truth.is_file():
        return {}
    images = json.loads(truth.read_text(encoding="utf-8"))["images"]
    return {str(image["id"]): image["file_name"] for image in images}


def resolve(predictions, lookup: dict[str, str], ids: dict[str, str]) -> list[str]:
    names = []
    for item in predictions:
        if item.get("file_name"):
            key = Path(item["file_name"]).name
        elif str(item["image_id"]) in ids:
            key = ids[str(item["image_id"])]
        elif isinstance(item["image_id"], str):
            key = item["image_id"]
        else:
            raise ValueError(f"cannot map image_id {item['image_id']} to a file name")
        if key not in lookup:
            raise ValueError(f"prediction for unknown image {key}")
        names.append(lookup[key])
    return names


def guard(names, mapping: dict[str, str], locked: dict[str, str]) -> None:
    assert_unlocked(mapping, locked)
    outside = sorted({n for n in names if n not in mapping or locked.get(n) == "test"})
    if outside:
        raise AssertionError(f"predictions for images outside train and val: {outside[:5]}")


def ground_truth(record, height: int, width: int) -> tuple[list[Instance], int]:
    instances, empty = [], 0
    for class_id, points in record.polygons:
        mask = rasterize(points * [width, height], height, width)
        box = bounding_box(mask)
        if box is None:
            empty += 1
            continue
        instances.append(Instance(class_id, mask, box))
    return instances, empty


def score_images(records, predictions, names, val_names, config: dict, offset: int):
    by_name = {record.name: record for record in records}
    per_image = {name: [] for name in val_names}
    for item, name in zip(predictions, names, strict=True):
        if name in per_image:
            per_image[name].append(item)
    tolerance = config["tolerance_px"]

    def tolerant(pred: np.ndarray, gt: np.ndarray) -> float:
        return tolerant_iou(pred, gt, tolerance)

    tolerant_margin = math.ceil(tolerance) + 1
    cldice_margin = math.ceil(config["cldice_tol_px"]) + 2
    pred_rows, gt_rows, dropped = [], [], 0
    for image_index, name in enumerate(val_names):
        record = by_name[name]
        with Image.open(record.path) as image:
            width, height = image.size
        gts, empty = ground_truth(record, height, width)
        dropped += empty
        preds = []
        for item in per_image[name]:
            class_id = int(item["category_id"]) - offset
            if not 0 <= class_id < len(CLASSES):
                raise ValueError(f"category id {item['category_id']} out of range")
            mask = decode(item["segmentation"], height, width)
            preds.append(Instance(class_id, mask, bounding_box(mask), float(item["score"])))
        for class_id in range(len(CLASSES)):
            pred_ids = [i for i, p in enumerate(preds) if p.class_id == class_id]
            group_preds = [preds[i] for i in pred_ids]
            group_gts = [g for g in gts if g.class_id == class_id]
            scores = [p.score for p in group_preds]
            plain = overlap_matrix(group_preds, group_gts, plain_iou, 0)
            loose = overlap_matrix(group_preds, group_gts, tolerant, tolerant_margin)
            matched = greedy(plain, scores, config["iou"])
            matched_loose = greedy(loose, scores, config["iou"])
            gt_cldice = [np.nan] * len(group_gts)
            for k, pred in enumerate(group_preds):
                j = matched[k]
                if j is not None and class_id == YARN:
                    window = crop(pred.box, group_gts[j].box, cldice_margin, pred.mask.shape)
                    gt_cldice[j] = cldice(
                        pred.mask[window], group_gts[j].mask[window], config["cldice_tol_px"]
                    )
                pred_rows.append(
                    (
                        image_index,
                        class_id,
                        pred.score,
                        pred_ids[k],
                        j is not None,
                        matched_loose[k] is not None,
                        bucket_of(pred.box),
                        bucket_of(group_gts[j].box) if j is not None else -1,
                    )
                )
            hit = {j for j in matched if j is not None}
            for j, gt in enumerate(group_gts):
                gt_rows.append((image_index, class_id, bucket_of(gt.box), j in hit, gt_cldice[j]))
    columns = list(zip(*pred_rows, strict=True)) if pred_rows else [()] * 8
    gt_columns = list(zip(*gt_rows, strict=True)) if gt_rows else [()] * 5
    scored = Scored(
        pred_image=np.array(columns[0], dtype=int),
        pred_class=np.array(columns[1], dtype=int),
        pred_score=np.array(columns[2], dtype=float),
        pred_order=np.array(columns[3], dtype=int),
        pred_tp=np.array(columns[4], dtype=bool),
        pred_tp_tolerant=np.array(columns[5], dtype=bool),
        pred_bucket=np.array(columns[6], dtype=int),
        pred_matched_bucket=np.array(columns[7], dtype=int),
        gt_image=np.array(gt_columns[0], dtype=int),
        gt_class=np.array(gt_columns[1], dtype=int),
        gt_bucket=np.array(gt_columns[2], dtype=int),
        gt_matched=np.array(gt_columns[3], dtype=bool),
        gt_cldice=np.array(gt_columns[4], dtype=float),
        n_images=len(val_names),
    )
    return scored, dropped


PRED_COLUMNS = (
    ("pred_image", int),
    ("pred_class", int),
    ("pred_score", float),
    ("pred_order", int),
    ("pred_tp", bool),
    ("pred_tp_tolerant", bool),
    ("pred_bucket", int),
    ("pred_matched_bucket", int),
)
GT_COLUMNS = (
    ("gt_image", int),
    ("gt_class", int),
    ("gt_bucket", int),
    ("gt_matched", bool),
    ("gt_cldice", float),
)
SCORED_FILES = ("scored_images.csv", "scored_predictions.csv", "scored_truth.csv")


def cell(value, kind) -> str:
    if kind is bool:
        return str(int(value))
    if kind is int:
        return str(int(value))
    return "" if math.isnan(value) else repr(float(value))


def parse(text: str, kind):
    if kind is bool:
        return text == "1"
    if kind is int:
        return int(text)
    return float(text) if text else float("nan")


def save_scored(run_dir: Path, scored: Scored, val_names: list[str]) -> None:
    write_csv(run_dir / SCORED_FILES[0], ["image", "name"], list(enumerate(val_names)))
    for file_name, columns in zip(SCORED_FILES[1:], (PRED_COLUMNS, GT_COLUMNS), strict=True):
        arrays = [getattr(scored, name) for name, _ in columns]
        rows = [
            [cell(value, kind) for value, (_, kind) in zip(row, columns, strict=True)]
            for row in zip(*arrays, strict=True)
        ]
        write_csv(run_dir / file_name, [name for name, _ in columns], rows)


def load_scored(run_dir: Path) -> tuple[Scored, list[str]] | None:
    if not all((run_dir / name).is_file() for name in SCORED_FILES):
        return None
    with (run_dir / SCORED_FILES[0]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["image"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{run_dir / SCORED_FILES[0]}: image index is not 0..n-1")
    val_names = [row["name"] for row in rows]
    fields = {}
    for file_name, columns in zip(SCORED_FILES[1:], (PRED_COLUMNS, GT_COLUMNS), strict=True):
        with (run_dir / file_name).open(newline="", encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
        for name, kind in columns:
            fields[name] = np.array([parse(row[name], kind) for row in table], dtype=kind)
    for key in ("pred_image", "gt_image"):
        if fields[key].size and not 0 <= fields[key].min() <= fields[key].max() < len(val_names):
            raise ValueError(f"{run_dir}: {key} points outside the image list")
    return Scored(**fields, n_images=len(val_names)), val_names


def cached_run(run: dict, mapping: dict[str, str] | None, locked: dict[str, str], base: Path):
    loaded = load_scored(base / run["name"])
    metrics_path = base / run["name"] / "metrics.json"
    if loaded is None or not metrics_path.is_file():
        return None
    scored, val_names = loaded
    outside = sorted(n for n in val_names if locked.get(n) not in ("train", "val"))
    if outside:
        raise AssertionError(f"{run['name']}: stored scores cover locked images {outside[:5]}")
    if mapping is not None:
        expected = sorted(n for n, part in mapping.items() if part == "val")
        if expected != val_names:
            raise ValueError(f"{run['name']}: stored scores do not match the split file")
    return scored, val_names, json.loads(metrics_path.read_text(encoding="utf-8"))


def average_precision(tp: np.ndarray, weights: np.ndarray, n_gt: float) -> float:
    if n_gt <= 0:
        return float("nan")
    keep = weights > 0
    tp, weights = tp[keep], weights[keep]
    if tp.size == 0:
        return 0.0
    hits = np.cumsum(weights * tp)
    recall = hits / n_gt
    precision = hits / np.cumsum(weights)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_LEVELS, side="left")
    values = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0.0)
    return float(values.mean())


def best_f1(tp, scores, weights, n_gt: float) -> dict[str, float]:
    nan = float("nan")
    if n_gt <= 0:
        return {"threshold": nan, "precision": nan, "recall": nan, "f1": nan}
    keep = weights > 0
    tp, scores, weights = tp[keep], scores[keep], weights[keep]
    if tp.size == 0:
        return {"threshold": nan, "precision": nan, "recall": 0.0, "f1": 0.0}
    hits = np.cumsum(weights * tp)
    total = np.cumsum(weights)
    ends = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))
    precision = hits[ends] / total[ends]
    recall = hits[ends] / n_gt
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )
    best = int(np.argmax(f1))
    return {
        "threshold": float(scores[ends[best]]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }


def class_mean(values) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


class Scorer:
    def __init__(self, scored: Scored):
        self.scored = scored
        self.orders = {}
        for class_id in range(len(CLASSES)):
            members = np.flatnonzero(scored.pred_class == class_id)
            keys = (
                scored.pred_order[members],
                scored.pred_image[members],
                -scored.pred_score[members],
            )
            self.orders[class_id] = members[np.lexsort(keys)]

    def evaluate(self, image_weights: np.ndarray) -> dict[str, float]:
        s = self.scored
        out = {}
        plain, loose = [], []
        sizes = {b: [] for b in range(len(BUCKETS))}
        for class_id, name in enumerate(CLASSES):
            order = self.orders[class_id]
            weights = image_weights[s.pred_image[order]].astype(float)
            gt_rows = s.gt_class == class_id
            gt_weights = image_weights[s.gt_image[gt_rows]].astype(float)
            n_gt = float(gt_weights.sum())
            ap = average_precision(s.pred_tp[order], weights, n_gt)
            ap_loose = average_precision(s.pred_tp_tolerant[order], weights, n_gt)
            out[f"ap50/{name}"], out[f"tolerant_ap50/{name}"] = ap, ap_loose
            plain.append(ap)
            loose.append(ap_loose)
            point = best_f1(s.pred_tp[order], s.pred_score[order], weights, n_gt)
            for key, value in point.items():
                out[f"f1_point/{name}/{key}"] = value
            for b, bucket in enumerate(BUCKETS):
                matched = s.pred_matched_bucket[order]
                tp = s.pred_tp[order]
                inside = np.where(tp, matched == b, s.pred_bucket[order] == b)
                n_bucket = float(gt_weights[s.gt_bucket[gt_rows] == b].sum())
                value = average_precision(tp[inside], weights[inside], n_bucket)
                out[f"size_ap50/{bucket}/{name}"] = value
                sizes[b].append(value)
        out["ap50"] = class_mean(plain)
        out["tolerant_ap50"] = class_mean(loose)
        for b, bucket in enumerate(BUCKETS):
            out[f"size_ap50/{bucket}"] = class_mean(sizes[b])
        yarn = s.gt_class == YARN
        yarn_weights = image_weights[s.gt_image[yarn]].astype(float)
        total = yarn_weights.sum()
        hit = s.gt_matched[yarn]
        out["missing_yarn/matched_share"] = (
            float((yarn_weights * hit).sum() / total) if total else float("nan")
        )
        hit_weights = yarn_weights[hit]
        out["missing_yarn/cldice"] = (
            float((hit_weights * s.gt_cldice[yarn][hit]).sum() / hit_weights.sum())
            if hit_weights.sum()
            else float("nan")
        )
        return out


def draws(sessions: np.ndarray, resamples: int, seed: int, name: str) -> list[np.ndarray]:
    labels, index = np.unique(sessions, return_inverse=True)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    picks = rng.integers(0, len(labels), size=(resamples, len(labels)))
    return [np.bincount(row, minlength=len(labels))[index] for row in picks]


def bootstrap(scorer: Scorer, weights: list[np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    point = scorer.evaluate(np.ones(scorer.scored.n_images))
    samples = [scorer.evaluate(w) for w in weights]
    resampled = {key: np.array([sample[key] for sample in samples]) for key in point}
    estimates = {}
    for key, value in point.items():
        values = resampled[key][~np.isnan(resampled[key])]
        low, high = np.percentile(values, [2.5, 97.5]) if values.size else (np.nan, np.nan)
        estimates[key] = {"value": clean(value), "low": clean(low), "high": clean(high)}
    return estimates, resampled


def clean(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 6)


def run_identity(run: dict) -> tuple[str | None, str | None]:
    match = RUN_NAME.match(run["name"])
    model = run.get("model") or (match["model"] if match else None)
    arm = run.get("arm") or (match["arm"] if match else None)
    return model, arm


def per_class_rows(estimates: dict, gt_counts: dict[str, int]) -> list[list]:
    rows = []
    for name in CLASSES:
        ap, loose = estimates[f"ap50/{name}"], estimates[f"tolerant_ap50/{name}"]
        point = [
            estimates[f"f1_point/{name}/{k}"]["value"]
            for k in ("threshold", "precision", "recall", "f1")
        ]
        sizes = [estimates[f"size_ap50/{b}/{name}"]["value"] for b in BUCKETS]
        rows.append(
            [
                name,
                gt_counts[name],
                ap["value"],
                ap["low"],
                ap["high"],
                loose["value"],
                loose["low"],
                loose["high"],
                *point,
                *sizes,
            ]
        )
    return rows


def run_metrics(run, model, arm, paths, scored, dropped, estimates, extra) -> dict:
    nested_classes = {
        "mask_ap50": {
            "mean": estimates["ap50"],
            "per_class": {c: estimates[f"ap50/{c}"] for c in CLASSES},
        },
        "tolerant_ap50": {
            "mean": estimates["tolerant_ap50"],
            "per_class": {c: estimates[f"tolerant_ap50/{c}"] for c in CLASSES},
        },
        "f1_operating_point": {
            c: {
                k: estimates[f"f1_point/{c}/{k}"]
                for k in ("threshold", "precision", "recall", "f1")
            }
            for c in CLASSES
        },
        "size_ap50": {
            b: {
                "mean": estimates[f"size_ap50/{b}"],
                "per_class": {c: estimates[f"size_ap50/{b}/{c}"] for c in CLASSES},
                "gt_instances": int((scored.gt_bucket == i).sum()),
            }
            for i, b in enumerate(BUCKETS)
        },
        "missing_yarn": {
            "cldice": estimates["missing_yarn/cldice"],
            "matched_share": estimates["missing_yarn/matched_share"],
        },
    }
    return {
        "run": run["name"],
        "model": model,
        "arm": arm,
        "predictions": str(paths[0]),
        "split_csv": str(paths[1]),
        **extra,
        "gt_without_pixels": dropped,
        "primary": {"metric": "mask AP50", **estimates["ap50"]},
        **nested_classes,
    }


def evaluate_run(run: dict, records, locked, sessions, config: dict, roots):
    model, arm = run_identity(run)
    step = run.get("step", "baselines")
    predictions_rel = run.get("predictions", f"{step}/{run['name']}/val_predictions.json")
    predictions_path = find_file(predictions_rel, roots)
    split_rel = run.get("split", f"results/{step}/{run['name']}/split.csv")
    split_path = find_file(split_rel, [runio.repo_root(), *roots])
    mapping = read_split(split_path) if split_path is not None else None
    base = runio.repo_root() / "results" / config["step"]
    fresh = predictions_path is not None and split_path is not None
    if fresh:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        lookup = {}
        for record in records:
            lookup[record.name] = record.name
            lookup[Path(record.name).stem] = record.name
        names = resolve(predictions, lookup, image_ids(predictions_path))
        guard(names, mapping, locked)
        val_names = sorted(n for n, s in mapping.items() if s == "val")
        dropped_train = sum(mapping[n] == "train" for n in names)
        offset = int(run.get("category_offset", 1))
        scored, empty = score_images(records, predictions, names, val_names, config, offset)
    else:
        cached = cached_run(run, mapping, locked, base)
        if cached is None:
            missing = predictions_rel if predictions_path is None else split_rel
            print(f"{run['name']}: {missing} not found and no stored scores, skipped")
            return None
        scored, val_names, metrics = cached
    scorer = Scorer(scored)
    image_sessions = np.array([sessions[n] for n in val_names])
    weights = draws(image_sessions, config["resamples"], config["seed"], run["name"])
    estimates, resampled = bootstrap(scorer, weights)
    primary = estimates["ap50"]
    if fresh:
        gt_counts = {c: int((scored.gt_class == i).sum()) for i, c in enumerate(CLASSES)}
        extra = {
            "n_val_images": len(val_names),
            "n_val_sessions": len(set(image_sessions.tolist())),
            "n_predictions_scored": int(scored.pred_score.size),
            "n_predictions_train_dropped": int(dropped_train),
            "gt_instances": gt_counts,
            "resamples": config["resamples"],
            "seed": config["seed"],
        }
        run_dir = runio.start_run(config["step"], run["name"], {**config, "run": run})
        metrics = run_metrics(
            run, model, arm, (predictions_path, split_path), scored, empty, estimates, extra
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        header = [
            "class",
            "gt",
            "ap50",
            "ap50_low",
            "ap50_high",
            "tolerant_ap50",
            "tolerant_ap50_low",
            "tolerant_ap50_high",
            "f1_threshold",
            "f1_precision",
            "f1_recall",
            "f1",
            *[f"ap50_{b}" for b in BUCKETS],
        ]
        write_csv(run_dir / "per_class.csv", header, per_class_rows(estimates, gt_counts))
        save_scored(run_dir, scored, val_names)
        print(
            f"{run['name']}: mask AP50 {primary['value']} [{primary['low']}, {primary['high']}] "
            f"on {len(val_names)} val images, wrote {run_dir}"
        )
    else:
        stored = metrics["primary"]["value"]
        if stored != primary["value"]:
            raise ValueError(
                f"{run['name']}: stored scores give {primary['value']}, metrics.json says {stored}"
            )
        print(f"{run['name']}: mask AP50 {stored} read from stored scores in {base / run['name']}")
    return {
        "run": run["name"],
        "model": model,
        "arm": arm,
        "metrics": metrics,
        "draws": resampled["ap50"],
        "scorer": scorer,
        "val_names": val_names,
        "sessions": image_sessions,
        "fresh": fresh,
    }


def verdict(difference, low, high) -> str:
    if difference is None or low is None or high is None:
        return "not shown"
    if difference > 0 and low > 0:
        return "higher"
    if difference < 0 and high < 0:
        return "lower"
    return "not shown"


def paired(baseline: Scorer, candidate: Scorer, weights: list[np.ndarray]) -> dict[str, dict]:
    keys = [
        f"{metric}{suffix}" for metric in COMPARED for suffix in ("", *[f"/{c}" for c in CLASSES])
    ]
    ones = np.ones(baseline.scored.n_images)
    first, second = baseline.evaluate(ones), candidate.evaluate(ones)
    samples = [(baseline.evaluate(w), candidate.evaluate(w)) for w in weights]
    table = {}
    for key in keys:
        differences = np.array([b[key] - a[key] for a, b in samples])
        differences = differences[~np.isnan(differences)]
        low, high = (
            np.percentile(differences, [2.5, 97.5]) if differences.size else (np.nan, np.nan)
        )
        entry = {
            "baseline": clean(first[key]),
            "candidate": clean(second[key]),
            "difference": clean(second[key] - first[key]),
            "low": clean(low),
            "high": clean(high),
        }
        entry["verdict"] = verdict(entry["difference"], entry["low"], entry["high"])
        table[key] = entry
    return table


def compare(comparison: dict, found: dict[str, dict], config: dict) -> dict:
    baseline, candidate = found[comparison["baseline"]], found[comparison["candidate"]]
    if baseline["val_names"] != candidate["val_names"]:
        only = sorted(set(baseline["val_names"]) ^ set(candidate["val_names"]))
        raise ValueError(
            f"comparison {comparison['name']}: validation images differ, for example {only[:5]}"
        )
    weights = draws(baseline["sessions"], config["resamples"], config["seed"], comparison["name"])
    table = paired(baseline["scorer"], candidate["scorer"], weights)
    metrics = {
        COMPARED[metric]: {
            "mean": table[metric],
            "per_class": {c: table[f"{metric}/{c}"] for c in CLASSES},
        }
        for metric in COMPARED
    }
    return {
        "name": comparison["name"],
        "baseline": comparison["baseline"],
        "candidate": comparison["candidate"],
        "n_val_images": len(baseline["val_names"]),
        "n_val_sessions": len(set(baseline["sessions"].tolist())),
        "resamples": config["resamples"],
        "seed": config["seed"],
        "verdict": table["ap50"]["verdict"],
        **metrics,
    }


def comparison_rows(result: dict) -> list[list]:
    rows = []
    for name in COMPARED.values():
        entries = [("mean", result[name]["mean"]), *result[name]["per_class"].items()]
        for label, entry in entries:
            rows.append([name, label, *[entry[k] for k in COMPARISON_FIELDS]])
    return rows


def leakage(results: list[dict]) -> list[list]:
    rows = []
    models = sorted({r["model"] for r in results if r["model"]})
    for model in models:
        arms = {r["arm"]: r for r in results if r["model"] == model}
        if "session" not in arms or "random" not in arms:
            continue
        session, random = arms["session"], arms["random"]
        a = session["metrics"]["primary"]["value"]
        b = random["metrics"]["primary"]["value"]
        diffs = random["draws"] - session["draws"]
        diffs = diffs[~np.isnan(diffs)]
        if a is None or b is None or diffs.size == 0:
            continue
        low, high = np.percentile(diffs, [2.5, 97.5])
        difference = b - a
        claimed = bool(difference > 0 and low > 0)
        rows.append(
            [
                model,
                session["run"],
                random["run"],
                a,
                b,
                clean(difference),
                clean(low),
                clean(high),
                claimed,
            ]
        )
    return rows


def summary_text(results: list[dict], skipped: list[str], leak_rows: list[list]) -> str:
    lines = [
        "# Evaluation",
        "",
        "Validation only, one evaluator for every run. Intervals: 95% session bootstrap.",
        "",
        "| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |",
        "|---|---|---|---|---|---|",
    ]

    def show(estimate) -> str:
        if estimate["value"] is None:
            return "n/a"
        if estimate["low"] is None:
            return f"{estimate['value']:.3f}"
        return f"{estimate['value']:.3f} [{estimate['low']:.3f}, {estimate['high']:.3f}]"

    for result in results:
        m = result["metrics"]
        lines.append(
            f"| {result['run']} | {m['n_val_images']} | {m['n_val_sessions']} "
            f"| {show(m['primary'])} | {show(m['tolerant_ap50']['mean'])} "
            f"| {show(m['missing_yarn']['cldice'])} |"
        )
    lines += ["", "## Leakage", ""]
    if leak_rows:
        lines += [
            "| model | session AP50 | random AP50 | difference | 95% interval | inflation |",
            "|---|---|---|---|---|---|",
        ]
        for model, _, _, a, b, diff, low, high, claimed in leak_rows:
            lines.append(
                f"| {model} | {a:.3f} | {b:.3f} | {diff:.3f} | [{low:.3f}, {high:.3f}] "
                f"| {'yes' if claimed else 'no'} |"
            )
    else:
        lines.append("No model has both arms evaluated.")
    if skipped:
        lines += ["", "Runs listed but not found: " + ", ".join(skipped) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    roots = [Path(root) for root in config.get("roots", DEFAULT_ROOTS)]
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    locked = session_split(records, config["gap_seconds"])
    sessions = assign_sessions(records, config["gap_seconds"])
    results, skipped = [], []
    for run in config["runs"]:
        result = evaluate_run(run, records, locked, sessions, config, roots)
        if result is None:
            skipped.append(run["name"])
        else:
            results.append(result)
    base = runio.repo_root() / "results" / config["step"]
    written = [base / result["run"] for result in results if result["fresh"]]
    header = [
        "model",
        "session_run",
        "random_run",
        "session_ap50",
        "random_ap50",
        "difference",
        "low",
        "high",
        "inflation_claimed",
    ]
    for model in sorted({r["model"] for r in results if r["model"] and r["fresh"]}):
        group = [r for r in results if r["model"] == model]
        missing = [name for name in skipped if name.startswith(f"{model}_")]
        model_dir = runio.start_run(config["step"], f"summary_{model}", config)
        leak_rows = leakage(group)
        write_csv(model_dir / "leakage.csv", header, leak_rows)
        text = summary_text(group, missing, leak_rows)
        (model_dir / "summary.md").write_text(text, encoding="utf-8")
        written.append(model_dir)
    found = {result["run"]: result for result in results}
    for comparison in config.get("comparisons") or []:
        absent = [comparison[k] for k in ("baseline", "candidate") if comparison[k] not in found]
        if absent:
            print(f"compare_{comparison['name']}: {', '.join(absent)} not evaluated, skipped")
            continue
        result = compare(comparison, found, config)
        compare_dir = runio.start_run(
            config["step"], f"compare_{comparison['name']}", {**config, "comparison": comparison}
        )
        text = json.dumps(result, indent=2)
        (compare_dir / "comparison.json").write_text(text, encoding="utf-8")
        header = ["metric", "class", *COMPARISON_FIELDS]
        write_csv(compare_dir / "comparison.csv", header, comparison_rows(result))
        primary = result["mask_ap50"]["mean"]
        print(
            f"compare_{comparison['name']}: {comparison['candidate']} minus "
            f"{comparison['baseline']} mask AP50 {primary['difference']} "
            f"[{primary['low']}, {primary['high']}], {result['verdict']}, wrote {compare_dir}"
        )
        written.append(compare_dir)
    print(f"evaluated {len(results)} runs, skipped {len(skipped)}, wrote under {base}")
    if config.get("publish", True):
        for path in written:
            runio.publish(path)


if __name__ == "__main__":
    main()
