from typing import NamedTuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

SMALL = 32**2
LARGE = 96**2


class Estimate(NamedTuple):
    value: float
    low: float
    high: float


class Matches(NamedTuple):
    tp: list[tuple[int, int]]
    fp: list[int]
    fn: list[int]


def as_stack(masks) -> np.ndarray:
    stack = np.asarray(masks, dtype=bool)
    if stack.size == 0 and stack.ndim < 3:
        return np.zeros((0, 0, 0), dtype=bool)
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3:
        raise ValueError(f"expected masks of shape (n, h, w), got {stack.shape}")
    return stack


def mask_iou(pred, gt) -> np.ndarray:
    pred, gt = as_stack(pred), as_stack(gt)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError(f"mask sizes differ: {pred.shape[1:]} and {gt.shape[1:]}")
    a = pred.reshape(len(pred), -1).astype(np.float64)
    b = gt.reshape(len(gt), -1).astype(np.float64)
    inter = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_instances(pred, gt, iou_threshold: float = 0.5, scores=None) -> Matches:
    iou = mask_iou(pred, gt)
    n_pred, n_gt = iou.shape
    scores = np.zeros(n_pred) if scores is None else np.asarray(scores, dtype=np.float64)
    if scores.shape != (n_pred,):
        raise ValueError("one score per predicted mask is required")
    order = sorted(range(n_pred), key=lambda i: (-scores[i], i))
    taken = np.zeros(n_gt, dtype=bool)
    tp, fp = [], []
    for i in order:
        if n_gt == 0:
            fp.append(i)
            continue
        candidates = np.where(taken, -1.0, iou[i])
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_threshold and candidates[j] > 0:
            taken[j] = True
            tp.append((i, j))
        else:
            fp.append(i)
    fn = [j for j in range(n_gt) if not taken[j]]
    return Matches(tp, sorted(fp), fn)


def check_pair(pred_mask, gt_mask) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = np.asarray(pred_mask, dtype=bool), np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"expected two 2d masks of one size, got {pred.shape} and {gt.shape}")
    return pred, gt


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) <= radius


def tolerant_iou(pred_mask, gt_mask, tol_px: float) -> float:
    pred, gt = check_pair(pred_mask, gt_mask)
    if not pred.any() and not gt.any():
        return 1.0
    pred_hit = (pred & dilate(gt, tol_px)).sum()
    gt_hit = (gt & dilate(pred, tol_px)).sum()
    inter = (pred_hit + gt_hit) / 2
    union = inter + (pred.sum() - pred_hit) + (gt.sum() - gt_hit)
    return float(inter / union)


def cldice(pred_mask, gt_mask, tol_px: float = 0) -> float:
    pred, gt = check_pair(pred_mask, gt_mask)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    pred_skeleton, gt_skeleton = skeletonize(pred), skeletonize(gt)
    precision = (pred_skeleton & dilate(gt, tol_px)).sum() / max(pred_skeleton.sum(), 1)
    sensitivity = (gt_skeleton & dilate(pred, tol_px)).sum() / max(gt_skeleton.sum(), 1)
    if precision + sensitivity == 0:
        return 0.0
    return float(2 * precision * sensitivity / (precision + sensitivity))


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def boundary_f1(pred_mask, gt_mask, tol_px: float) -> float:
    pred, gt = check_pair(pred_mask, gt_mask)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    pred_edge, gt_edge = boundary(pred), boundary(gt)
    precision = (pred_edge & dilate(gt_edge, tol_px)).sum() / pred_edge.sum()
    recall = (gt_edge & dilate(pred_edge, tol_px)).sum() / gt_edge.sum()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def size_bucket(bbox_area: float) -> str:
    if bbox_area < 0:
        raise ValueError(f"negative area: {bbox_area}")
    if bbox_area < SMALL:
        return "small"
    if bbox_area < LARGE:
        return "medium"
    return "large"


def group_sums(values, groups) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or len(values) != len(groups):
        raise ValueError("values and groups must be one dimensional and of equal length")
    if len(values) == 0:
        raise ValueError("no values to resample")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    _, labels = np.unique(groups, return_inverse=True)
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels).astype(np.float64)
    return sums, counts


def group_bootstrap(values, groups, n: int = 2000, seed: int = 0, level: float = 0.95) -> Estimate:
    sums, counts = group_sums(values, groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return Estimate(float(sums.sum() / counts.sum()), float(low), float(high))


def paired_group_bootstrap(
    a, b, groups, n: int = 2000, seed: int = 0, level: float = 0.95
) -> Estimate:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired values differ in shape: {a.shape} and {b.shape}")
    return group_bootstrap(a - b, groups, n, seed, level)
