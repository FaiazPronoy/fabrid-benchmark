import numpy as np
import pytest

from fabrid import metrics


def line(row, size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[row : row + 2, 6:58] = True
    return mask


def box(top, left, height, width, size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def test_shifted_thin_line_is_forgiven_by_tolerant_metrics():
    gt, pred = line(30), line(31)
    plain = metrics.mask_iou(pred, gt)[0, 0]
    assert plain == pytest.approx(1 / 3)
    assert metrics.tolerant_iou(pred, gt, 2) > 0.95
    assert metrics.tolerant_iou(pred, gt, 0) == pytest.approx(plain)
    assert metrics.cldice(pred, gt, 1) > 0.9
    assert metrics.boundary_f1(pred, gt, 1) > 0.9


def test_plain_cldice_on_shifted_two_pixel_line():
    gt, pred = line(30), line(31)
    assert metrics.cldice(pred, gt) == 0
    assert metrics.cldice(gt, gt) == 1
    wide_gt, wide_pred = box(28, 6, 5, 52), box(29, 6, 5, 52)
    assert metrics.cldice(wide_pred, wide_gt) == 1


def test_far_line_scores_zero():
    gt, pred = line(10), line(40)
    assert metrics.tolerant_iou(pred, gt, 2) == 0
    assert metrics.cldice(pred, gt) == 0
    assert metrics.boundary_f1(pred, gt, 2) == 0


@pytest.mark.parametrize(
    "function",
    [
        lambda p, g: metrics.tolerant_iou(p, g, 2),
        metrics.cldice,
        lambda p, g: metrics.boundary_f1(p, g, 2),
    ],
)
def test_empty_masks(function):
    empty, full = np.zeros((16, 16), dtype=bool), box(4, 4, 6, 6, 16)
    assert function(empty, empty) == 1
    assert function(full, empty) == 0
    assert function(empty, full) == 0
    assert function(full, full) == 1


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        metrics.tolerant_iou(np.zeros((4, 4)), np.zeros((4, 5)), 1)


def test_match_instances_greedy_by_score():
    gt = [box(0, 0, 10, 10), box(30, 30, 10, 10)]
    pred = [box(0, 0, 10, 8), box(0, 0, 10, 10), box(50, 50, 5, 5)]
    found = metrics.match_instances(pred, gt, 0.5, scores=[0.9, 0.5, 0.8])
    assert found.tp == [(0, 0)]
    assert found.fp == [1, 2]
    assert found.fn == [1]
    by_order = metrics.match_instances(pred, gt, 0.5, scores=[0.1, 0.9, 0.8])
    assert by_order.tp == [(1, 0)]
    assert by_order.fp == [0, 2]


def test_match_instances_threshold_and_empty():
    gt = [box(0, 0, 10, 10)]
    half = [box(0, 0, 10, 5)]
    assert metrics.match_instances(half, gt, 0.5).tp == [(0, 0)]
    assert metrics.match_instances(half, gt, 0.6).fp == [0]
    none = metrics.match_instances([], gt, 0.5)
    assert none == ([], [], [0])
    only_pred = metrics.match_instances(half, [], 0.5)
    assert only_pred == ([], [0], [])
    with pytest.raises(ValueError):
        metrics.match_instances(half, gt, 0.5, scores=[0.1, 0.2])


def test_size_bucket_limits():
    assert metrics.size_bucket(0) == "small"
    assert metrics.size_bucket(32**2 - 1) == "small"
    assert metrics.size_bucket(32**2) == "medium"
    assert metrics.size_bucket(96**2 - 1) == "medium"
    assert metrics.size_bucket(96**2) == "large"
    with pytest.raises(ValueError):
        metrics.size_bucket(-1)


def test_group_bootstrap_constant_and_repeatable():
    values = np.full(50, 0.4)
    estimate = metrics.group_bootstrap(values, np.arange(50) % 7)
    assert estimate == pytest.approx((0.4, 0.4, 0.4))
    values = np.random.default_rng(1).random(80)
    groups = np.arange(80) // 4
    first = metrics.group_bootstrap(values, groups, n=500, seed=3)
    assert first == metrics.group_bootstrap(values, groups, n=500, seed=3)
    assert first.low <= first.value <= first.high
    assert first.value == pytest.approx(values.mean())


def test_group_bootstrap_widens_with_few_groups():
    values = np.sort(np.random.default_rng(0).normal(size=300))
    own = metrics.group_bootstrap(values, np.arange(300))
    few = metrics.group_bootstrap(values, np.arange(300) // 60)
    assert few.high - few.low > 2 * (own.high - own.low)
    assert own.value == pytest.approx(few.value)


def test_group_bootstrap_guards():
    with pytest.raises(ValueError):
        metrics.group_bootstrap([1.0, np.nan], [0, 1])
    with pytest.raises(ValueError):
        metrics.group_bootstrap([1.0, 2.0], [0])
    with pytest.raises(ValueError):
        metrics.group_bootstrap([], [])


def test_paired_group_bootstrap():
    rng = np.random.default_rng(2)
    b = rng.random(120)
    a = b + 0.1
    groups = np.arange(120) // 10
    difference = metrics.paired_group_bootstrap(a, b, groups)
    assert difference == pytest.approx((0.1, 0.1, 0.1))
    noisy = metrics.paired_group_bootstrap(a + rng.normal(0, 0.05, 120), b, groups)
    assert noisy.low < 0.1 < noisy.high
    with pytest.raises(ValueError):
        metrics.paired_group_bootstrap(a[:-1], b, groups)
