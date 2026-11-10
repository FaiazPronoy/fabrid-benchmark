from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fabrid import semantic


def square(x0, y0, x1, y1, width, height):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) / [width, height]


def test_target_later_line_wins_and_overlap_counted():
    first = (0, square(0, 0, 6, 6, 10, 10))
    second = (3, square(4, 4, 10, 10, 10, 10))
    same = (3, square(8, 8, 10, 10, 10, 10))
    target, contested = semantic.build_target([first, second, same], 10, 10)
    assert target[0, 0] == 1
    assert target[5, 5] == 4
    assert target[9, 9] == 4
    assert contested == 4
    reverse, contested_reverse = semantic.build_target([second, first], 10, 10)
    assert reverse[5, 5] == 1
    assert contested_reverse == 4
    assert (target == 1).sum() == 36 - 4


def test_target_empty_image():
    target, contested = semantic.build_target([], 12, 8)
    assert target.shape == (12, 8)
    assert target.dtype == np.uint8
    assert not target.any()
    assert contested == 0


def test_scores_on_hand_built_case():
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0, :2] = 1
    target[3, 3] = semantic.IGNORE
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, 0] = 1
    pred[1, 0] = 1
    pred[3, 3] = 2
    counts = semantic.confusion(pred, target)
    assert counts[1].tolist() == [1, 1, 1]
    assert counts[2].tolist() == [0, 0, 0]
    per_class = semantic.class_scores(counts)
    needle = per_class["needle mark"]
    assert needle["dice"] == pytest.approx(0.5)
    assert needle["iou"] == pytest.approx(1 / 3)
    assert needle["precision"] == pytest.approx(0.5)
    assert needle["recall"] == pytest.approx(0.5)
    assert per_class["hole"]["dice"] is None
    assert semantic.mean_dice(per_class) == pytest.approx(0.5)
    counts[2] = [0, 3, 0]
    assert semantic.mean_dice(semantic.class_scores(counts)) == pytest.approx(0.25)


def test_pooled_dice_is_not_mean_of_images():
    target_a, pred_a = np.ones((2, 2), np.uint8), np.ones((2, 2), np.uint8)
    target_b, pred_b = np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)
    pred_b[0, 0] = 1
    counts = semantic.confusion(pred_a, target_a) + semantic.confusion(pred_b, target_b)
    assert semantic.class_scores(counts)["needle mark"]["dice"] == pytest.approx(8 / 9)


def test_alarm_threshold_ignores_padding():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[:, 8:] = semantic.IGNORE
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[:, 6:] = 3
    assert semantic.is_alarm(pred, target, 20)
    assert not semantic.is_alarm(pred, target, 21)


def test_padding_keeps_labels_aligned(tmp_path):
    height, width = 40, 56
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[5:15, 30:50] = 255
    path = tmp_path / "IMG_20260501_090000.jpg"
    Image.fromarray(pixels).save(path, format="PNG")
    record = SimpleNamespace(path=path, polygons=[(2, square(30, 5, 50, 15, width, height))])
    for flips in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        image, target = semantic.Samples([record], 64, flips).load(0)
        assert image.shape == (64, 64, 3) and target.shape == (64, 64)
        assert (target[height:] == semantic.IGNORE).all()
        assert (target[:, width:] == semantic.IGNORE).all()
        bright = image[:height, :width, 0] > 0
        assert ((target[:height, :width] == 3) == bright).all()
        assert (image[height:] == 0).all()


def test_padding_rejects_large_image():
    with pytest.raises(ValueError):
        semantic.pad(np.zeros((70, 10, 3)), np.zeros((70, 10), np.uint8), 64)


def test_dice_loss_ignores_padding():
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 0, :2] = 1
    perfect = torch.full((1, semantic.N_CLASSES, 4, 4), -20.0)
    perfect[:, 0] = 20
    perfect[0, 1, 0, :2], perfect[0, 0, 0, :2] = 20, -20
    assert float(semantic.dice_loss(perfect, target)) < 0.01
    padded = target.clone()
    padded[0, 3] = semantic.IGNORE
    noisy = perfect.clone()
    noisy[0, 1, 3], noisy[0, 0, 3] = 20, -20
    assert float(semantic.dice_loss(noisy, padded)) < 0.01
    assert float(semantic.dice_loss(noisy, target)) > 0.1
