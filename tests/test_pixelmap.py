import numpy as np
import pytest

from fabrid import pixelmap
from fabrid.labels import parse_line


def test_touching_components_of_different_classes_stay_separate():
    label_map = np.zeros((6, 8), dtype=np.uint8)
    label_map[1:5, 1:4] = 1
    label_map[1:5, 4:7] = 2
    found = pixelmap.instances_from_map(label_map, connectivity=8)
    assert [value for value, _ in found] == [1, 2]
    assert [int(mask.sum()) for _, mask in found] == [12, 12]
    assert not (found[0][1] & found[1][1]).any()


def test_diagonal_neighbours_follow_connectivity():
    label_map = np.zeros((5, 5), dtype=np.int64)
    label_map[1, 1] = 3
    label_map[2, 2] = 3
    assert len(pixelmap.instances_from_map(label_map, connectivity=4)) == 2
    joined = pixelmap.instances_from_map(label_map, connectivity=8)
    assert len(joined) == 1
    assert joined[0][1].sum() == 2
    with pytest.raises(ValueError):
        pixelmap.instances_from_map(label_map, connectivity=6)


def test_small_components_are_dropped_and_order_is_fixed():
    label_map = np.zeros((10, 10), dtype=np.uint8)
    label_map[0:3, 6:9] = 2
    label_map[5, 5] = 1
    label_map[6:9, 0:3] = 1
    label_map[0:2, 0:2] = 255
    found = pixelmap.instances_from_map(label_map, background=255, min_pixels=2)
    assert [(value, int(mask.sum())) for value, mask in found] == [(0, 77), (1, 9), (2, 9)]
    found = pixelmap.instances_from_map(label_map, min_pixels=2)
    assert [(value, int(mask.sum())) for value, mask in found] == [(1, 9), (2, 9), (255, 4)]
    with_single = pixelmap.instances_from_map(label_map, min_pixels=1, connectivity=4)
    assert [(v, int(m.sum())) for v, m in with_single][:2] == [(1, 1), (1, 9)]


@pytest.mark.parametrize(
    "pixels",
    [
        [(4, c) for c in range(2, 15)],
        [(r, 7) for r in range(1, 12)],
        [(r, r) for r in range(2, 13)],
        [(3, 3)],
        [(2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (5, 6)],
    ],
)
def test_thin_shapes_survive_the_round_trip(pixels):
    mask = np.zeros((16, 18), dtype=bool)
    for row, col in pixels:
        mask[row, col] = True
    polygons = pixelmap.polygons_from_mask(mask)
    assert len(polygons) == 1
    assert pixelmap.round_trip_iou(mask, polygons) == 1.0


def test_holes_are_filled_and_reported_by_iou():
    from scipy.ndimage import binary_fill_holes

    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    mask[4:7, 4:7] = False
    polygons = pixelmap.polygons_from_mask(mask)
    assert len(polygons) == 1
    filled = binary_fill_holes(mask)
    assert pixelmap.round_trip_iou(mask, polygons) == pytest.approx(mask.sum() / filled.sum())


@pytest.mark.parametrize("seed", range(5))
def test_random_masks_round_trip_up_to_holes(seed):
    from scipy.ndimage import binary_fill_holes

    mask = np.random.default_rng(seed).random((24, 20)) < 0.45
    polygons = pixelmap.polygons_from_mask(mask)
    filled = binary_fill_holes(mask)
    assert pixelmap.round_trip_iou(mask, polygons) == pytest.approx(mask.sum() / filled.sum())


def test_polygons_are_normalised_label_lines():
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:10, 10:30] = True
    (polygon,) = pixelmap.polygons_from_mask(mask)
    assert polygon.min() >= 0
    assert polygon.max() <= 1
    assert len(polygon) == 4
    class_id, points = parse_line(pixelmap.label_line(2, polygon))
    assert class_id == 2
    np.testing.assert_allclose(points * [40, 20], [[10, 5], [30, 5], [30, 10], [10, 10]])


def test_tolerance_simplifies_with_small_loss():
    pytest.importorskip("cv2")
    rows, cols = np.mgrid[0:60, 0:60]
    mask = (rows - 30) ** 2 + (cols - 30) ** 2 < 20**2
    exact = pixelmap.polygons_from_mask(mask)
    rough = pixelmap.polygons_from_mask(mask, tolerance=1.0)
    assert len(rough[0]) < len(exact[0])
    assert pixelmap.round_trip_iou(mask, rough) > 0.95


def test_empty_mask_gives_no_polygon():
    assert pixelmap.polygons_from_mask(np.zeros((4, 4), dtype=bool)) == []
    assert pixelmap.instances_from_map(np.zeros((4, 4), dtype=np.uint8)) == []
