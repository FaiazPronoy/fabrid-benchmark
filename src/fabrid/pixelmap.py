import numpy as np

from fabrid.geometry import rasterize
from fabrid.labels import polygon_area

STRUCTURES = {
    4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    8: np.ones((3, 3), dtype=bool),
}


def instances_from_map(
    label_map: np.ndarray, background: int = 0, min_pixels: int = 1, connectivity: int = 8
) -> list[tuple[int, np.ndarray]]:
    from scipy.ndimage import label

    label_map = np.asarray(label_map)
    if label_map.ndim != 2:
        raise ValueError(f"label map must be two dimensional, got shape {label_map.shape}")
    if connectivity not in STRUCTURES:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    found = []
    for value in np.unique(label_map):
        if value == background:
            continue
        components, count = label(label_map == value, structure=STRUCTURES[connectivity])
        for index in range(1, count + 1):
            mask = components == index
            if mask.sum() >= min_pixels:
                first = int(np.flatnonzero(mask.ravel())[0])
                found.append((int(value), first, mask))
    found.sort(key=lambda item: item[:2])
    return [(value, mask) for value, _, mask in found]


def boundary_edges(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    inner = padded[1:-1, 1:-1]
    sides = (
        (padded[:-2, 1:-1], (0, 0), (1, 0)),
        (padded[1:-1, 2:], (1, 0), (1, 1)),
        (padded[2:, 1:-1], (1, 1), (0, 1)),
        (padded[1:-1, :-2], (0, 1), (0, 0)),
    )
    edges = {}
    for neighbour, start, end in sides:
        rows, cols = np.nonzero(inner & ~neighbour)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            begin = (col + start[0], row + start[1])
            edges.setdefault(begin, []).append((col + end[0], row + end[1]))
    return edges


def successor(edge, edges) -> tuple:
    begin, end = edge
    choices = edges[end]
    if len(choices) == 1:
        return end, choices[0]
    heading = (end[0] - begin[0], end[1] - begin[1])
    left = (end[0] + heading[1], end[1] - heading[0])
    return end, left if left in choices else choices[0]


def loops(mask: np.ndarray) -> list[np.ndarray]:
    edges = boundary_edges(mask)
    unused = {(begin, end) for begin, ends in edges.items() for end in ends}
    found = []
    while unused:
        start = min(unused)
        points, edge = [], start
        while True:
            unused.discard(edge)
            points.append(edge[0])
            edge = successor(edge, edges)
            if edge == start:
                break
        found.append(np.array(points, dtype=float))
    return found


def drop_collinear(points: np.ndarray) -> np.ndarray:
    before = np.roll(points, 1, axis=0)
    after = np.roll(points, -1, axis=0)
    cross = (points - before)[:, 0] * (after - points)[:, 1]
    cross -= (points - before)[:, 1] * (after - points)[:, 0]
    return points[cross != 0]


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    import cv2

    curve = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(curve, tolerance, True).reshape(-1, 2).astype(float)


def polygons_from_mask(mask: np.ndarray, tolerance: float = 0.0) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    polygons = []
    for points in loops(mask):
        if signed_area(points) <= 0:
            continue
        points = drop_collinear(points)
        if tolerance > 0:
            points = simplify(points, tolerance)
        if len(points) >= 3 and polygon_area(points) > 0:
            polygons.append(points / [width, height])
    return polygons


def label_line(class_id: int, polygon: np.ndarray) -> str:
    return " ".join([str(class_id), *(f"{v:.6f}" for v in polygon.ravel())])


def round_trip_iou(mask: np.ndarray, polygons: list[np.ndarray]) -> float:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    drawn = np.zeros_like(mask)
    for polygon in polygons:
        drawn |= rasterize(polygon * [width, height], height, width)
    union = (mask | drawn).sum()
    return float((mask & drawn).sum() / union) if union else 1.0
