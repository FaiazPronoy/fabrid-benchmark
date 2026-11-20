from dataclasses import dataclass

import numpy as np

NEEDLE = 0


@dataclass
class Cluster:
    members: list[np.ndarray]
    merged: np.ndarray


def segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    span = ends - starts
    length = (span**2).sum(axis=1)
    offset = points[:, None, :] - starts[None, :, :]
    safe = np.where(length > 0, length, 1.0)
    t = np.clip((offset * span[None]).sum(axis=2) / safe, 0, 1)
    nearest = starts[None] + t[..., None] * span[None]
    return np.sqrt(((points[:, None, :] - nearest) ** 2).sum(axis=2))


def edges(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return polygon, np.roll(polygon, -1, axis=0)


def inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0:1], points[:, 1:2]
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    rise = np.where(y1 != y0, y1 - y0, 1.0)
    at = x0 + (y - y0) / rise * (x1 - x0)
    return ((crosses & (x < at)).sum(axis=1) % 2).astype(bool)


def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def edges_cross(first: np.ndarray, second: np.ndarray) -> bool:
    a0, a1 = (e[:, None, :] for e in edges(first))
    b0, b1 = (e[None, :, :] for e in edges(second))
    d1, d2 = cross(b0, b1, a0), cross(b0, b1, a1)
    d3, d4 = cross(a0, a1, b0), cross(a0, a1, b1)
    return bool((((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))).any())


def polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    if inside(first, second).any() or inside(second, first).any() or edges_cross(first, second):
        return 0.0
    one = segment_distances(first, *edges(second)).min()
    two = segment_distances(second, *edges(first)).min()
    return float(min(one, two))


def clusters(polygons: list[np.ndarray], threshold: float) -> list[list[int]]:
    parent = list(range(len(polygons)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if root(i) != root(j) and polygon_distance(polygons[i], polygons[j]) <= threshold:
                parent[max(root(i), root(j))] = min(root(i), root(j))
    groups: dict[int, list[int]] = {}
    for index in range(len(polygons)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return unique

    def chain(ordered) -> list:
        out = []
        for point in ordered:
            while len(out) >= 2 and cross(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
        return out

    lower, upper = chain(unique), chain(unique[::-1])
    return np.array(lower[:-1] + upper[:-1])


def rotated_rectangle(points: np.ndarray, min_width: float) -> np.ndarray:
    outline = hull(points)
    if len(outline) == 1:
        angles = np.array([0.0])
    else:
        span = np.roll(outline, -1, axis=0) - outline
        angles = np.arctan2(span[:, 1], span[:, 0])
    best = None
    for angle in angles:
        axis = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        u, v = outline @ axis, outline @ normal
        area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area < best[0] - 1e-12:
            best = (area, axis, normal, u, v)
    _, axis, normal, u, v = best
    centre_u, centre_v = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    half_u = max(u.max() - u.min(), min_width) / 2
    half_v = max(v.max() - v.min(), min_width) / 2
    corners = [(-half_u, -half_v), (half_u, -half_v), (half_u, half_v), (-half_u, half_v)]
    return np.array([(centre_u + a) * axis + (centre_v + b) * normal for a, b in corners])


def merge_needles(
    polygons: list[tuple[int, np.ndarray]], width: int, height: int, link: float, min_width: float
) -> tuple[list[tuple[int, np.ndarray]], list[Cluster]]:
    diagonal = float(np.hypot(width, height))
    scale = np.array([width, height], dtype=np.float64)
    positions = [i for i, (class_id, _) in enumerate(polygons) if class_id == NEEDLE]
    pixels = [polygons[i][1] * scale for i in positions]
    replaced: dict[int, np.ndarray | None] = {}
    found = []
    for group in clusters(pixels, link * diagonal):
        if len(group) < 2:
            continue
        corners = rotated_rectangle(np.vstack([pixels[k] for k in group]), min_width * diagonal)
        merged = np.clip(corners / scale, 0.0, 1.0)
        replaced[positions[group[0]]] = merged
        for k in group[1:]:
            replaced[positions[k]] = None
        found.append(Cluster([polygons[positions[k]][1] for k in group], merged))
    out = []
    for index, (class_id, points) in enumerate(polygons):
        if index not in replaced:
            out.append((class_id, points))
        elif replaced[index] is not None:
            out.append((class_id, replaced[index]))
    return out, found
