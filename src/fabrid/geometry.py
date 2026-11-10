import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fabrid import runio
from fabrid.labels import CLASSES, load_index, polygon_area
from fabrid.sessions import assign_sessions, check_disjoint, session_split
from fabrid.yolo import assert_unlocked, write_csv

MEASURES = (
    "area_px",
    "bbox_w",
    "bbox_h",
    "aspect",
    "rect_long",
    "median_width",
    "solidity",
    "extent",
)
PERCENTILES = (5, 25, 50, 75, 95)
SIZE_STRATA = ("under_32", "32_to_96", "over_96")
WIDTH_STRATA = ("up_to_2", "2_to_4", "4_to_8", "over_8")
WIDTH_LIMITS = (2, 4, 8)
COLORS = ((255, 64, 64), (64, 160, 255), (255, 210, 0), (64, 230, 120))


def rasterize(points: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    first = max(0, int(np.floor(y.min())))
    last = min(height, int(np.ceil(y.max())))
    if first >= last:
        return mask
    centres = np.arange(first, last)[:, None] + 0.5
    crossing = (centres >= np.minimum(y, y_next)) & (centres < np.maximum(y, y_next))
    rise = np.where(y_next != y, y_next - y, 1.0)
    at = x + (centres - y) / rise * (x_next - x)
    for offset in range(last - first):
        edges = np.sort(at[offset][crossing[offset]])
        starts = np.ceil(edges[0::2] - 0.5).astype(int).clip(0, width)
        stops = np.ceil(edges[1::2] - 0.5).astype(int).clip(0, width)
        for start, stop in zip(starts, stops, strict=True):
            mask[first + offset, start:stop] = True
    return mask


def pixel_corners(mask: np.ndarray) -> np.ndarray:
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    rows, cols = np.nonzero(mask & ~inner)
    corners = [np.stack([cols + dx, rows + dy], axis=1) for dx in (0, 1) for dy in (0, 1)]
    return np.unique(np.concatenate(corners), axis=0).astype(float)


def convex_hull(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    return points[ConvexHull(points).vertices]


def min_rect(hull: np.ndarray) -> tuple[float, float]:
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2).round(12))
    best = None
    for angle in angles:
        cos, sin = np.cos(angle), np.sin(angle)
        turned = hull @ np.array([[cos, -sin], [sin, cos]])
        sides = turned.max(axis=0) - turned.min(axis=0)
        if best is None or sides.prod() < best.prod():
            best = sides
    return float(best.max().round(6)), float(best.min().round(6))


def median_width(mask: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    padded = np.pad(mask, 1)
    depth = distance_transform_edt(padded) - 0.5
    skeleton = skeletonize(padded)
    values = depth[skeleton] if skeleton.any() else depth[padded]
    return float(2 * np.median(values))


def measure(mask: np.ndarray) -> dict | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    crop = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    height, width = crop.shape
    area = int(crop.sum())
    hull = convex_hull(pixel_corners(crop))
    long_side, short_side = min_rect(hull)
    return {
        "area_px": area,
        "bbox_w": width,
        "bbox_h": height,
        "aspect": long_side / short_side,
        "rect_long": long_side,
        "median_width": median_width(crop),
        "solidity": area / polygon_area(hull),
        "extent": area / (width * height),
    }


def size_stratum(bbox_area: float) -> str:
    if bbox_area < 32 * 32:
        return SIZE_STRATA[0]
    return SIZE_STRATA[1] if bbox_area <= 96 * 96 else SIZE_STRATA[2]


def width_stratum(width: float) -> str:
    for limit, label in zip(WIDTH_LIMITS, WIDTH_STRATA, strict=False):
        if width <= limit:
            return label
    return WIDTH_STRATA[-1]


def describe(class_id: int, points: np.ndarray, height: int, width: int) -> dict:
    found = measure(rasterize(points * [width, height], height, width))
    row = {"class_id": class_id, "class": CLASSES[class_id], "n_points": len(points)}
    if found is None:
        return {**row, **dict.fromkeys(MEASURES), "size_stratum": None, "width_stratum": None}
    return {
        **row,
        **found,
        "size_stratum": size_stratum(found["bbox_w"] * found["bbox_h"]),
        "width_stratum": width_stratum(found["median_width"]),
    }


def instance_rows(records, locked: dict[str, str]) -> list[dict]:
    rows = []
    for record in records:
        with Image.open(record.path) as image:
            width, height = image.size
        for index, (class_id, points) in enumerate(record.polygons):
            row = {"name": record.name, "split": locked[record.name], "index": index}
            rows.append({**row, **describe(class_id, points, height, width)})
    return rows


def groups(rows: list[dict]) -> dict[str, list[dict]]:
    measured = [row for row in rows if row["area_px"] is not None]
    found = {name: [row for row in measured if row["class"] == name] for name in CLASSES}
    return {**found, "all": measured}


def rounded(value) -> float | None:
    if value is None or isinstance(value, int):
        return value
    return round(float(value), 4)


def percentile_table(rows: list[dict]) -> list[list]:
    table = []
    for group, members in groups(rows).items():
        for name in MEASURES:
            values = np.array([row[name] for row in members], dtype=float)
            stats = np.percentile(values, PERCENTILES) if values.size else [None] * len(PERCENTILES)
            table.append([group, name, len(values), *[rounded(v) for v in stats]])
    return table


def strata_table(rows: list[dict], key: str, labels: tuple[str, ...]) -> list[list]:
    table = []
    for group, members in groups(rows).items():
        shares = [
            rounded(sum(row[key] == label for row in members) / len(members)) if members else None
            for label in labels
        ]
        table.append([group, len(members), *shares])
    return table


def summary_text(rows: list[dict], n_images: int) -> str:
    found = groups(rows)
    empty = sum(row["area_px"] is None for row in rows)
    lines = [
        "# Geometry",
        "",
        f"Images: {n_images}. Valid polygons: {len(rows)}. Rasterized to no pixel: {empty}.",
        "Endpoint: per class median of the median width in pixels.",
        "",
        "| class | count | median width px | width up to 2 | width up to 4 | median area px "
        "| median bbox w x h | bbox under 32 x 32 | median aspect |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group, members in found.items():
        if not members:
            lines.append(f"| {group} | 0 | | | | | | | |")
            continue
        median = {name: float(np.median([row[name] for row in members])) for name in MEASURES}
        widths = np.array([row["median_width"] for row in members])
        small = np.mean([row["size_stratum"] == SIZE_STRATA[0] for row in members])
        lines.append(
            f"| {group} | {len(members)} | {median['median_width']:.2f} "
            f"| {np.mean(widths <= 2):.3f} | {np.mean(widths <= 4):.3f} "
            f"| {median['area_px']:.0f} | {median['bbox_w']:.0f} x {median['bbox_h']:.0f} "
            f"| {small:.3f} | {median['aspect']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def sample_sheets(rows, locked, class_id: int, sheets: int, per_sheet: int, seed: int):
    pool = [
        row
        for row in rows
        if row["class_id"] == class_id
        and row["area_px"] is not None
        and locked[row["name"]] != "test"
    ]
    rng = np.random.default_rng([seed, class_id])
    picks = rng.choice(len(pool), size=min(len(pool), sheets * per_sheet), replace=False)
    chosen = [pool[int(i)] for i in picks]
    return [chosen[i : i + per_sheet] for i in range(0, len(chosen), per_sheet)]


def tile_image(record, points: np.ndarray, class_id: int, tile: int) -> Image.Image:
    with Image.open(record.path) as image:
        rgb = image.convert("RGB")
    pixels = points * rgb.size
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    side = max(16, round(float((high - low).max()) * 1.6 + 16))
    left, top = (np.round((low + high) / 2) - side // 2).astype(int)
    crop = rgb.crop((left, top, left + side, top + side)).resize((tile, tile))
    scale = tile / side
    outline = [((x - left) * scale, (y - top) * scale) for x, y in pixels]
    ImageDraw.Draw(crop).polygon(outline, outline=COLORS[class_id], width=2)
    return crop


def draw_sheet(items, by_name, locked, tile: int, columns: int, path: Path) -> None:
    n_rows = -(-len(items) // columns)
    canvas = Image.new("RGB", (columns * tile, n_rows * tile), (20, 20, 20))
    for slot, item in enumerate(items):
        assert_unlocked({item["name"]: locked[item["name"]]}, locked)
        record = by_name[item["name"]]
        class_id, points = record.polygons[item["index"]]
        position = ((slot % columns) * tile, (slot // columns) * tile)
        canvas.paste(tile_image(record, points, class_id, tile), position)
    canvas.save(path, quality=90)


def write_sheets(rows, records, locked, config: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {record.name: record for record in records}
    written = []
    for class_id, name in enumerate(CLASSES):
        sheets = sample_sheets(
            rows, locked, class_id, config["sheets_per_class"], config["per_sheet"], config["seed"]
        )
        for number, items in enumerate(sheets, start=1):
            path = out_dir / f"{name.replace(' ', '_')}_{number}.jpg"
            draw_sheet(items, by_name, locked, config["tile"], config["columns"], path)
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    records = load_index()
    locked = session_split(records, config["gap_seconds"])
    check_disjoint(locked, assign_sessions(records, config["gap_seconds"]))
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    run_dir = runio.start_run(config["step"], config["name"], config)
    rows = instance_rows(records, locked)
    columns = ["name", "split", "index", "class_id", "class", "n_points", *MEASURES]
    columns += ["size_stratum", "width_stratum"]
    with (run_dir / "instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {k: rounded(v) if k in MEASURES else v for k, v in r.items()} for r in rows
        )
    header = ["class", "measure", "count", *[f"p{p}" for p in PERCENTILES]]
    write_csv(run_dir / "per_class.csv", header, percentile_table(rows))
    size_table = strata_table(rows, "size_stratum", SIZE_STRATA)
    write_csv(run_dir / "size_strata.csv", ["class", "count", *SIZE_STRATA], size_table)
    width_table = strata_table(rows, "width_stratum", WIDTH_STRATA)
    write_csv(run_dir / "width_strata.csv", ["class", "count", *WIDTH_STRATA], width_table)
    (run_dir / "summary.md").write_text(summary_text(rows, len(records)), encoding="utf-8")
    print(f"measured {len(rows)} instances, wrote {run_dir}")
    sheets_dir = runio.runs_dir(config["step"], config["name"]) / "sheets"
    sheets = write_sheets(rows, records, locked, config, sheets_dir)
    print(f"wrote {len(sheets)} contact sheets to {sheets_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
