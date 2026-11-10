import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fabrid import runio
from fabrid.labels import CLASSES, is_valid, label_lines, load_index, parse_line
from fabrid.sessions import assign_sessions

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
DCT_SIZE = 32


def dct_matrix(size: int) -> np.ndarray:
    k = np.arange(size)[:, None]
    n = np.arange(size)[None, :]
    return np.cos(np.pi * (2 * n + 1) * k / (2 * size))


DCT = dct_matrix(DCT_SIZE)


def pack(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits.ravel()), 2))


def dhash(image: Image.Image) -> np.uint64:
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
    return pack(small[:, 1:] > small[:, :-1])


def phash(image: Image.Image) -> np.uint64:
    gray = image.convert("L").resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=float)
    low = (DCT @ pixels @ DCT.T)[:8, :8]
    return pack(low > np.median(low))


def popcount(values: np.ndarray) -> np.ndarray:
    as_bytes = np.ascontiguousarray(values, dtype=np.uint64).view(np.uint8)
    return POPCOUNT[as_bytes].reshape(-1, 8).sum(axis=1)


def near_pairs(first: np.ndarray, second: np.ndarray, threshold: int) -> list[tuple]:
    pairs = []
    for i in range(len(first) - 1):
        d1 = popcount(first[i + 1 :] ^ first[i])
        d2 = popcount(second[i + 1 :] ^ second[i])
        for offset in np.flatnonzero((d1 <= threshold) | (d2 <= threshold)):
            pairs.append((i, i + 1 + int(offset), int(d1[offset]), int(d2[offset])))
    return pairs


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
        return {
            "width": rgb.width,
            "height": rgb.height,
            "md5": file_md5(path),
            "pixels": hashlib.md5(pixels.tobytes() + str(pixels.shape).encode()).hexdigest(),
            "phash": phash(rgb),
            "dhash": dhash(rgb),
        }


def inspect_labels(record) -> dict:
    lines = label_lines(record.label_path)
    unparsed, invalid, outside = 0, 0, 0
    per_class = Counter()
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            unparsed += 1
            continue
        class_id, points = parsed
        per_class[class_id] += 1
        if not is_valid(points):
            invalid += 1
        if (points < 0).any() or (points > 1).any():
            outside += 1
    return {"unparsed": unparsed, "invalid": invalid, "outside": outside, "lines": per_class}


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def audit(records, threshold: int, gap_seconds: float, out_dir: Path) -> dict:
    sessions = assign_sessions(records, gap_seconds)
    images = [inspect_image(r.path) for r in records]
    labels = [inspect_labels(r) for r in records]
    present = [sorted({c for c, _ in r.polygons}) for r in records]

    write_csv(
        out_dir / "manifest.csv",
        [
            "name",
            "export_folder",
            "stem",
            "timestamp",
            "day",
            "width",
            "height",
            "md5",
            "n_lines",
            "n_valid",
            "classes_present",
        ],
        [
            [
                r.name,
                r.folder,
                r.stem,
                r.timestamp.isoformat(),
                r.day.isoformat(),
                info["width"],
                info["height"],
                info["md5"],
                r.n_lines,
                r.n_valid,
                ";".join(str(c) for c in classes),
            ]
            for r, info, classes in zip(records, images, present, strict=True)
        ],
    )

    groups = defaultdict(list)
    for index, info in enumerate(images):
        groups[info["pixels"]].append(index)
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    exact_rows = []
    for number, members in enumerate(sorted(duplicate_groups, key=lambda m: records[m[0]].name)):
        for index in members:
            r = records[index]
            exact_rows.append(
                [number, r.name, r.folder, r.timestamp.isoformat(), images[index]["md5"]]
            )
    write_csv(
        out_dir / "duplicates_exact.csv",
        ["group", "name", "export_folder", "timestamp", "md5"],
        exact_rows,
    )

    hashes_p = np.array([info["phash"] for info in images], dtype=np.uint64)
    hashes_d = np.array([info["dhash"] for info in images], dtype=np.uint64)
    pairs = near_pairs(hashes_p, hashes_d, threshold)
    near_rows = []
    for i, j, dp, dd in pairs:
        a, b = records[i], records[j]
        near_rows.append(
            [
                a.name,
                a.folder,
                a.timestamp.isoformat(),
                b.name,
                b.folder,
                b.timestamp.isoformat(),
                abs(int((b.timestamp - a.timestamp).total_seconds())),
                sessions[a.name] == sessions[b.name],
                dp,
                dd,
                images[i]["pixels"] == images[j]["pixels"],
            ]
        )
    write_csv(
        out_dir / "duplicates_near.csv",
        [
            "name_a",
            "export_folder_a",
            "timestamp_a",
            "name_b",
            "export_folder_b",
            "timestamp_b",
            "seconds_apart",
            "same_session",
            "phash_distance",
            "dhash_distance",
            "pixels_identical",
        ],
        near_rows,
    )

    by_stem = defaultdict(list)
    for index, r in enumerate(records):
        by_stem[r.stem].append(index)
    stem_rows = []
    for stem, members in sorted(by_stem.items()):
        if len(members) < 2:
            continue
        versions = {images[i]["pixels"] for i in members}
        stem_rows.append(
            [
                stem,
                len(members),
                ";".join(records[i].name for i in members),
                ";".join(sorted({records[i].folder for i in members})),
                len(versions) > 1,
                len(versions),
            ]
        )
    write_csv(
        out_dir / "stems_repeated.csv",
        ["stem", "files", "names", "export_folders", "pixels_differ", "distinct_pixel_versions"],
        stem_rows,
    )

    per_class = {name: {"lines": 0, "valid_polygons": 0, "images": 0} for name in CLASSES}
    for r, info, classes in zip(records, labels, present, strict=True):
        for class_id, count in info["lines"].items():
            per_class[CLASSES[class_id]]["lines"] += count
        for class_id, _ in r.polygons:
            per_class[CLASSES[class_id]]["valid_polygons"] += 1
        for class_id in classes:
            per_class[CLASSES[class_id]]["images"] += 1
    counts = {
        "images": len(records),
        "images_per_export_folder": dict(sorted(Counter(r.folder for r in records).items())),
        "label_files_missing": sum(r.label_path is None for r in records),
        "empty_images": sum(r.n_lines == 0 for r in records),
        "images_without_valid_polygon": sum(r.n_valid == 0 for r in records),
        "lines": sum(r.n_lines for r in records),
        "unparsed_lines": sum(info["unparsed"] for info in labels),
        "parsed_invalid_polygons": sum(info["invalid"] for info in labels),
        "polygons_outside_unit_square": sum(info["outside"] for info in labels),
        "valid_polygons": sum(r.n_valid for r in records),
        "per_class": per_class,
    }
    (out_dir / "label_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    days = sorted({r.day for r in records})
    day_rows = []
    for day in days:
        members = [r for r in records if r.day == day]
        instances = Counter(c for r in members for c, _ in r.polygons)
        day_rows.append(
            [
                day.isoformat(),
                len({sessions[r.name] for r in members}),
                len(members),
                sum(r.n_lines == 0 for r in members),
                *[instances[c] for c in range(len(CLASSES))],
            ]
        )
    class_columns = [f"{name.replace(' ', '_')}_instances" for name in CLASSES]
    write_csv(
        out_dir / "day_class_table.csv",
        ["day", "sessions", "images", "empty_images", *class_columns],
        day_rows,
    )

    exact_files = sum(len(m) for m in duplicate_groups)
    checks = {
        "no_exact_duplicates": not duplicate_groups,
        "all_lines_parse": counts["unparsed_lines"] == 0,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "exact_groups": len(duplicate_groups),
        "exact_files": exact_files,
        "near_pairs": len(pairs),
        "near_pairs_same_session": sum(row[7] for row in near_rows),
        "repeated_stems": len(stem_rows),
        "repeated_stem_files": sum(row[1] for row in stem_rows),
        "repeated_stems_pixels_differ": sum(row[4] for row in stem_rows),
        "sessions": len(set(sessions.values())),
        "days": len(days),
        "sizes": dict(Counter(f"{i['width']}x{i['height']}" for i in images)),
        "counts": counts,
    }
    first, last = min(r.timestamp for r in records), max(r.timestamp for r in records)
    result["time_range"] = [first.isoformat(), last.isoformat()]
    (out_dir / "summary.md").write_text(render_summary(result, threshold), encoding="utf-8")
    return result


def render_summary(result: dict, threshold: int) -> str:
    counts = result["counts"]
    checks = result["checks"]
    verdict = "PASS" if result["passed"] else "FAIL"
    sizes = ", ".join(f"{size} ({n})" for size, n in sorted(result["sizes"].items()))
    folders = ", ".join(f"{k} {v}" for k, v in counts["images_per_export_folder"].items())
    lines = [
        "# Integrity audit",
        "",
        f"Verdict: {verdict}",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| No exact duplicate images | {'pass' if checks['no_exact_duplicates'] else 'fail'} "
        f"({result['exact_groups']} groups, {result['exact_files']} files) |",
        f"| Every label line parses | {'pass' if checks['all_lines_parse'] else 'fail'} "
        f"({counts['unparsed_lines']} unparsed of {counts['lines']}) |",
        "",
        "## Files",
        "",
        f"- Images: {counts['images']} ({folders})",
        f"- Image sizes: {sizes}",
        f"- Capture time range: {result['time_range'][0]} to {result['time_range'][1]}",
        f"- Capture days: {result['days']}, sessions: {result['sessions']}",
        f"- Missing label files: {counts['label_files_missing']}",
        "",
        "## Duplicates",
        "",
        f"- Near duplicate pairs (pHash or dHash distance up to {threshold}): "
        f"{result['near_pairs']}, of which {result['near_pairs_same_session']} "
        "fall in the same session",
        f"- Stems shared by more than one file: {result['repeated_stems']} stems, "
        f"{result['repeated_stem_files']} files, "
        f"{result['repeated_stems_pixels_differ']} stems with differing pixels",
        "",
        "## Labels",
        "",
        f"- Label lines: {counts['lines']}, valid polygons: {counts['valid_polygons']}",
        f"- Parsed but invalid polygons (under 3 points or zero area): "
        f"{counts['parsed_invalid_polygons']}",
        f"- Polygons with a point outside the unit square: "
        f"{counts['polygons_outside_unit_square']}",
        f"- Empty label files: {counts['empty_images']}, images without a valid polygon: "
        f"{counts['images_without_valid_polygon']}",
        "",
        "| Class | Lines | Valid polygons | Images |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in counts["per_class"].items():
        lines.append(f"| {name} | {row['lines']} | {row['valid_polygons']} | {row['images']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    out_dir = runio.start_run(config["step"], config["name"], config)
    result = audit(records, config["near_threshold"], config["gap_seconds"], out_dir)
    print(f"verdict: {'PASS' if result['passed'] else 'FAIL'} {result['checks']}")
    print(f"exact duplicate groups {result['exact_groups']}, near pairs {result['near_pairs']}")
    print(f"wrote {out_dir}")
    if config.get("publish", True):
        runio.publish(out_dir)


if __name__ == "__main__":
    main()
