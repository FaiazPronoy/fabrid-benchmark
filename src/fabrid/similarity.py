import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fabrid import runio
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import assign_sessions, check_disjoint, random_split, session_split

ARMS = ("export", "random", "session")
PARTS = ("val", "test")
EXPORT_PARTS = {"train": "train", "valid": "val", "test": "test"}
INPUT_ROOT = Path("/kaggle/input")
CACHE_NAME = "embeddings.npz"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def assert_aggregate(names, locked: dict[str, str]) -> None:
    leaked = sorted(name for name in names if locked.get(name) == "test")
    if leaked:
        raise AssertionError(f"locked test images reached an itemized output: {leaked[:5]}")


def export_split(records) -> dict[str, str]:
    return {r.name: EXPORT_PARTS[r.folder] for r in records}


def fill_defects(image: Image.Image, polygons) -> Image.Image:
    rgb = image.convert("RGB")
    mean = np.asarray(rgb, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    color = tuple(int(round(v)) for v in mean)
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size
    for _, points in polygons:
        draw.polygon([(float(x * width), float(y * height)) for x, y in points], fill=color)
    return rgb


def prepare(record, size: int) -> np.ndarray:
    with Image.open(record.path) as image:
        filled = fill_defects(image, record.polygons)
    resized = filled.resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / 255
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero embedding vector")
    return (vectors / norms).astype(np.float32)


def cache_key(config: dict) -> str:
    return json.dumps(
        {"model": config["model"], "size": config["image_size"], "fill": "image mean"},
        sort_keys=True,
    )


def read_caches(paths, key: str) -> dict[str, np.ndarray]:
    found = {}
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                print(f"ignored cache with other settings: {path}")
                continue
            for name, vector in zip(data["names"], data["vectors"], strict=True):
                found.setdefault(str(name), vector)
        print(f"read {len(found)} cached embeddings so far, from {path}")
    return found


def write_cache(path: Path, cache: dict[str, np.ndarray], key: str) -> None:
    names = sorted(cache)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(
        partial,
        names=np.array(names),
        vectors=np.stack([cache[n] for n in names]),
        key=np.array(key),
    )
    partial.replace(path)


def cache_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / CACHE_NAME]
    if INPUT_ROOT.is_dir():
        paths += sorted(INPUT_ROOT.glob(f"**/audit/similarity/{CACHE_NAME}"))
    return paths


def embed(records, config: dict, run_dir: Path, start: float) -> np.ndarray | None:
    key = cache_key(config)
    cache = read_caches(cache_paths(run_dir), key)
    todo = [r for r in records if r.name not in cache]
    if todo:
        import timm
        import torch

        model = timm.create_model(
            config["model"], pretrained=True, num_classes=0, img_size=config["image_size"]
        ).eval()
        limit = config["hours_limit"] * 3600
        size = config["batch_size"]
        print(f"embedding {len(todo)} images with {config['model']} at {config['image_size']}")
        for offset in range(0, len(todo), size):
            batch = todo[offset : offset + size]
            pixels = torch.from_numpy(np.stack([prepare(r, config["image_size"]) for r in batch]))
            with torch.inference_mode():
                output = model(pixels).numpy()
            for record, vector in zip(batch, normalize(output), strict=True):
                cache[record.name] = vector
            done = offset + len(batch)
            if done % config["save_every"] < size or done == len(todo):
                write_cache(run_dir / CACHE_NAME, cache, key)
                print(f"embedded {done} of {len(todo)}")
            if time.time() - start >= limit:
                write_cache(run_dir / CACHE_NAME, cache, key)
                print(f"wall clock limit reached after {done} images, cache saved, rerun")
                return None
    write_cache(run_dir / CACHE_NAME, cache, key)
    return normalize(np.stack([cache[r.name] for r in records]))


def consecutive_pairs(timestamps, max_seconds: float) -> list[tuple[int, int]]:
    order = sorted(range(len(timestamps)), key=lambda i: (timestamps[i], i))
    pairs = []
    for a, b in zip(order, order[1:], strict=False):
        same_day = timestamps[a].date() == timestamps[b].date()
        if same_day and (timestamps[b] - timestamps[a]).total_seconds() <= max_seconds:
            pairs.append((a, b))
    return pairs


def consecutive_threshold(
    vectors: np.ndarray, timestamps, max_seconds: float = 3, percentile: float = 5
) -> tuple[float, np.ndarray]:
    pairs = consecutive_pairs(timestamps, max_seconds)
    if not pairs:
        raise ValueError(f"no consecutive pairs within {max_seconds} seconds")
    first, second = np.array(pairs).T
    similarity = (vectors[first] * vectors[second]).sum(axis=1)
    return float(np.percentile(similarity, percentile)), similarity


def nearest_scores(queries: np.ndarray, train: np.ndarray, chunk: int = 512) -> np.ndarray:
    if len(train) == 0:
        raise ValueError("train part is empty")
    scores = [(queries[i : i + chunk] @ train.T).max(axis=1) for i in range(0, len(queries), chunk)]
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def leak_table(vectors, names, arms: dict, threshold: float, offset: float) -> list[dict]:
    index = {name: i for i, name in enumerate(names)}
    rows = []
    for arm, mapping in arms.items():
        train = vectors[[index[n] for n in names if mapping[n] == "train"]]
        for part in PARTS:
            members = [index[n] for n in names if mapping[n] == part]
            scores = nearest_scores(vectors[members], train)
            row = {
                "arm": arm,
                "part": part,
                "n": len(members),
                "hits": int((scores >= threshold).sum()),
            }
            for label, value in (
                ("share_t", threshold),
                ("share_t_minus", threshold - offset),
                ("share_t_plus", threshold + offset),
            ):
                row[label] = float((scores >= value).mean()) if len(members) else None
            rows.append(row)
    return rows


def session_max(similarity: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(groups, kind="stable")
    ordered = groups[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    block = similarity[np.ix_(order, order)]
    block = np.maximum.reduceat(np.maximum.reduceat(block, starts, axis=0), starts, axis=1)
    return block, ordered[starts]


def best_pair(similarity, members_a, members_b) -> tuple[int, int]:
    block = similarity[np.ix_(members_a, members_b)]
    i, j = np.unravel_index(int(np.argmax(block)), block.shape)
    return members_a[i], members_b[j]


def cross_pairs(similarity: np.ndarray, groups: np.ndarray) -> list[tuple]:
    block, ids = session_max(similarity, groups)
    upper = np.triu_indices(len(ids), k=1)
    values = block[upper]
    order = np.lexsort((upper[1], upper[0], -values))
    return [(int(ids[upper[0][k]]), int(ids[upper[1][k]]), float(values[k])) for k in order]


def class_names(records) -> str:
    present = sorted({c for r in records for c, _ in r.polygons})
    return ";".join(CLASSES[c] for c in present) or "none"


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def size_rows(records, sessions: dict[str, int], split: dict[str, str]) -> list[list]:
    members = {}
    for record in records:
        members.setdefault(sessions[record.name], []).append(record)
    rows = []
    for session_id in sorted(members):
        group = sorted(members[session_id], key=lambda r: (r.timestamp, r.name))
        instances = Counter(c for r in group for c, _ in r.polygons)
        rows.append(
            [
                session_id,
                group[0].day.isoformat(),
                group[0].timestamp.isoformat(),
                group[-1].timestamp.isoformat(),
                len(group),
                sum(not r.polygons for r in group),
                *[instances[c] for c in range(len(CLASSES))],
                split[group[0].name],
            ]
        )
    return rows


def contact_sheets(pairs, by_name: dict, out_dir: Path, per_sheet: int = 10, thumb: int = 192):
    out_dir.mkdir(parents=True, exist_ok=True)
    row_height = thumb + 20
    written = []
    for start in range(0, len(pairs), per_sheet):
        chunk = pairs[start : start + per_sheet]
        sheet = Image.new("RGB", (2 * thumb + 8, row_height * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (name_a, name_b, session_a, session_b, value) in enumerate(chunk):
            top = row * row_height
            for column, name in enumerate((name_a, name_b)):
                with Image.open(by_name[name].path) as image:
                    small = image.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
                sheet.paste(small, (column * (thumb + 8), top))
            label = f"{start + row + 1}. cos {value:.4f}  sessions {session_a} and {session_b}"
            draw.text((4, top + thumb + 4), label, fill="black")
        path = out_dir / f"pairs_{start // per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=85)
        written.append(path)
    return written


def audit(records, vectors, config: dict, out_dir: Path, sheet_dir: Path | None = None) -> dict:
    names = [r.name for r in records]
    by_name = {r.name: r for r in records}
    index = {name: i for i, name in enumerate(names)}
    sessions = assign_sessions(records, config["gap_seconds"])
    locked = session_split(records, config["gap_seconds"])
    check_disjoint(locked, sessions)

    pool = [n for n in names if locked[n] != "test"]
    assert_aggregate(pool, locked)
    pool_index = [index[n] for n in pool]
    threshold, pair_values = consecutive_threshold(
        vectors[pool_index],
        [by_name[n].timestamp for n in pool],
        config["pair_seconds"],
        config["percentile"],
    )
    threshold_record = {
        "threshold": threshold,
        "percentile": config["percentile"],
        "pair_seconds": config["pair_seconds"],
        "pairs": len(pair_values),
        "pair_similarity_min": float(pair_values.min()),
        "pair_similarity_median": float(np.median(pair_values)),
        "pair_similarity_max": float(pair_values.max()),
        "pool": "session split train and val",
        "pool_images": len(pool),
    }
    (out_dir / "threshold.json").write_text(json.dumps(threshold_record, indent=2), "utf-8")

    arms = {
        "export": export_split(records),
        "random": random_split(records, config["random_seed"]),
        "session": locked,
    }
    rows = leak_table(vectors, names, arms, threshold, config["offset"])
    columns = ["arm", "part", "n", "hits", "share_t", "share_t_minus", "share_t_plus"]
    write_csv(
        out_dir / "leak_share.csv",
        [*columns[:5], f"share_t_minus_{config['offset']}", f"share_t_plus_{config['offset']}"],
        [[row[c] if row[c] is not None else "" for c in columns] for row in rows],
    )
    val = {row["arm"]: row["share_t"] for row in rows if row["part"] == "val"}
    limit = config["max_ratio"] * val["random"]
    passed = val["session"] <= limit

    groups = np.array([sessions[n] for n in names])
    members = {}
    for i, n in enumerate(names):
        members.setdefault(sessions[n], []).append(i)
    similarity = vectors @ vectors.T
    all_links = [p for p in cross_pairs(similarity, groups) if p[2] >= threshold]
    open_names = [n for n in names if locked[n] != "test"]
    open_index = np.array([index[n] for n in open_names])
    open_pairs = cross_pairs(similarity[np.ix_(open_index, open_index)], groups[open_index])
    open_members = {s: [i for i in members[s] if locked[names[i]] != "test"] for s in members}
    link_rows, sheet_pairs = [], []
    for rank, (a, b, value) in enumerate(open_pairs):
        if value < threshold and rank >= config["sheet_pairs"]:
            break
        i, j = best_pair(similarity, open_members[a], open_members[b])
        name_a, name_b = names[i], names[j]
        assert_aggregate((name_a, name_b), locked)
        if rank < config["sheet_pairs"]:
            sheet_pairs.append((name_a, name_b, a, b, value))
        if value >= threshold:
            group_a = [by_name[names[k]] for k in members[a]]
            group_b = [by_name[names[k]] for k in members[b]]
            link_rows.append(
                [
                    a,
                    b,
                    round(value, 6),
                    len(group_a),
                    len(group_b),
                    group_a[0].day.isoformat(),
                    group_b[0].day.isoformat(),
                    class_names(group_a),
                    class_names(group_b),
                    locked[name_a],
                    locked[name_b],
                    name_a,
                    name_b,
                ]
            )
    for row in link_rows:
        assert_aggregate(row[-2:], locked)
    write_csv(
        out_dir / "session_links.csv",
        [
            "session_a",
            "session_b",
            "max_similarity",
            "images_a",
            "images_b",
            "day_a",
            "day_b",
            "classes_a",
            "classes_b",
            "split_a",
            "split_b",
            "name_a",
            "name_b",
        ],
        link_rows,
    )
    class_columns = [f"{name.replace(' ', '_')}_instances" for name in CLASSES]
    write_csv(
        out_dir / "session_sizes.csv",
        ["session", "day", "start", "end", "images", "empty_images", *class_columns, "split"],
        size_rows(records, sessions, locked),
    )
    sheets = []
    if sheet_dir is not None:
        for pair in sheet_pairs:
            assert_aggregate(pair[:2], locked)
        sheets = contact_sheets(sheet_pairs, by_name, sheet_dir)

    result = {
        "passed": passed,
        "threshold": threshold_record,
        "val_share": val,
        "limit": limit,
        "rows": rows,
        "links": len(link_rows),
        "links_all": len(all_links),
        "links_locked": len(all_links) - len(link_rows),
        "sessions": len(members),
        "sheets": [str(p) for p in sheets],
        "offset": config["offset"],
        "max_ratio": config["max_ratio"],
    }
    (out_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return result


def share_text(value) -> str:
    return "" if value is None else f"{value:.4f}"


def render_summary(result: dict) -> str:
    t = result["threshold"]
    offset = result["offset"]
    verdict = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# Similarity audit",
        "",
        f"Verdict for the session split: {verdict}",
        "",
        f"- Threshold T: {t['threshold']:.4f}, the {t['percentile']}th percentile of cosine "
        f"similarity over {t['pairs']} consecutive pairs at most {t['pair_seconds']} seconds "
        f"apart on the same day, from {t['pool_images']} images of the {t['pool']}",
        f"- Session val share at T: {result['val_share']['session']:.4f}, limit "
        f"{result['max_ratio']} times the random val share "
        f"{result['val_share']['random']:.4f} = {result['limit']:.4f}",
        f"- Export val share at T: {result['val_share']['export']:.4f} (descriptive)",
        "",
        f"| Arm | Part | Images | At or above T | Share at T | Share at T - {offset} "
        f"| Share at T + {offset} |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['arm']} | {row['part']} | {row['n']} | {row['hits']} | "
            f"{share_text(row['share_t'])} | {share_text(row['share_t_minus'])} | "
            f"{share_text(row['share_t_plus'])} |"
        )
    lines += [
        "",
        "## Sessions",
        "",
        f"- Sessions: {result['sessions']}",
        f"- Pairs of different sessions with highest cross similarity at or above T: "
        f"{result['links_all']}, of which {result['links_locked']} involve a locked test "
        f"session and are counted only here; the other {result['links']} are listed in "
        "`session_links.csv`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    start = float(os.environ.get("FABRID_JOB_START", time.time()))
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    heavy_dir = runio.runs_dir(config["step"], config["name"])
    vectors = embed(records, config, heavy_dir, start)
    if vectors is None:
        return
    print(f"embeddings cached in {heavy_dir / CACHE_NAME}")
    out_dir = runio.start_run(config["step"], config["name"], config)
    result = audit(records, vectors, config, out_dir, heavy_dir / "sheets")
    print(f"threshold T {result['threshold']['threshold']:.4f}")
    for arm, share in result["val_share"].items():
        print(f"{arm}: val share at T {share:.4f}")
    print(f"session split verdict: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"{result['links']} session links listed, {len(result['sheets'])} contact sheets")
    print(f"wrote {out_dir} and {heavy_dir}")
    if config.get("publish", True):
        runio.publish(out_dir)


if __name__ == "__main__":
    main()
