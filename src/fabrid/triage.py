import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fabrid import runio
from fabrid.geometry import COLORS
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import session_split
from fabrid.yolo import assert_unlocked, write_csv

ANSWERS = ("clean", "defect", "unsure")
CANDIDATE_COLUMNS = ["name", "split", "sheet", "position", "top_score", "top_class", "detections"]
FORM_COLUMNS = ["name", "reviewer_a", "reviewer_b", "note"]
CAPTION = 34


def empty_images(records, locked: dict[str, str]) -> tuple[list, list]:
    empty = [r for r in records if r.n_lines == 0]
    open_part = [r for r in empty if locked[r.name] in ("train", "val")]
    locked_part = [r for r in empty if locked[r.name] == "test"]
    return open_part, locked_part


def detect(records, locked: dict[str, str], predictor) -> dict[str, list]:
    assert_unlocked({r.name: locked[r.name] for r in records}, locked)
    return predictor([r.path for r in records])


def by_score(names, detections: dict[str, list]) -> list[str]:
    def top(name):
        found = detections.get(name, [])
        return max((d[1] for d in found), default=-1.0)

    return sorted(names, key=lambda n: (-top(n), n))


def shuffled(names, seed: int) -> list[str]:
    ordered = sorted(names)
    return [ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]


def find_weights(relative: str, roots) -> Path | None:
    for root in roots:
        root = Path(root)
        if (root / relative).is_file():
            return root / relative
        if root.is_dir():
            found = sorted(root.glob(f"**/{relative}"))
            if found:
                return found[0]
    return None


def yolo_predictor(weights: Path, config: dict):
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(paths) -> dict[str, list]:
        found = {}
        for start in range(0, len(paths), config["batch"]):
            chunk = [str(p) for p in paths[start : start + config["batch"]]]
            results = model.predict(
                chunk,
                conf=config["conf"],
                imgsz=config["imgsz"],
                device=config["device"],
                verbose=False,
            )
            for path, result in zip(chunk, results, strict=True):
                boxes = result.boxes
                found[Path(path).name] = [
                    (int(c), float(s), [float(v) for v in xyxy])
                    for c, s, xyxy in zip(
                        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
                    )
                ]
        return found

    return predict


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, typeface, width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        if current and draw.textlength(current + char, font=typeface) > width:
            lines.append(current)
            current = ""
        current += char
    return [*lines, current][:2]


def tile(record, found, size: int, typeface) -> Image.Image:
    with Image.open(record.path) as image:
        rgb = image.convert("RGB")
    scale = size / max(rgb.size)
    shown = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)))
    left, top = (size - shown.width) // 2, (size - shown.height) // 2
    canvas = Image.new("RGB", (size, size + CAPTION), (24, 24, 24))
    canvas.paste(shown, (left, top))
    draw = ImageDraw.Draw(canvas)
    for class_id, score, (x0, y0, x1, y1) in found or []:
        box = [left + x0 * scale, top + y0 * scale, left + x1 * scale, top + y1 * scale]
        draw.rectangle(box, outline=COLORS[class_id], width=2)
        label = f"{CLASSES[class_id]} {score:.2f}"
        corner = (box[0] + 2, box[1] + 2)
        draw.rectangle(draw.textbbox(corner, label, font=typeface), fill=(0, 0, 0))
        draw.text(corner, label, fill=COLORS[class_id], font=typeface)
    for row, line in enumerate(wrap(draw, record.name, typeface, size - 8)):
        draw.text((4, size + 3 + row * 15), line, fill=(235, 235, 235), font=typeface)
    return canvas


def write_sheets(groups, by_name, detections, config: dict, out_dir: Path) -> dict[str, tuple]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet, size, columns = config["per_sheet"], config["tile"], config["columns"]
    typeface = font(13)
    placed = {}
    for prefix, names in groups:
        for start in range(0, len(names), per_sheet):
            members = names[start : start + per_sheet]
            sheet = f"{prefix}_{start // per_sheet + 1:03d}.jpg"
            rows = math.ceil(len(members) / columns)
            canvas = Image.new("RGB", (columns * size, rows * (size + CAPTION)), (10, 10, 10))
            for position, name in enumerate(members):
                found = detections.get(name) if prefix != "test" else None
                image = tile(by_name[name], found, size, typeface)
                spot = ((position % columns) * size, (position // columns) * (size + CAPTION))
                canvas.paste(image, spot)
                placed[name] = (sheet, position + 1)
            canvas.save(out_dir / sheet, quality=90)
    return placed


def candidate_rows(order, locked, detections, placed) -> list[list]:
    rows = []
    for name in order:
        sheet, position = placed[name]
        if locked[name] == "test":
            rows.append([name, "test", sheet, position, "", "", ""])
            continue
        found = detections.get(name, [])
        best = max(found, key=lambda d: d[1], default=None)
        score = round(best[1], 4) if best else 0.0
        top_class = CLASSES[best[0]] if best else ""
        rows.append([name, locked[name], sheet, position, score, top_class, len(found)])
    return rows


def sheets(config: dict, records, locked, predictor=None) -> Path:
    open_part, locked_part = empty_images(records, locked)
    if predictor is None:
        roots = [Path(root) for root in config["roots"]]
        weights = find_weights(config["weights"], roots)
        if weights is None:
            raise SystemExit(f"weights {config['weights']} not found under {config['roots']}")
        print(f"ordering train and val with {weights}")
        predictor = yolo_predictor(weights, config)
    detections = detect(open_part, locked, predictor)
    first = by_score([r.name for r in open_part], detections)
    second = shuffled([r.name for r in locked_part], config["seed"])
    by_name = {r.name: r for r in records}
    sheets_dir = runio.runs_dir(config["step"], config["name"]) / "sheets"
    groups = [("trainval", first), ("test", second)]
    placed = write_sheets(groups, by_name, detections, config, sheets_dir)
    run_dir = runio.start_run(config["step"], config["name"], config)
    write_csv(
        run_dir / "candidates.csv",
        CANDIDATE_COLUMNS,
        candidate_rows(first + second, locked, detections, placed),
    )
    write_csv(run_dir / "review_form.csv", FORM_COLUMNS, [[n, "", "", ""] for n in first + second])
    n_sheets = len({sheet for sheet, _ in placed.values()})
    print(
        f"{len(first)} train and val and {len(second)} test empty-label images "
        f"on {n_sheets} sheets in {sheets_dir}"
    )
    print(f"wrote {run_dir / 'candidates.csv'} and {run_dir / 'review_form.csv'}")
    return run_dir


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_form(path: Path, expected: list[str]) -> list[tuple[str, str, str]]:
    rows = read_table(path)
    names = [row["name"] for row in rows]
    if sorted(names) != sorted(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            f"form names differ from candidates: missing {missing[:5]}, extra {extra[:5]}"
        )
    answers = []
    for row in rows:
        pair = []
        for column in ("reviewer_a", "reviewer_b"):
            answer = (row.get(column) or "").strip().lower()
            if answer not in ANSWERS:
                raise ValueError(
                    f"{row['name']}: {column} answer {answer!r} is not one of {ANSWERS}"
                )
            pair.append(answer)
        answers.append((row["name"], *pair))
    return answers


def kappa(table: np.ndarray) -> float | None:
    total = table.sum()
    if total == 0:
        return None
    observed = np.trace(table) / total
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / total**2
    if expected == 1:
        return None
    return float((observed - expected) / (1 - expected))


def agreement(answers, splits: dict[str, str]) -> dict:
    def summarize(subset) -> dict:
        table = np.zeros((len(ANSWERS), len(ANSWERS)), dtype=int)
        for _, a, b in subset:
            table[ANSWERS.index(a), ANSWERS.index(b)] += 1
        n = len(subset)
        both_clean = sum(a == b == "clean" for _, a, b in subset)
        any_defect = sum("defect" in (a, b) for _, a, b in subset)
        value = kappa(table)
        return {
            "images": n,
            "pairs": {
                f"{a}|{b}": int(table[i, j])
                for i, a in enumerate(ANSWERS)
                for j, b in enumerate(ANSWERS)
            },
            "cohen_kappa": None if value is None else round(value, 4),
            "both_clean": both_clean,
            "both_clean_share": round(both_clean / n, 4) if n else None,
            "any_defect_share": round(any_defect / n, 4) if n else None,
        }

    result = summarize(answers)
    result["per_split"] = {
        split: summarize([a for a in answers if splits[a[0]] == split])
        for split in ("train", "val", "test")
        if any(splits[a[0]] == split for a in answers)
    }
    return result


def collect(config: dict) -> Path:
    root = runio.repo_root()
    base = root / "results" / config["step"] / config["name"]
    candidates = read_table(base / "candidates.csv")
    splits = {row["name"]: row["split"] for row in candidates}
    form = Path(config.get("form") or base / "review_form.csv")
    if not form.is_absolute():
        form = root / form
    answers = read_form(form, list(splits))
    summary = agreement(answers, splits)
    run_dir = runio.start_run(config["step"], f"{config['name']}/review", config)
    (run_dir / "agreement.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    clean = sorted((n, splits[n]) for n, a, b in answers if a == b == "clean")
    write_csv(run_dir / "verified_clean.csv", ["name", "split"], clean)
    print(
        f"{summary['both_clean']} of {summary['images']} empty-label images verified clean "
        f"(share {summary['both_clean_share']}, kappa {summary['cohen_kappa']})"
    )
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("sheets", "collect"))
    args = parser.parse_args()
    config = runio.load_config(args.config)
    mode = args.mode or config["mode"]
    config = {**config, "mode": mode}
    if mode == "sheets":
        records = load_index()
        print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
        run_dir = sheets(config, records, session_split(records, config["gap_seconds"]))
    else:
        run_dir = collect(config)
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
