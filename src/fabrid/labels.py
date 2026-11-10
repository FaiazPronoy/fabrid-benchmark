import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "/kaggle/input/fabrid-v9"
CLASSES = ("needle mark", "hole", "missing yarn", "stain")
EXPORT_FOLDERS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NAME_PATTERN = re.compile(r"^IMG_(\d{8})_(\d{6})")


@dataclass
class Record:
    path: Path
    label_path: Path | None
    name: str
    stem: str
    folder: str
    timestamp: datetime
    day: date
    polygons: list[tuple[int, np.ndarray]] = field(default_factory=list)
    n_lines: int = 0
    n_valid: int = 0


def data_root(root: str | Path | None = None) -> Path:
    base = Path(root or os.environ.get("FABRID_DATA", DEFAULT_ROOT))
    if not base.is_dir():
        raise FileNotFoundError(f"data root not found: {base}")
    found = [
        Path(dirpath)
        for dirpath, dirnames, _ in os.walk(base)
        if "train" in dirnames and (Path(dirpath) / "train" / "images").is_dir()
    ]
    if not found:
        raise FileNotFoundError(f"no directory with train/images under {base}")
    return min(found, key=lambda p: (len(p.parts), str(p)))


def parse_timestamp(name: str) -> datetime:
    match = NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"file name does not start with IMG_YYYYMMDD_HHMMSS: {name}")
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def stem_of(name: str) -> str:
    base = name.split(".rf.")[0] if ".rf." in name else Path(name).stem
    return re.sub(r"_(jpe?g|png)$", "", base, flags=re.IGNORECASE)


def parse_line(line: str) -> tuple[int, np.ndarray] | None:
    parts = line.split()
    if len(parts) < 3 or len(parts) % 2 == 0:
        return None
    try:
        class_id = int(parts[0])
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float64)
    except ValueError:
        return None
    if not 0 <= class_id < len(CLASSES) or not np.isfinite(coords).all():
        return None
    return class_id, coords.reshape(-1, 2)


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def is_valid(points: np.ndarray) -> bool:
    return len(points) >= 3 and polygon_area(points) > 0


def label_lines(label_path: Path | None) -> list[str]:
    if label_path is None or not label_path.is_file():
        return []
    text = label_path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_labels(label_path: Path | None) -> tuple[list[tuple[int, np.ndarray]], int]:
    lines = label_lines(label_path)
    polygons = []
    for line in lines:
        parsed = parse_line(line)
        if parsed is not None and is_valid(parsed[1]):
            polygons.append(parsed)
    return polygons, len(lines)


def load_index(root: str | Path | None = None) -> list[Record]:
    base = data_root(root)
    records = []
    for folder in EXPORT_FOLDERS:
        image_dir = base / folder / "images"
        if not image_dir.is_dir():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = base / folder / "labels" / f"{path.stem}.txt"
            polygons, n_lines = read_labels(label_path)
            timestamp = parse_timestamp(path.name)
            records.append(
                Record(
                    path=path,
                    label_path=label_path if label_path.is_file() else None,
                    name=path.name,
                    stem=stem_of(path.name),
                    folder=folder,
                    timestamp=timestamp,
                    day=timestamp.date(),
                    polygons=polygons,
                    n_lines=n_lines,
                    n_valid=len(polygons),
                )
            )
    names = [r.name for r in records]
    if len(set(names)) != len(names):
        raise ValueError("image file names are not unique across export folders")
    return sorted(records, key=lambda r: r.name)
