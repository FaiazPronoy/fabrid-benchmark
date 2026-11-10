from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SQUARE = "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"


def write_image(path, seed, size=32):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=95)


def make_dataset(base, items):
    for folder, name, lines, seed in items:
        (base / folder / "images").mkdir(parents=True, exist_ok=True)
        (base / folder / "labels").mkdir(parents=True, exist_ok=True)
        write_image(base / folder / "images" / name, seed)
        stem = name.rsplit(".", 1)[0]
        (base / folder / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return base


def fake_record(name, timestamp, classes):
    square = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])
    polygons = [(c, square) for c in classes]
    return SimpleNamespace(
        name=name,
        timestamp=timestamp,
        day=timestamp.date(),
        polygons=polygons,
        n_lines=len(polygons),
        n_valid=len(polygons),
    )


@pytest.fixture
def records():
    rng = np.random.default_rng(7)
    start = datetime(2026, 5, 1, 9, 0, 0)
    out = []
    clock = start
    for session in range(60):
        clock += timedelta(minutes=10)
        if session % 20 == 0:
            clock += timedelta(days=1)
        for frame in range(int(rng.integers(1, 8))):
            stamp = clock + timedelta(seconds=5 * frame)
            classes = [] if rng.random() < 0.2 else [int(session % 4)] * int(rng.integers(1, 3))
            name = f"IMG_{stamp:%Y%m%d_%H%M%S}_{frame}_jpg.rf.x{session}.jpg"
            out.append(fake_record(name, stamp, classes))
    return out
