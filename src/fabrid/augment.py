import argparse
import zlib
from functools import partial

import numpy as np
from PIL import Image

from fabrid import maskrcnn, runio
from fabrid.geometry import rasterize
from fabrid.labels import load_index
from fabrid.sessions import assign_sessions, session_split
from fabrid.yolo import assert_unlocked

AUG_STREAM = 1
PASTE_STREAM = 2
GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def generator(seed: int, epoch: int, name: str, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, epoch, zlib.crc32(name.encode()), stream])


def keep_instances(masks: np.ndarray, labels: np.ndarray, min_pixels: int) -> dict:
    keep = masks.sum(axis=(1, 2)) >= min_pixels
    masks = np.ascontiguousarray(masks[keep])
    return {"boxes": maskrcnn.boxes_from_masks(masks), "labels": labels[keep], "masks": masks}


def stack(masks: list[np.ndarray], height: int, width: int) -> np.ndarray:
    return np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)


def jitter(pixels: np.ndarray, rng: np.random.Generator, settings: dict) -> np.ndarray:
    import cv2

    low, high = settings["jitter"]
    brightness, contrast, saturation = rng.uniform(low, high, 3)
    hue = rng.uniform(-settings["hue"], settings["hue"])
    image = np.clip(pixels.astype(np.float32) * brightness, 0, 255)
    mean = float((image @ GRAY).mean())
    image = np.clip((image - mean) * contrast + mean, 0, 255)
    gray = (image @ GRAY)[..., None]
    image = np.clip((image - gray) * saturation + gray, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + round(hue * 180)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def affine(rng: np.random.Generator, height: int, width: int, settings: dict) -> np.ndarray:
    import cv2

    scale = rng.uniform(*settings["scale"])
    angle = rng.uniform(-settings["rotation"], settings["rotation"])
    room = np.array([width, height]) * abs(scale - 1) / 2
    shift = rng.uniform(-room, room)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += shift
    return matrix


class Augment:
    def __init__(self, settings: dict, seed: int, min_pixels: int):
        self.settings = settings
        self.seed = seed
        self.min_pixels = min_pixels

    def __call__(self, pixels: np.ndarray, target: dict, record, epoch: int):
        import cv2

        rng = generator(self.seed, epoch, record.name, AUG_STREAM)
        pixels = jitter(pixels, rng, self.settings)
        height, width = pixels.shape[:2]
        matrix = affine(rng, height, width, self.settings)
        fill = tuple(float(v) for v in pixels.reshape(-1, 3).mean(axis=0))
        pixels = cv2.warpAffine(
            pixels, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=fill
        )
        masks = [
            cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
            for mask in target["masks"]
        ]
        masks = stack(masks, height, width)
        return pixels, keep_instances(masks, target["labels"], self.min_pixels)


def paste_instance(pixels, masks, patch, piece, top: int, left: int, sigma: float):
    import cv2

    height, width = piece.shape
    inside = piece > 0
    region = pixels[top : top + height, left : left + width].astype(np.float32)
    patch = patch + (region[inside].mean(axis=0) - patch[inside].mean(axis=0))
    alpha = np.clip(cv2.GaussianBlur(piece.astype(np.float32), (0, 0), sigma), 0, 1)[..., None]
    blended = alpha * patch + (1 - alpha) * region
    pixels[top : top + height, left : left + width] = np.clip(blended, 0, 255).astype(np.uint8)
    added = np.zeros(pixels.shape[:2], dtype=np.uint8)
    added[top : top + height, left : left + width] = inside
    for mask in masks:
        mask[added > 0] = 0
    masks.append(added)


class Paste:
    def __init__(self, train_records, sessions, mapping, locked, settings, seed, min_pixels):
        for record in train_records:
            if mapping.get(record.name) != "train":
                raise AssertionError(f"paste donor {record.name} is not in the training part")
        assert_unlocked({r.name: mapping[r.name] for r in train_records}, locked)
        self.sessions = {record.name: sessions[record.name] for record in train_records}
        self.donors = [
            (record, index) for record in train_records for index in range(len(record.polygons))
        ]
        self.settings = settings
        self.seed = seed
        self.min_pixels = min_pixels

    def candidates(self, name: str) -> list:
        if name not in self.sessions:
            raise AssertionError(f"paste destination {name} is not in the training part")
        session = self.sessions[name]
        return [donor for donor in self.donors if self.sessions[donor[0].name] != session]

    def cut(self, record, index: int, rng: np.random.Generator):
        import cv2

        with Image.open(record.path) as image:
            source = np.array(image.convert("RGB"))
        height, width = source.shape[:2]
        class_id, points = record.polygons[index]
        mask = rasterize(points * [width, height], height, width)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return class_id, np.zeros((0, 0, 3), np.float32), np.zeros((0, 0), np.uint8)
        margin = self.settings["margin"]
        top, bottom = max(0, rows.min() - margin), min(height, rows.max() + 1 + margin)
        left, right = max(0, cols.min() - margin), min(width, cols.max() + 1 + margin)
        patch = source[top:bottom, left:right].astype(np.float32)
        piece = mask[top:bottom, left:right].astype(np.uint8)
        factor = rng.uniform(*self.settings["scale"])
        size = (max(1, round(piece.shape[1] * factor)), max(1, round(piece.shape[0] * factor)))
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR)
        piece = cv2.resize(piece, size, interpolation=cv2.INTER_NEAREST)
        if rng.random() < self.settings["flip"]:
            patch, piece = patch[:, ::-1], piece[:, ::-1]
        if rng.random() < self.settings["flip"]:
            patch, piece = patch[::-1], piece[::-1]
        return class_id, np.ascontiguousarray(patch), np.ascontiguousarray(piece)

    def __call__(self, pixels: np.ndarray, target: dict, record, epoch: int):
        rng = generator(self.seed, epoch, record.name, PASTE_STREAM)
        pool = self.candidates(record.name)
        if rng.random() >= self.settings["probability"] or not pool:
            return pixels, target
        low, high = self.settings["count"]
        pixels = pixels.copy()
        height, width = pixels.shape[:2]
        masks = [mask.copy() for mask in target["masks"]]
        labels = list(target["labels"])
        for _ in range(int(rng.integers(low, high + 1))):
            donor, index = pool[int(rng.integers(len(pool)))]
            class_id, patch, piece = self.cut(donor, index, rng)
            if piece.shape[0] > height or piece.shape[1] > width or not piece.any():
                continue
            top = int(rng.integers(0, height - piece.shape[0] + 1))
            left = int(rng.integers(0, width - piece.shape[1] + 1))
            paste_instance(pixels, masks, patch, piece, top, left, self.settings["sigma"])
            labels.append(class_id + 1)
        merged = stack(masks, height, width)
        return pixels, keep_instances(merged, np.array(labels, dtype=np.int64), self.min_pixels)


class Chain:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, pixels: np.ndarray, target: dict, record, epoch: int):
        for step in self.steps:
            pixels, target = step(pixels, target, record, epoch)
        return pixels, target


def build_transform(config: dict, train_records, mapping, locked, sessions) -> Chain | None:
    remedies = config["remedies"]
    steps = []
    if remedies["paste"]:
        steps.append(
            Paste(
                train_records,
                sessions,
                mapping,
                locked,
                config["paste"],
                config["seed"],
                config["min_pixels"],
            )
        )
    if remedies["aug"]:
        steps.append(Augment(config["aug"], config["seed"], config["min_pixels"]))
    return Chain(steps) if steps else None


def last_epoch(state: dict, remedies: dict) -> dict:
    row = state["history"][-1]
    return {
        "remedies": remedies,
        "last_epoch": {
            "epoch": row["epoch"],
            "train_loss": row["loss"],
            "val_mask_AP50": row["mask_AP50"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = maskrcnn.with_fold(runio.load_config(parser.parse_args().config))
    gap = config["gap_seconds"]
    records = load_index()
    locked = session_split(records, gap)
    mapping = maskrcnn.view_of(records, config)
    train_records, _ = maskrcnn.split_records(records, mapping, locked)
    transform = build_transform(
        config, train_records, mapping, locked, assign_sessions(records, gap)
    )
    print(f"{config['model']}: remedies {config['remedies']} on {len(train_records)} train images")
    extra = partial(last_epoch, remedies=config["remedies"])
    maskrcnn.run(config, extra=extra, transform=transform)


if __name__ == "__main__":
    main()
