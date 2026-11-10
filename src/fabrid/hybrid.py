import argparse
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.misc import FrozenBatchNorm2d

from fabrid import maskrcnn, runio
from fabrid.labels import CLASSES

STRIPS = ((1, 3), (3, 1), (1, 5), (5, 1), (1, 7), (7, 1))
LEVELS = ("0", "1", "2", "3")
ANCHOR_SIZES = ((8, 16, 24), (16, 32, 48), (32, 64, 96), (64, 128, 192))
ANCHOR_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
COMPONENTS = ("anchors", "mssa", "aux", "preprocess")
N_OUTPUTS = len(CLASSES) + 1


def conv_bn_silu(inputs: int, outputs: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(),
    )


class MSSA(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(16, channels // reduction)
        self.branches = nn.ModuleList(
            conv_bn_silu(channels, mid, size, (size[0] // 2, size[1] // 2)) for size in STRIPS
        )
        self.fuse = conv_bn_silu(mid * len(STRIPS), channels, 1, 0)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        y = y * self.channel(y)
        pooled = torch.cat([y.mean(dim=1, keepdim=True), y.amax(dim=1, keepdim=True)], dim=1)
        y = y * self.spatial(pooled)
        return F.silu(x + self.out(y))


class Neck(nn.Module):
    def __init__(self, base: nn.Module, levels: tuple[str, ...], mssa: bool):
        super().__init__()
        self.base = base
        self.out_channels = base.out_channels
        self.levels = levels
        blocks = {level: MSSA(base.out_channels) for level in levels}
        self.blocks = nn.ModuleDict(blocks) if mssa else None

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features = self.base(x)
        kept = OrderedDict((k, v) for k, v in features.items() if k in self.levels)
        if self.blocks is None:
            return kept
        return OrderedDict((k, self.blocks[k](v)) for k, v in kept.items())


def morphological_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    import cv2

    plane = (np.asarray(mask) > 0).astype(np.uint8)
    if plane.max() == 0:
        return np.zeros_like(plane, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return (cv2.morphologyEx(plane, cv2.MORPH_GRADIENT, kernel) > 0).astype(np.float32)


def nearest(plane: np.ndarray, grid) -> torch.Tensor:
    tensor = torch.from_numpy(plane)[None, None].float()
    return F.interpolate(tensor, size=tuple(grid), mode="nearest")[0]


def aux_targets(targets, padded, grid, n_outputs: int, device):
    semantic, boundary = [], []
    for target in targets:
        plane = np.zeros(tuple(padded), dtype=np.int64)
        masks = target["masks"].detach().cpu().numpy()
        labels = target["labels"].detach().cpu().numpy().astype(np.int64)
        for mask, label in zip(masks, labels, strict=True):
            height, width = min(padded[0], mask.shape[0]), min(padded[1], mask.shape[1])
            inside = mask[:height, :width] > 0
            plane[:height, :width][inside] = int(np.clip(label, 1, n_outputs - 1))
        semantic.append(nearest(plane, grid)[0].long())
        boundary.append(nearest(morphological_boundary(plane > 0), grid))
    return torch.stack(semantic).to(device), torch.stack(boundary).to(device)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
    pt = torch.exp(-ce.clamp(max=50))
    valid = target != 255
    return (((1 - pt) ** gamma) * ce)[valid].mean() if valid.any() else logits.sum() * 0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, n_outputs: int) -> torch.Tensor:
    probs = logits.softmax(1)
    losses = []
    for class_index in range(1, n_outputs):
        p = probs[:, class_index]
        t = (target == class_index).float()
        target_pixels = t.sum(dim=(1, 2))
        valid = target_pixels > 0
        if valid.any():
            inter = (p[valid] * t[valid]).sum(dim=(1, 2))
            den = p[valid].sum(dim=(1, 2)) + target_pixels[valid]
            losses.append(1 - ((2 * inter + 1e-5) / (den + 1e-5)).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1e-5) / (den + 1e-5)).mean()
    return 0.5 * bce + 0.5 * dice


class AuxHeads(nn.Module):
    def __init__(self, channels: int, settings: dict):
        super().__init__()
        self.settings = dict(settings)
        self.semantic = nn.Sequential(
            conv_bn_silu(channels, 128, 3, 1), nn.Conv2d(128, N_OUTPUTS, 1)
        )
        self.boundary = nn.Sequential(conv_bn_silu(channels, 64, 3, 1), nn.Conv2d(64, 1, 1))

    def losses(self, feature: torch.Tensor, padded, targets) -> dict[str, torch.Tensor]:
        grid = feature.shape[-2:]
        semantic, boundary = aux_targets(targets, padded, grid, N_OUTPUTS, feature.device)
        semantic_logits = self.semantic(feature)
        boundary_logits = self.boundary(feature)
        weights = self.settings
        with torch.autocast(feature.device.type, enabled=False):
            semantic_logits = semantic_logits.float()
            focal = focal_loss(semantic_logits, semantic, weights["gamma"])
            dice = dice_loss(semantic_logits, semantic, N_OUTPUTS)
            edge = boundary_loss(boundary_logits.float(), boundary)
        return {
            "loss_aux_focal": weights["focal"] * focal,
            "loss_aux_dice": weights["dice"] * dice,
            "loss_aux_boundary": weights["boundary"] * edge,
        }


def enhance(pixels: np.ndarray, settings: dict) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    tile = int(settings["clahe_tile"])
    clahe = cv2.createCLAHE(clipLimit=float(settings["clahe_clip"]), tileGridSize=(tile, tile))
    rgb = cv2.cvtColor(cv2.merge([clahe.apply(light), a, b]), cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigma = float(settings["gabor_sigma"])
    size = max(7, int(round(sigma * 6)) | 1)
    wavelength = 1.0 / max(float(settings["gabor_frequency"]), 1e-3)
    responses = []
    for theta in np.deg2rad(settings["gabor_angles"]):
        kernel = cv2.getGaborKernel(
            (size, size),
            sigma,
            theta,
            wavelength,
            float(settings["gabor_gamma"]),
            0,
            ktype=cv2.CV_32F,
        )
        responses.append(np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel)))
    g = np.max(np.stack(responses, axis=0), axis=0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    strength = float(settings["gabor_strength"])
    out = rgb.astype(np.float32) * (1 - strength) + 255 * g[..., None] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def preparer(config: dict):
    if not config["components"]["preprocess"]:
        return None
    return partial(enhance, settings=config["preprocess"])


class Hybrid(MaskRCNN):
    def __init__(self, backbone, num_classes: int, aux=None, **kwargs):
        super().__init__(backbone, num_classes=num_classes, **kwargs)
        self.aux = aux
        self.captured = {}
        if aux is not None:
            self.transform.register_forward_hook(self.capture("transform"))
            self.backbone.register_forward_hook(self.capture("features"))

    def capture(self, key: str):
        def hook(module, inputs, output):
            self.captured[key] = output

        return hook

    def forward(self, images, targets=None):
        output = super().forward(images, targets)
        captured, self.captured = self.captured, {}
        if self.aux is None or not self.training:
            return output
        image_list, resized = captured["transform"]
        padded = image_list.tensors.shape[-2:]
        return {**output, **self.aux.losses(captured["features"]["0"], padded, resized)}


def load_coco(model: nn.Module) -> tuple[int, int]:
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    source = weights.get_state_dict(progress=False, check_hash=True)
    target = model.state_dict()
    matched = {}
    for key, value in source.items():
        name = (
            "backbone.base." + key.removeprefix("backbone.") if key.startswith("backbone.") else key
        )
        if name in target and target[name].shape == value.shape:
            matched[name] = value
    model.load_state_dict(matched, strict=False)
    overwrite_eps(model, 0.0)
    return len(matched), len(target) - len(matched)


def build_hybrid(config: dict, pretrained: bool = True) -> Hybrid:
    parts = config["components"]
    base = _resnet_fpn_extractor(resnet50(norm_layer=FrozenBatchNorm2d), 3)
    levels = LEVELS if parts["anchors"] else (*LEVELS, "pool")
    options = {"min_size": config["imgsz"], "max_size": config["imgsz"]}
    if parts["anchors"]:
        ratios = (ANCHOR_RATIOS,) * len(ANCHOR_SIZES)
        options["rpn_anchor_generator"] = AnchorGenerator(ANCHOR_SIZES, ratios)
    aux = AuxHeads(base.out_channels, config["aux"]) if parts["aux"] else None
    model = Hybrid(Neck(base, levels, parts["mssa"]), N_OUTPUTS, aux, **options)
    if pretrained:
        loaded, fresh = load_coco(model)
        print(f"loaded {loaded} COCO tensors, {fresh} tensors start from initialization")
    return model


def count(module: nn.Module | None, trainable: bool = False) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable)


def parameter_counts(model: Hybrid) -> dict:
    parts = {
        "backbone_fpn": model.backbone.base,
        "mssa": model.backbone.blocks,
        "rpn": model.rpn,
        "roi_heads": model.roi_heads,
        "aux": model.aux,
    }
    total = count(model)
    return {
        "total": total,
        "trainable": count(model, trainable=True),
        "inference": total - count(model.aux),
        "parts": {name: count(module) for name, module in parts.items()},
    }


def parent_config(config: dict) -> dict:
    return {**config, "components": dict.fromkeys(COMPONENTS, False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    unknown = set(config["components"]) ^ set(COMPONENTS)
    if unknown:
        raise ValueError(f"components must be exactly {COMPONENTS}, got {sorted(unknown)}")
    extra = {
        "components": config["components"],
        "parameters": parameter_counts(build_hybrid(config, pretrained=False)),
        "parameters_parent": parameter_counts(build_hybrid(parent_config(config), False)),
    }
    print(
        f"{config['model']}: {extra['parameters']['total']} parameters, parent "
        f"{extra['parameters_parent']['total']}"
    )
    maskrcnn.run(config, build_hybrid, extra, preparer(config))


if __name__ == "__main__":
    main()
