import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from conftest import make_dataset  # noqa: E402
from fabrid import hybrid, maskrcnn  # noqa: E402
from fabrid.labels import load_index  # noqa: E402

ARMS = {
    "parent": (False, False, False, False),
    "anchors": (True, False, False, False),
    "anchors_mssa": (True, True, False, False),
    "anchors_mssa_aux": (True, True, True, False),
    "anchors_mssa_aux_preprocess": (True, True, True, True),
}
SETTINGS = {
    "imgsz": 128,
    "aux": {"focal": 1.0, "dice": 1.0, "boundary": 0.5, "gamma": 2.0},
    "preprocess": {
        "clahe_clip": 2.0,
        "clahe_tile": 8,
        "gabor_angles": [0, 45, 90, 135],
        "gabor_frequency": 0.18,
        "gabor_sigma": 3.0,
        "gabor_gamma": 0.5,
        "gabor_strength": 0.15,
    },
}


def config_for(arm: str) -> dict:
    return {**SETTINGS, "components": dict(zip(hybrid.COMPONENTS, ARMS[arm], strict=True))}


def synthetic_batch(prepare):
    rng = np.random.default_rng(0)
    pixels = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(2)]
    if prepare is not None:
        pixels = [prepare(p) for p in pixels]
    images = [torch.from_numpy(p).permute(2, 0, 1).float() / 255 for p in pixels]
    masks = torch.zeros(1, 128, 128, dtype=torch.uint8)
    masks[0, 40:80, 30:90] = 1
    targets = [
        {
            "boxes": torch.tensor([[30.0, 40.0, 90.0, 80.0]]),
            "labels": torch.tensor([2]),
            "masks": masks,
        },
        {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros(0, 128, 128, dtype=torch.uint8),
        },
    ]
    return images, targets


def layout(state) -> dict:
    return {k: tuple(v.shape) for k, v in state.items() if "num_batches_tracked" not in k}


def has_gradient(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("arm", list(ARMS))
def test_switch_builds_and_runs_forward_backward(arm):
    config = config_for(arm)
    parts = config["components"]
    torch.manual_seed(0)
    model = hybrid.build_hybrid(config, pretrained=False).train()
    prepare = hybrid.preparer(config)
    assert (prepare is not None) == parts["preprocess"]
    images, targets = synthetic_batch(prepare)
    losses = model(images, targets)
    aux_keys = {"loss_aux_focal", "loss_aux_dice", "loss_aux_boundary"}
    assert (aux_keys <= set(losses)) == parts["aux"]
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    anchors = model.rpn.anchor_generator.num_anchors_per_location()
    assert anchors == ([15] * 4 if parts["anchors"] else [3] * 5)
    if parts["mssa"]:
        assert has_gradient(model.backbone.blocks)
    if parts["aux"]:
        assert has_gradient(model.aux)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 2
    assert {"boxes", "labels", "scores", "masks"} <= set(outputs[0])
    del model
    gc.collect()


def test_all_switches_off_matches_parent_layout():
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    model = hybrid.build_hybrid(config_for("parent"), pretrained=False)
    parent = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=5)
    ours = layout(model.state_dict())
    assert {k.replace("backbone.base.", "backbone."): v for k, v in ours.items()} == layout(
        parent.state_dict()
    )
    assert hybrid.parameter_counts(model)["parts"]["mssa"] == 0
    del model, parent
    gc.collect()


def test_boundary_of_filled_square_is_its_outline():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    expected = np.zeros((20, 20), dtype=np.float32)
    expected[4:16, 5:15] = 1
    expected[5:15, 4:16] = 1
    expected[6:14, 6:14] = 0
    np.testing.assert_array_equal(hybrid.morphological_boundary(mask), expected)
    assert hybrid.morphological_boundary(np.zeros((5, 5))).sum() == 0


def test_aux_targets_overwrite_and_follow_padding():
    masks = torch.zeros(2, 10, 12, dtype=torch.uint8)
    masks[0, 0:8, 0:8] = 1
    masks[1, 4:8, 4:8] = 1
    target = {"labels": torch.tensor([4, 1]), "masks": masks}
    empty = {"labels": torch.zeros(0, dtype=torch.int64), "masks": torch.zeros(0, 10, 12)}
    semantic, boundary = hybrid.aux_targets([target, empty], (16, 16), (8, 8), 5, "cpu")
    assert semantic.shape == (2, 8, 8)
    assert boundary.shape == (2, 1, 8, 8)
    assert semantic.dtype == torch.int64
    assert semantic[0, 0, 0] == 4
    assert semantic[0, 2, 2] == 1
    assert semantic[0, 6:].sum() == 0
    assert semantic[1].sum() == 0
    assert boundary[1].sum() == 0


def test_aux_losses_are_finite_on_empty_targets():
    logits = torch.randn(2, 5, 8, 8)
    target = torch.zeros(2, 8, 8, dtype=torch.int64)
    assert torch.isfinite(hybrid.focal_loss(logits, target, 2.0))
    assert hybrid.dice_loss(logits, target, 5) == 0
    assert torch.isfinite(hybrid.boundary_loss(torch.randn(2, 1, 8, 8), torch.zeros(2, 1, 8, 8)))


def test_strip_block_keeps_shape_and_size():
    block = hybrid.MSSA(256)
    x = torch.randn(2, 256, 17, 23)
    assert block(x).shape == x.shape
    assert sum(p.numel() for p in block.parameters()) == 378626


def thesis_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(cv2.merge([clahe.apply(light), a, b]), cv2.COLOR_LAB2RGB)


def thesis_gabor(rgb, frequency=0.18, sigma=3.0, gamma=0.5, strength=0.15):
    orientations = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ksize = max(7, int(round(sigma * 6)) | 1)
    lambd = 1.0 / max(frequency, 1e-3)
    responses = []
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel)))
    g = np.max(np.stack(responses, axis=0), axis=0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    out = rgb.astype(np.float32) * (1 - strength) + 255 * g[..., None] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def test_preprocess_matches_thesis_cells():
    pixels = np.random.default_rng(3).integers(0, 256, (96, 80, 3), dtype=np.uint8)
    ours = hybrid.enhance(pixels, SETTINGS["preprocess"])
    assert ours.dtype == np.uint8
    np.testing.assert_array_equal(ours, thesis_gabor(thesis_clahe(pixels)))


def mark_left_column(pixels):
    marked = pixels.copy()
    marked[:, 0] = 255
    return marked


def test_prepare_runs_before_flips(tmp_path):
    items = [("train", "IMG_20260501_090000_jpg.rf.a.jpg", [], 1)]
    records = load_index(make_dataset(tmp_path, items))
    plain = maskrcnn.Samples(records, prepare=mark_left_column)
    flipped = maskrcnn.Samples(records, (1.0, 0.0), prepare=mark_left_column)
    assert (plain.load(0)[0][:, 0] == 255).all()
    assert (flipped.load(0)[0][:, -1] == 255).all()
