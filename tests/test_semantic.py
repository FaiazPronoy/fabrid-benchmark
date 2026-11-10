from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fabrid import semantic  # noqa: E402


def square(x0, y0, x1, y1, width, height):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) / [width, height]


def test_target_later_line_wins_and_overlap_counted():
    first = (0, square(0, 0, 6, 6, 10, 10))
    second = (3, square(4, 4, 10, 10, 10, 10))
    same = (3, square(8, 8, 10, 10, 10, 10))
    target, contested = semantic.build_target([first, second, same], 10, 10)
    assert target[0, 0] == 1
    assert target[5, 5] == 4
    assert target[9, 9] == 4
    assert contested == 4
    reverse, contested_reverse = semantic.build_target([second, first], 10, 10)
    assert reverse[5, 5] == 1
    assert contested_reverse == 4
    assert (target == 1).sum() == 36 - 4


def test_target_empty_image():
    target, contested = semantic.build_target([], 12, 8)
    assert target.shape == (12, 8)
    assert target.dtype == np.uint8
    assert not target.any()
    assert contested == 0


def test_scores_on_hand_built_case():
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0, :2] = 1
    target[3, 3] = semantic.IGNORE
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, 0] = 1
    pred[1, 0] = 1
    pred[3, 3] = 2
    counts = semantic.confusion(pred, target)
    assert counts[1].tolist() == [1, 1, 1]
    assert counts[2].tolist() == [0, 0, 0]
    per_class = semantic.class_scores(counts)
    needle = per_class["needle mark"]
    assert needle["dice"] == pytest.approx(0.5)
    assert needle["iou"] == pytest.approx(1 / 3)
    assert needle["precision"] == pytest.approx(0.5)
    assert needle["recall"] == pytest.approx(0.5)
    assert per_class["hole"]["dice"] is None
    assert semantic.mean_dice(per_class) == pytest.approx(0.5)
    counts[2] = [0, 3, 0]
    assert semantic.mean_dice(semantic.class_scores(counts)) == pytest.approx(0.25)


def test_pooled_dice_is_not_mean_of_images():
    target_a, pred_a = np.ones((2, 2), np.uint8), np.ones((2, 2), np.uint8)
    target_b, pred_b = np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)
    pred_b[0, 0] = 1
    counts = semantic.confusion(pred_a, target_a) + semantic.confusion(pred_b, target_b)
    assert semantic.class_scores(counts)["needle mark"]["dice"] == pytest.approx(8 / 9)


def test_alarm_threshold_ignores_padding():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[:, 8:] = semantic.IGNORE
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[:, 6:] = 3
    assert semantic.is_alarm(pred, target, 20)
    assert not semantic.is_alarm(pred, target, 21)


def test_padding_keeps_labels_aligned(tmp_path):
    height, width = 40, 56
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[5:15, 30:50] = 255
    path = tmp_path / "IMG_20260501_090000.jpg"
    Image.fromarray(pixels).save(path, format="PNG")
    record = SimpleNamespace(path=path, polygons=[(2, square(30, 5, 50, 15, width, height))])
    for flips in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        image, target = semantic.Samples([record], 64, flips).load(0)
        assert image.shape == (64, 64, 3) and target.shape == (64, 64)
        assert (target[height:] == semantic.IGNORE).all()
        assert (target[:, width:] == semantic.IGNORE).all()
        bright = image[:height, :width, 0] > 0
        assert ((target[:height, :width] == 3) == bright).all()
        assert (image[height:] == 0).all()


def test_padding_rejects_large_image():
    with pytest.raises(ValueError):
        semantic.pad(np.zeros((70, 10, 3)), np.zeros((70, 10), np.uint8), 64)


def test_dice_loss_ignores_padding():
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 0, :2] = 1
    perfect = torch.full((1, semantic.N_CLASSES, 4, 4), -20.0)
    perfect[:, 0] = 20
    perfect[0, 1, 0, :2], perfect[0, 0, 0, :2] = 20, -20
    assert float(semantic.dice_loss(perfect, target)) < 0.01
    padded = target.clone()
    padded[0, 3] = semantic.IGNORE
    noisy = perfect.clone()
    noisy[0, 1, 3], noisy[0, 0, 3] = 20, -20
    assert float(semantic.dice_loss(noisy, padded)) < 0.01
    assert float(semantic.dice_loss(noisy, target)) > 0.1


def tiny_head(size=56):
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    backbone = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        img_size=size,
        dynamic_img_size=True,
        num_classes=0,
    )
    return semantic.FrozenHead(backbone, layers=4, channels=16)


@pytest.mark.parametrize("shape", [(45, 59), (56, 70), (29, 13)])
def test_frozen_head_output_matches_odd_input(shape):
    model = tiny_head().eval()
    with torch.no_grad():
        logits = semantic.upsampled_logits(model, torch.zeros(2, 3, *shape))
    assert logits.shape == (2, semantic.N_CLASSES, *shape)


def test_frozen_backbone_gets_no_gradient_and_stays_in_eval():
    model = tiny_head()
    model.train()
    assert model.training and model.blocks.training
    assert not model.backbone.training
    assert all(not module.training for module in model.backbone.modules())
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    head_before = model.classifier.weight.detach().clone()
    optimizer = semantic.make_optimizer(model, {"lr": 1e-2, "weight_decay": 0.01})
    trained = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not trained & {id(p) for p in model.backbone.parameters()}
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    images = torch.randn(2, 3, 42, 56)
    targets = torch.zeros(2, 42, 56, dtype=torch.long)
    targets[:, 10:20, 10:30] = 2
    for _ in range(2):
        semantic.train_step(model, optimizer, scaler, images, targets, {"dice_weight": 1.0}, False)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert not model.backbone.training
    after = model.backbone.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not torch.equal(head_before, model.classifier.weight)


def test_builders_keep_segformer_default():
    assert set(semantic.BUILDERS) == {"segformer_b0", "dinov2_s14"}
    assert semantic.BUILDERS["segformer_b0"] is semantic.build_segformer


def compare_case():
    targets, sessions = {}, {}
    for index in range(12):
        target = np.zeros((8, 8), dtype=np.uint8)
        target[1:4, 1:5] = 1 + index % 4
        if index % 5 == 0:
            target[:] = 0
        name = f"img{index}"
        targets[name] = target
        sessions[name] = index // 3
    return targets, sessions


def test_compare_known_answers():
    targets, sessions = compare_case()
    perfect = {n: t.copy() for n, t in targets.items()}
    blank = {n: np.zeros_like(t) for n, t in targets.items()}
    better = semantic.compare_maps(blank, perfect, targets, sessions, resamples=300)
    assert better["first"]["mean_dice"] == 0
    assert better["second"]["mean_dice"] == 1
    assert (better["difference"], better["low"], better["high"]) == (1, 1, 1)
    assert better["second_higher"]
    assert better["sessions"] == 4 and better["images"] == 12
    worse = semantic.compare_maps(perfect, blank, targets, sessions, resamples=300)
    assert worse["difference"] == -1 and not worse["second_higher"]
    same = semantic.compare_maps(perfect, perfect, targets, sessions, resamples=300)
    assert (same["difference"], same["low"], same["high"]) == (0, 0, 0)
    assert not same["second_higher"]


def test_compare_resamples_whole_sessions():
    targets, sessions = compare_case()
    perfect = {n: t.copy() for n, t in targets.items()}
    one_session = {
        n: t.copy() if sessions[n] == 0 else np.zeros_like(t) for n, t in targets.items()
    }
    result = semantic.compare_maps(one_session, perfect, targets, sessions, resamples=500)
    assert result["difference"] > 0
    assert result["low"] < result["difference"] < result["high"]
    again = semantic.compare_maps(one_session, perfect, targets, sessions, resamples=500)
    assert (again["low"], again["high"]) == (result["low"], result["high"])


def test_compare_refuses_mismatched_runs():
    targets, sessions = compare_case()
    perfect = {n: t.copy() for n, t in targets.items()}
    missing = dict(list(perfect.items())[:-1])
    with pytest.raises(ValueError):
        semantic.compare_maps(perfect, missing, targets, sessions)
    resized = {**perfect, "img0": np.zeros((4, 4), np.uint8)}
    with pytest.raises(ValueError):
        semantic.compare_maps(resized, perfect, targets, sessions)
