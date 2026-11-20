import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import SQUARE, make_dataset, write_image
from fabrid import evaluate, needles, pilot, pilotarm, pilotscore
from fabrid.labels import CLASSES, is_valid, load_index

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPORT_NAMES = ["hole", "missing yarn", "needle mark", "stain"]
CONFIG = {
    "needle_link": 0.05,
    "needle_min_width": 0.006,
    "val_share": 0.15,
    "val_block": 10,
    "split_seed": 0,
    "gap_seconds": 30,
}
SCORING = {"iou": 0.5, "tolerance_px": 2, "cldice_tol_px": 1}
SQUARE_POINTS = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])


def dot(x: float, y: float, r: float = 0.004) -> np.ndarray:
    return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]])


def line(class_id: int, points: np.ndarray) -> str:
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in points.ravel())


def fake_export(base: Path, n_train: int = 40, n_test: int = 8, n_excluded: int = 2):
    root = base / "fabrid-new"
    needle = EXPORT_NAMES.index("needle mark")
    rows, names = [], []
    total = n_train + n_test + n_excluded
    for index in range(total):
        folder = ("train", "valid", "test")[index % 3]
        stem = f"IMG_{4000 + index}-Custom-_JPG" if index % 7 else f"Copy-of-F{index:03d}_clean"
        name = f"{stem}.rf.{index:04x}.jpg"
        lines = [line(needle, dot(0.30, 0.50)), line(needle, dot(0.33, 0.50))]
        lines += [line(needle, dot(0.90, 0.10)), "0 0.6 0.6 0.7 0.6 0.7 0.7"]
        (root / folder / "images").mkdir(parents=True, exist_ok=True)
        (root / folder / "labels").mkdir(parents=True, exist_ok=True)
        write_image(root / folder / "images" / name, index, size=64)
        (root / folder / "labels" / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        if index < n_train:
            split, fabric = "train", ""
        elif index < n_train + n_test:
            split, fabric = "test", f"fabric_{index % 2}"
        else:
            split, fabric = "excluded", ""
        rows.append([name, split, fabric])
        names.append(name)
    data = {"train": "../train/images", "val": "../valid/images", "nc": 4, "names": EXPORT_NAMES}
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    fabrics = base / "fabrics.csv"
    with fabrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split", "test_fabric"])
        writer.writerows(rows)
    return root, fabrics


def fake_old(base: Path) -> Path:
    items = []
    for day in range(1, 8):
        for frame in range(3):
            name = f"IMG_202605{day:02d}_09{frame:02d}00_jpg.rf.v{day}{frame}.jpg"
            items.append(("train", name, [SQUARE] if frame else [], day * 10 + frame))
    return make_dataset(base / "fabrid-v9", items)


def test_class_names_map_to_old_order():
    assert pilot.class_map(EXPORT_NAMES) == {0: 1, 1: 2, 2: 0, 3: 3}
    old = ["Needle Mark Defect", "hole Defect", "missing yarn Defect", "stain Defect"]
    assert pilot.class_map(old) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert pilot.class_map(dict(enumerate(CLASSES))) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert pilot.class_index("  Missing  Yarn DEFECT ") == 2
    with pytest.raises(ValueError, match="unknown class"):
        pilot.class_map(["hole", "missing yarn", "needle mark", "crease"])
    with pytest.raises(ValueError, match="same class"):
        pilot.class_map(["hole", "Hole Defect"])
    with pytest.raises(ValueError):
        pilotarm.class_lookup(["hole", "stain"])


def test_two_collinear_dots_merge():
    polygons = [(0, dot(0.30, 0.50)), (0, dot(0.33, 0.50)), (1, SQUARE_POINTS)]
    merged, found = needles.merge_needles(polygons, 1000, 1000, 0.05, 0.006)
    assert len(merged) == 2 and len(found) == 1
    assert [c for c, _ in merged] == [0, 1]
    rectangle = merged[0][1]
    assert rectangle.shape == (4, 2)
    assert is_valid(rectangle)
    assert ((rectangle >= 0) & (rectangle <= 1)).all()
    low, high = rectangle.min(axis=0), rectangle.max(axis=0)
    assert low[0] == pytest.approx(0.296) and high[0] == pytest.approx(0.334)
    assert high[1] - low[1] >= 0.006 * np.sqrt(2) - 1e-9
    assert merged[1][1] is SQUARE_POINTS


def test_far_dots_stay_apart_and_singleton_unchanged():
    first, second = dot(0.10, 0.10), dot(0.90, 0.90)
    merged, found = needles.merge_needles([(0, first), (0, second)], 640, 480, 0.05, 0.006)
    assert found == []
    assert merged[0][1] is first and merged[1][1] is second
    single, found = needles.merge_needles([(0, first)], 640, 480, 0.05, 0.006)
    assert found == [] and single[0][1] is first


def test_merge_is_single_linkage_and_widens_a_line():
    chain = [(0, dot(0.10 + 0.03 * k, 0.5, 0.001)) for k in range(5)]
    merged, found = needles.merge_needles(chain, 1000, 1000, 0.05, 0.006)
    assert len(merged) == 1 and len(found[0].members) == 5
    rectangle = merged[0][1]
    sides = np.linalg.norm(np.roll(rectangle, -1, axis=0) - rectangle, axis=1)
    assert sides.min() == pytest.approx(0.006 * np.sqrt(2))
    edge = np.array([[0.0, 0.0], [0.004, 0.0], [0.004, 0.004]])
    tilted = [(0, edge + [0.995, 0.5]), (0, edge + [0.999, 0.501])]
    clipped, _ = needles.merge_needles(tilted, 100, 100, 0.05, 0.05)
    assert ((clipped[0][1] >= 0) & (clipped[0][1] <= 1)).all()


def test_polygon_distance():
    assert needles.polygon_distance(dot(0.5, 0.5, 0.1), dot(0.5, 0.5, 0.01)) == 0.0
    assert needles.polygon_distance(dot(0, 0, 1), dot(3, 0, 1)) == pytest.approx(1.0)
    bar = np.array([[-2, -0.1], [2, -0.1], [2, 0.1], [-2, 0.1]])
    assert needles.polygon_distance(bar, bar[:, ::-1]) == 0.0


def test_fabric_list_must_cover_the_export(tmp_path):
    root, fabrics_path = fake_export(tmp_path)
    fabrics = pilot.read_fabrics(fabrics_path)
    files = pilot.export_files(pilot.data_root(root))
    pilot.check_coverage(files, fabrics)
    with pytest.raises(ValueError, match="1 export images not listed"):
        pilot.check_coverage(files, dict(list(fabrics.items())[1:]))
    with pytest.raises(ValueError, match="1 listed images not in the export"):
        pilot.check_coverage(files, {**fabrics, "IMG_1.jpg": ("train", "")})
    bad = tmp_path / "bad.csv"
    bad.write_text("image,split,test_fabric\na.jpg,test,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no fabric"):
        pilot.read_fabrics(bad)
    bad.write_text("image,split,test_fabric\na.jpg,holdout,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        pilot.read_fabrics(bad)


def test_validation_blocks_are_deterministic():
    names = [f"Copy-of-IMG_{n}-Custom-_JPG.rf.a.jpg" for n in range(3000, 3095)]
    names += ["F001_clean.jpg", "F002_clean.jpg"]
    first = pilot.validation_blocks(names, 0.15, 10, 0)
    assert first == pilot.validation_blocks(list(reversed(names)), 0.15, 10, 0)
    assert set(first) == set(names)
    ordered = sorted(names, key=pilot.photo_key)
    assert ordered[-2:] == ["F001_clean.jpg", "F002_clean.jpg"]
    assert ordered[0].startswith("Copy-of-IMG_3000")
    blocks = [ordered[i : i + 10] for i in range(0, len(ordered), 10)]
    assert all(len({first[n] for n in block}) == 1 for block in blocks)
    assert sum(first[block[0]] == "val" for block in blocks) == 2
    assert first != pilot.validation_blocks(names, 0.15, 10, 1)


def test_build_keeps_test_and_excluded_out(tmp_path, monkeypatch):
    root, fabrics_path = fake_export(tmp_path)
    old_root = fake_old(tmp_path)
    fabrics = pilot.read_fabrics(fabrics_path)
    photos = pilot.load_new(pilot.data_root(root), fabrics, CONFIG)
    photos += pilot.load_old(old_root, CONFIG)
    blocked = pilot.blocked_names(fabrics)
    built = pilot.build(photos, CONFIG, tmp_path / "views", pilot.DATASETS, blocked)
    test_names = {n for n, (split, _) in fabrics.items() if split == "test"}
    excluded = {n for n, (split, _) in fabrics.items() if split == "excluded"}
    for dataset in ("pilot_new", "pilot_merged"):
        folder = built[dataset].parent
        held = pilot.folder_names(folder, "train") | pilot.folder_names(folder, "val")
        assert not held & (test_names | excluded)
        data = yaml.safe_load(built[dataset].read_text(encoding="utf-8"))
        assert "test" not in data and data["names"] == dict(enumerate(CLASSES))
    assert pilot.folder_names(built["pilot_test"].parent, "test") == test_names
    old_records = load_index(old_root)
    locked = {r.name for r in old_records} - {p.name for p in photos if p.source == "old"}
    merged = built["pilot_merged"].parent
    assert not locked & (pilot.folder_names(merged, "train") | pilot.folder_names(merged, "val"))
    victim = sorted(test_names)[0]
    leak = merged / "train" / "images" / victim
    leak.write_bytes(b"0")
    with pytest.raises(AssertionError, match="blocked"):
        pilot.check_built(merged, blocked)
    label = next((built["pilot_test"].parent / "test" / "labels").iterdir())
    lines = label.read_text(encoding="utf-8").splitlines()
    assert [int(text.split()[0]) for text in lines] == [0, 0, 1]
    summary = pilot.counts(photos, CONFIG)
    assert summary["pilot_test"] == 8 and summary["training_pool"] == 40
    assert summary["needle"]["polygons_before"] == 3 * 48
    assert summary["needle"]["polygons_after"] == 2 * 48
    assert summary["pilot_merged"]["new"] == summary["pilot_new"]["new"]
    sheet = tmp_path / "needles.jpg"
    assert pilot.contact_sheet(photos, sheet, 24) == 24 and sheet.is_file()


def test_new_and_old_names_must_not_clash(tmp_path):
    root, fabrics_path = fake_export(tmp_path)
    fabrics = pilot.read_fabrics(fabrics_path)
    photos = pilot.load_new(pilot.data_root(root), fabrics, CONFIG)
    twin = pilot.Photo(photos[0].name, photos[0].path, [], "train", source="old")
    with pytest.raises(ValueError, match="share a file name"):
        pilot.build([*photos, twin], CONFIG, tmp_path / "views", ("pilot_new",), set())


def test_input_dir_finds_nested_slug(tmp_path):
    (tmp_path / "datasets" / "owner" / "fabrid-new" / "train" / "images").mkdir(parents=True)
    found = pilot.input_dir("fabrid-new", tmp_path)
    assert found == tmp_path / "datasets" / "owner" / "fabrid-new"
    with pytest.raises(FileNotFoundError):
        pilot.input_dir("fabrid-weights", tmp_path)


def test_mask_encoding_round_trips():
    rng = np.random.default_rng(4)
    for shape in ((32, 32), (19, 45)):
        mask = rng.random(shape) > 0.5
        decoded = evaluate.decode({"size": list(shape), "counts": pilotarm.encode(mask)}, *shape)
        assert (decoded == mask).all()


def perfect(photos, score=0.9) -> list[dict]:
    predictions = []
    for photo in photos:
        instances, _ = evaluate.ground_truth(photo, 64, 64)
        for instance in instances:
            top, bottom, left, right = instance.box
            box = (left, top, right, bottom)
            predictions.append(
                pilotarm.entry(photo.name, instance.class_id, score, box, instance.mask)
            )
    return predictions


def test_arms_scored_and_compared(tmp_path, monkeypatch):
    root, fabrics_path = fake_export(tmp_path)
    fabrics = pilot.read_fabrics(fabrics_path)
    photos = pilot.load_new(pilot.data_root(root), fabrics, CONFIG)
    test = [p for p in photos if p.split == "test"]
    good = perfect(test)
    with pytest.raises(AssertionError, match="outside the pilot test"):
        pilotarm.score(test, perfect([p for p in photos if p.split == "train"][:1]), SCORING)
    monkeypatch.setenv("FABRID_REPO", str(tmp_path))
    arms = {"pilot_old": good[::2], "pilot_new": good}
    for arm, predictions in arms.items():
        names, mask, box, _ = pilotarm.score(test, predictions, SCORING)
        folder = tmp_path / "results" / "pilot" / arm
        (folder / "box").mkdir(parents=True)
        evaluate.save_scored(folder, mask, names)
        evaluate.save_scored(folder / "box", box, names)
        summary = {"mask_ap50": pilotarm.point(mask)}
        (folder / "metrics.json").write_text(json.dumps(summary), encoding="utf-8")
    test_names = sorted(p.name for p in test)
    found = {arm: pilotscore.load_arm(arm, [tmp_path], test_names) for arm in arms}
    assert found["pilot_new"]["metrics"]["mask_ap50"]["mean"] == 1.0
    weights = evaluate.draws(np.arange(len(test_names)), 200, 0, "pilot")
    entry = {"name": "new_vs_old", "baseline": "pilot_old", "candidate": "pilot_new"}
    result = pilotscore.comparison(entry, found, weights)
    assert result["mask_ap50"]["mean"]["difference"] > 0
    assert result["box_ap50"]["mean"]["candidate"] == 1.0
    assert result["verdict"] in (pilotscore.BETTER, pilotscore.NOT_SHOWN)
    fabric_of = np.array([fabrics[n][1] for n in test_names])
    scorer = found["pilot_new"]["scorers"]["mask"]
    rows = pilotscore.fabric_rows("pilot_new", scorer, fabric_of, weights)
    assert [row[1] for row in rows] == ["fabric_0", "fabric_1"]
    assert all(row[4] == 1.0 for row in rows)
    with pytest.raises(ValueError, match="differ from the pilot test"):
        pilotscore.load_arm("pilot_new", [tmp_path], test_names[1:])


def test_pass_rule_needs_lower_end_above_zero():
    assert pilotscore.pass_verdict(0.01) == pilotscore.BETTER
    assert pilotscore.pass_verdict(0.0) == pilotscore.NOT_SHOWN
    assert pilotscore.pass_verdict(-0.2) == pilotscore.NOT_SHOWN
    assert pilotscore.pass_verdict(None) == pilotscore.NOT_SHOWN


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_pilot_configs_share_build_and_inference_settings():
    base = load("yolov8m_seg_aug.yaml")
    arms = {name: load(f"pilot/{name}.yaml") for name in ("old", "new", "merged")}
    prep = load("pilot/prep.yaml")
    for key in CONFIG:
        assert len({str(prep[key]), *(str(arm[key]) for arm in arms.values())}) == 1, key
    assert arms["old"]["predict"] == arms["new"]["predict"] == arms["merged"]["predict"]
    trained = ("model", "weights", "seed", "imgsz", "batch", "epochs", "patience", "train_args")
    for name in ("new", "merged"):
        assert {k: arms[name][k] for k in trained} == {k: base[k] for k in trained}
        assert arms[name]["train_on"] == f"pilot_{name}"
    assert "train_on" not in arms["old"]
    assert load("pilot/evaluate.yaml")["resamples"] == 2000
