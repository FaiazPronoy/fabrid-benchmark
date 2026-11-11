import argparse
import csv
from pathlib import Path

import numpy as np

from fabrid import runio
from fabrid.groups import check_groups, check_view, group_order
from fabrid.labels import CLASSES, load_index
from fabrid.sessions import assign_sessions, record_vector, session_split
from fabrid.yolo import write_csv

FOLDS = 5
STEP = "crossval"
FOLDS_FILE = Path("results") / STEP / "folds" / "folds.csv"


def pool_names(locked: dict[str, str]) -> list[str]:
    return sorted(name for name, part in locked.items() if part in ("train", "val"))


def assign_folds(records, gap_seconds: float = 30, n_folds: int = FOLDS) -> dict[str, int]:
    locked = session_split(records, gap_seconds)
    sessions = assign_sessions(records, gap_seconds)
    pool = [record for record in records if locked[record.name] in ("train", "val")]
    vectors = {}
    for record in pool:
        session = sessions[record.name]
        vectors[session] = vectors.get(session, 0) + record_vector(record)
    total = np.sum(list(vectors.values()), axis=0)
    active = total > 0
    scale = np.where(active, total, 1.0)
    span = slice(1, 1 + len(CLASSES))
    holders = np.sum([vector[span] > 0 for vector in vectors.values()], axis=0)
    sums = np.zeros((n_folds, len(total)))
    held = np.zeros((n_folds, len(CLASSES)), dtype=int)
    chosen = {}
    for session in sorted(vectors, key=lambda s: (-vectors[s][0], group_order(s))):
        present = vectors[session][span] > 0
        best_cost, best_fold = None, None
        for fold in range(n_folds):
            if (present & (holders >= 2) & (held[fold] + 1 == holders)).any():
                continue
            trial = sums.copy()
            trial[fold] += vectors[session]
            cost = float(((trial / scale - 1 / n_folds)[:, active] ** 2).sum())
            if best_cost is None or cost < best_cost:
                best_cost, best_fold = cost, fold
        sums[best_fold] += vectors[session]
        held[best_fold] += present
        chosen[session] = best_fold
    folds = {record.name: chosen[sessions[record.name]] for record in pool}
    check_groups(folds, {name: sessions[name] for name in folds})
    return dict(sorted(folds.items()))


def check_fold_view(view: dict[str, str], locked: dict[str, str], sessions: dict) -> None:
    check_view(view, locked)
    check_groups(view, {name: sessions[name] for name in view})


def fold_view(folds: dict[str, int], fold: int, locked: dict[str, str], sessions: dict):
    if fold not in set(folds.values()):
        raise ValueError(f"fold {fold} is not one of {sorted(set(folds.values()))}")
    view = {name: "val" if index == fold else "train" for name, index in folds.items()}
    check_fold_view(view, locked, sessions)
    return view


def read_folds(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: int(row["fold"]) for row in csv.DictReader(handle)}


def fold_mapping(records, gap_seconds: float, fold: int) -> dict[str, str]:
    if not 0 <= fold < FOLDS:
        raise ValueError(f"fold must be 0 to {FOLDS - 1}, got {fold}")
    folds = assign_folds(records, gap_seconds)
    committed = read_folds(runio.repo_root() / FOLDS_FILE)
    if committed is not None and committed != folds:
        raise ValueError(f"fold assignment differs from the committed {FOLDS_FILE}")
    locked = session_split(records, gap_seconds)
    return fold_view(folds, fold, locked, assign_sessions(records, gap_seconds))


def fold_table(records, folds: dict[str, int], sessions: dict) -> list[list]:
    by_name = {record.name: record for record in records}
    rows = []
    for fold in sorted(set(folds.values())):
        val = [by_name[name] for name, index in folds.items() if index == fold]
        train = [by_name[name] for name, index in folds.items() if index != fold]
        counts = np.sum([record_vector(record) for record in val], axis=0)
        seen = {class_id for record in train for class_id, _ in record.polygons}
        missing = [name for class_id, name in enumerate(CLASSES) if class_id not in seen]
        rows.append(
            [
                fold,
                len({sessions[record.name] for record in val}),
                len(val),
                int(counts[-1]),
                *[int(v) for v in counts[1 : 1 + len(CLASSES)]],
                len(train),
                ";".join(missing),
            ]
        )
    return rows


SIZE_HEADER = [
    "fold",
    "sessions",
    "images",
    "empty_images",
    *[name.replace(" ", "_") for name in CLASSES],
    "train_images",
    "train_missing_classes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = runio.load_config(parser.parse_args().config)
    gap = config["gap_seconds"]
    records = load_index()
    print(f"indexed {len(records)} images from {records[0].path.parents[2]}")
    folds = assign_folds(records, gap)
    sessions = assign_sessions(records, gap)
    locked = session_split(records, gap)
    for fold in range(FOLDS):
        fold_view(folds, fold, locked, sessions)
    rows = fold_table(records, folds, sessions)
    run_dir = runio.start_run(config["step"], config["name"], config)
    write_csv(
        run_dir / "folds.csv",
        ["name", "session", "fold"],
        [[name, sessions[name], fold] for name, fold in folds.items()],
    )
    write_csv(run_dir / "sizes.csv", SIZE_HEADER, rows)
    print(
        f"{len(folds)} pool images in {len({sessions[n] for n in folds})} sessions, test left out"
    )
    for row in rows:
        print(
            f"fold {row[0]}: {row[1]} sessions, {row[2]} images, instances "
            f"{dict(zip(CLASSES, row[4:8], strict=True))}, training side lacks "
            f"{row[-1] or 'no class'}"
        )
    print(f"wrote {run_dir}")
    if config.get("publish", True):
        runio.publish(run_dir)


if __name__ == "__main__":
    main()
