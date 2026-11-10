import numpy as np

from fabrid.labels import CLASSES

SPLITS = ("train", "val", "test")
TARGETS = (0.70, 0.15, 0.15)


def assign_sessions(records, gap_seconds: float = 30) -> dict[str, int]:
    ordered = sorted(records, key=lambda r: (r.timestamp, r.name))
    sessions = {}
    session_id = -1
    previous = None
    for record in ordered:
        new_day = previous is None or record.timestamp.date() != previous.timestamp.date()
        if new_day or (record.timestamp - previous.timestamp).total_seconds() > gap_seconds:
            session_id += 1
        sessions[record.name] = session_id
        previous = record
    return sessions


def record_vector(record) -> np.ndarray:
    vector = np.zeros(len(CLASSES) + 2)
    vector[0] = 1
    for class_id, _ in record.polygons:
        vector[1 + class_id] += 1
    vector[-1] = 1 if not record.polygons else 0
    return vector


def session_vectors(records, sessions: dict[str, int]) -> dict[int, np.ndarray]:
    vectors = {}
    for record in records:
        session_id = sessions[record.name]
        vectors[session_id] = vectors.get(session_id, 0) + record_vector(record)
    return vectors


def session_split(records, gap_seconds: float = 30) -> dict[str, str]:
    sessions = assign_sessions(records, gap_seconds)
    vectors = session_vectors(records, sessions)
    total = np.sum(list(vectors.values()), axis=0)
    active = total > 0
    scale = np.where(active, total, 1.0)
    targets = np.array(TARGETS)[:, None]
    sums = np.zeros((len(SPLITS), len(total)))
    chosen = {}
    for session_id in sorted(vectors, key=lambda s: (-vectors[s][0], s)):
        best_cost, best_split = None, None
        for index in range(len(SPLITS)):
            trial = sums.copy()
            trial[index] += vectors[session_id]
            cost = float(((trial / scale - targets)[:, active] ** 2).sum())
            if best_cost is None or cost < best_cost:
                best_cost, best_split = cost, index
        sums[best_split] += vectors[session_id]
        chosen[session_id] = SPLITS[best_split]
    return {name: chosen[session_id] for name, session_id in sessions.items()}


def random_split(records, seed: int = 0) -> dict[str, str]:
    names = sorted(r.name for r in records)
    order = np.random.default_rng(seed).permutation(len(names))
    n_train = round(len(names) * TARGETS[0])
    n_val = round(len(names) * TARGETS[1])
    mapping = {}
    for rank, index in enumerate(order):
        if rank < n_train:
            mapping[names[index]] = "train"
        elif rank < n_train + n_val:
            mapping[names[index]] = "val"
        else:
            mapping[names[index]] = "test"
    return mapping


def check_disjoint(mapping: dict[str, str], sessions: dict[str, int]) -> None:
    seen = {}
    for name, session_id in sessions.items():
        if name not in mapping:
            raise KeyError(f"{name} has a session but no split")
        split = mapping[name]
        first = seen.setdefault(session_id, split)
        if first != split:
            raise ValueError(f"session {session_id} spans {first} and {split}")
