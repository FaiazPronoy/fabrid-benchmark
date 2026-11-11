from collections import Counter

import numpy as np

SPLITS = ("train", "val", "test")
SHARES = (0.70, 0.15, 0.15)


def group_order(group) -> tuple[str, str]:
    return str(group), type(group).__name__


def check_shares(shares) -> np.ndarray:
    values = np.asarray(shares, dtype=float)
    if values.shape != (len(SPLITS),) or (values < 0).any() or abs(values.sum() - 1) > 1e-6:
        raise ValueError(f"shares must be three non-negative numbers summing to 1, got {shares}")
    return values


def cost(counts: np.ndarray, targets: np.ndarray) -> float:
    return float(((counts - targets) ** 2).sum())


def reserved(order: list, strata: dict | None) -> list:
    if strata is None:
        return []
    missing = sorted(group_order(g) for g in order if g not in strata)
    if missing:
        raise KeyError(f"groups without a stratum: {missing[:5]}")
    first = {}
    for group in order:
        first.setdefault(strata[group], group)
    return [first[stratum] for stratum in sorted(first, key=group_order)]


def reachable(sizes: list[int], base: int, total: int) -> tuple[np.ndarray, np.ndarray]:
    reach = np.zeros((total + 1, total + 1), dtype=bool)
    reach[base, 0] = True
    parent = np.full(reach.shape, -1, dtype=np.int32)
    for step, size in enumerate(sizes):
        to_train = np.zeros_like(reach)
        to_train[size:, :] = reach[:-size, :]
        to_val = np.zeros_like(reach)
        to_val[:, size:] = reach[:, :-size]
        parent[to_train & ~reach] = 2 * step
        parent[to_val & ~to_train & ~reach] = 2 * step + 1
        reach |= to_train | to_val
    return reach, parent


def grouped_split(group_of: dict, shares=SHARES, strata: dict | None = None) -> dict[str, str]:
    targets = check_shares(shares) * len(group_of)
    sizes = Counter(group_of.values())
    order = sorted(sizes, key=lambda g: (-sizes[g], group_order(g)))
    fixed = reserved(order, strata)
    free = [group for group in order if group not in fixed]
    total = len(group_of)
    base = sum(sizes[group] for group in fixed)
    reach, parent = reachable([sizes[group] for group in free], base, total)
    train, val = np.nonzero(reach)
    test = total - train - val
    keep = test >= 0
    train, val, test = train[keep], val[keep], test[keep]
    costs = (train - targets[0]) ** 2 + (val - targets[1]) ** 2 + (test - targets[2]) ** 2
    best = np.lexsort((-val, -train, costs))[0]
    count = [int(train[best]), int(val[best])]
    part = dict.fromkeys(fixed, 0)
    while parent[count[0], count[1]] >= 0:
        step, index = divmod(int(parent[count[0], count[1]]), 2)
        part[free[step]] = index
        count[index] -= sizes[free[step]]
    for group in free:
        part.setdefault(group, 2)
    return {name: SPLITS[part[group]] for name, group in sorted(group_of.items())}


def achieved_shares(mapping: dict[str, str]) -> dict[str, float]:
    counts = Counter(mapping.values())
    total = max(1, len(mapping))
    return {split: counts.get(split, 0) / total for split in SPLITS}


def grouped_arm(mapping: dict[str, str]) -> dict[str, str]:
    return {name: split for name, split in sorted(mapping.items()) if split != "test"}


def random_arm(mapping: dict[str, str], seed: int = 0) -> dict[str, str]:
    pool = sorted(name for name, split in mapping.items() if split != "test")
    n_train = sum(mapping[name] == "train" for name in pool)
    order = np.random.default_rng(seed).permutation(len(pool))
    view = {pool[index]: "train" if rank < n_train else "val" for rank, index in enumerate(order)}
    check_view(view, mapping)
    return view


def check_groups(mapping: dict[str, str], group_of: dict) -> None:
    seen = {}
    for name, group in group_of.items():
        if name not in mapping:
            raise KeyError(f"{name} has a group but no split")
        first = seen.setdefault(group, mapping[name])
        if first != mapping[name]:
            raise ValueError(f"group {group} spans {first} and {mapping[name]}")


def check_view(view: dict[str, str], locked: dict[str, str]) -> None:
    unknown = sorted(name for name in view if name not in locked)
    if unknown:
        raise AssertionError(f"view holds images outside the split: {unknown[:5]}")
    leaked = sorted(name for name in view if locked[name] == "test")
    if leaked:
        raise AssertionError(f"locked test images reached a training view: {leaked[:5]}")
    parts = set(view.values()) - {"train", "val"}
    if parts:
        raise AssertionError(f"view may only hold train and val, found {sorted(parts)}")
