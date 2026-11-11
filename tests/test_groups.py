from itertools import product

import numpy as np
import pytest

from fabrid import groups


def make_groups(sizes: list[int]) -> dict[str, int]:
    group_of = {}
    for group, size in enumerate(sizes):
        for frame in range(size):
            group_of[f"g{group:02d}_{frame:03d}.jpg"] = group
    return group_of


def best_cost(sizes: list[int], shares) -> float:
    targets = np.array(shares) * sum(sizes)
    best = None
    for parts in product(range(3), repeat=len(sizes)):
        counts = np.zeros(3)
        for size, part in zip(sizes, parts, strict=True):
            counts[part] += size
        value = groups.cost(counts, targets)
        best = value if best is None else min(best, value)
    return best


def split_cost(mapping, shares) -> float:
    counts = np.array([sum(v == s for v in mapping.values()) for s in groups.SPLITS])
    return groups.cost(counts, np.array(shares) * len(mapping))


def test_no_group_is_split():
    rng = np.random.default_rng(1)
    group_of = make_groups(rng.integers(1, 30, 40).tolist())
    mapping = groups.grouped_split(group_of)
    assert set(mapping) == set(group_of)
    groups.check_groups(mapping, group_of)


@pytest.mark.parametrize("seed", range(6))
def test_shares_reach_best_whole_group_fit(seed):
    sizes = np.random.default_rng(seed).integers(1, 40, 7).tolist()
    mapping = groups.grouped_split(make_groups(sizes))
    assert split_cost(mapping, groups.SHARES) == pytest.approx(best_cost(sizes, groups.SHARES))


def test_achieved_shares_on_an_exact_fit():
    mapping = groups.grouped_split(make_groups([70, 15, 15]))
    assert groups.achieved_shares(mapping) == {"train": 0.7, "val": 0.15, "test": 0.15}
    other = groups.grouped_split(make_groups([5, 3, 2]), shares=(0.5, 0.3, 0.2))
    assert groups.achieved_shares(other) == {"train": 0.5, "val": 0.3, "test": 0.2}


def test_same_input_same_output():
    group_of = make_groups([12, 7, 7, 3, 9, 1, 20, 4])
    shuffled = dict(reversed(list(group_of.items())))
    assert groups.grouped_split(group_of) == groups.grouped_split(shuffled)
    named = {name: f"folder_{group}" for name, group in group_of.items()}
    assert list(groups.grouped_split(named).values()) == list(
        groups.grouped_split(group_of).values()
    )


def test_every_stratum_reaches_train():
    group_of = make_groups([30, 25, 20, 10, 8, 3, 2])
    strata = {0: "hole", 1: "hole", 2: "stain", 3: "stain", 4: "stain", 5: "yarn", 6: "needle"}
    split = groups.grouped_split(group_of, strata=strata)
    train_groups = {group_of[n] for n, s in split.items() if s == "train"}
    assert {strata[g] for g in train_groups} == set(strata.values())
    groups.check_groups(split, group_of)
    with pytest.raises(KeyError):
        groups.grouped_split(group_of, strata={0: "hole"})


def test_bad_shares_are_rejected():
    with pytest.raises(ValueError):
        groups.grouped_split(make_groups([3, 3]), shares=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        groups.grouped_split(make_groups([3, 3]), shares=(0.5, 0.5))


def test_check_groups_raises_on_a_split_group():
    group_of = {"a": 1, "b": 1, "c": 2}
    with pytest.raises(ValueError):
        groups.check_groups({"a": "train", "b": "val", "c": "test"}, group_of)
    with pytest.raises(KeyError):
        groups.check_groups({"a": "train", "b": "train"}, group_of)


def test_lock_raises_on_test_unknown_or_other_parts():
    locked = {"a": "train", "b": "val", "c": "test"}
    groups.check_view({"a": "train", "b": "val"}, locked)
    with pytest.raises(AssertionError):
        groups.check_view({"a": "train", "c": "val"}, locked)
    with pytest.raises(AssertionError):
        groups.check_view({"a": "train", "z": "val"}, locked)
    with pytest.raises(AssertionError):
        groups.check_view({"a": "test"}, {"a": "train"})


def test_random_arm_keeps_counts_and_never_touches_test():
    group_of = make_groups([30, 25, 20, 10, 8, 3, 2, 2])
    mapping = groups.grouped_split(group_of)
    grouped = groups.grouped_arm(mapping)
    random = groups.random_arm(mapping, seed=0)
    assert set(random) == set(grouped)
    assert sorted(random.values()) == sorted(grouped.values())
    assert not any(mapping[name] == "test" for name in random)
    assert random == groups.random_arm(mapping, seed=0)
    assert random != groups.random_arm(mapping, seed=1)
    assert random != grouped


def test_random_arm_matches_the_fabrid_rule():
    mapping = {f"n{i:02d}": s for i, s in enumerate(["train"] * 7 + ["val"] * 2 + ["test"] * 3)}
    pool = sorted(n for n, s in mapping.items() if s != "test")
    order = np.random.default_rng(5).permutation(len(pool))
    expected = {pool[i]: "train" if rank < 7 else "val" for rank, i in enumerate(order)}
    assert groups.random_arm(mapping, seed=5) == expected
