"""Source-balanced episode sampling and replay behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from experiment.rl.sampling import BalancedReplay, BalancedSampler, Transition


FAMILIES = ("uniform", "normal", "correlated", "skewed")
ALPHAS = (0.25, 0.50, 0.75)


def _instance(source: str, n: int, alpha: float, family: str | None, index: int) -> dict:
    family_token = family or "real"
    return {
        "alpha": alpha,
        "family": family,
        "instance_id": f"{source}-{family_token}-n{n}-a{alpha}-i{index}",
        "n": n,
        "source": source,
    }


def _balanced_instances() -> list[dict]:
    instances: list[dict] = []
    for n in (20, 50):
        for family in FAMILIES:
            for alpha in ALPHAS:
                instances.append(_instance("synthetic", n, alpha, family, 0))
        for alpha in ALPHAS:
            instances.append(_instance("real", n, alpha, None, 0))
    return instances


def _transition(source: str, action: int) -> Transition:
    state = SimpleNamespace(instance={"source": source})
    return Transition(state=state, action=action, reward=float(action), next_state=None)


def test_sampler_blocks_are_seeded_and_balance_every_available_source_size_pair():
    left = BalancedSampler(_balanced_instances(), seed=41)
    right = BalancedSampler(_balanced_instances(), seed=41)

    left_block = left.next_block()
    right_block = right.next_block()

    assert [item["instance_id"] for item in left_block] == [
        item["instance_id"] for item in right_block
    ]
    assert {(item["source"], item["n"]) for item in left_block} == {
        ("synthetic", 20),
        ("real", 20),
        ("synthetic", 50),
        ("real", 50),
    }


def test_sampler_cycles_every_synthetic_family_alpha_cell_evenly():
    sampler = BalancedSampler(_balanced_instances(), seed=7)
    observed = []
    for _ in range(len(FAMILIES) * len(ALPHAS)):
        block = sampler.next_block()
        observed.extend(
            (item["family"], item["alpha"])
            for item in block
            if item["source"] == "synthetic" and item["n"] == 20
        )

    assert sorted(observed) == sorted(
        (family, alpha) for family in FAMILIES for alpha in ALPHAS
    )


def test_sampler_seed_changes_exposure_order_but_input_row_order_does_not():
    records = _balanced_instances()
    left = BalancedSampler(records, 2)
    reordered = BalancedSampler(list(reversed(records)), 2)
    different = BalancedSampler(records, 3)
    a, b, c = [], [], []
    for _ in range(12):
        a.extend(i["instance_id"] for i in left.next_block())
        b.extend(i["instance_id"] for i in reordered.next_block())
        c.extend(i["instance_id"] for i in different.next_block())
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    "instances",
    [
        [_instance("synthetic", 20, 0.25, "uniform", 0)],
        [
            _instance("synthetic", 20, 0.25, "uniform", 0),
            _instance("real", 50, 0.25, None, 0),
        ],
        [
            _instance("synthetic", 20, 0.25, "uniform", 0),
            {"instance_id": "bad", "n": 20, "source": "unknown"},
        ],
    ],
)
def test_sampler_rejects_unbalanced_or_invalid_inputs(instances):
    with pytest.raises(ValueError):
        BalancedSampler(instances, seed=1)


def test_transition_is_frozen():
    transition = _transition("synthetic", 1)
    with pytest.raises(FrozenInstanceError):
        transition.action = 0


def test_replay_keeps_a_bounded_pool_per_source_and_reports_readiness():
    replay = BalancedReplay(capacity=6, seed=3)
    for action in range(4):
        replay.add(_transition("synthetic", action))
    for action in range(2):
        replay.add(_transition("real", action))

    assert replay.counts == {"synthetic": 3, "real": 2}
    assert replay.ready({"synthetic": 3, "real": 2})
    assert not replay.ready({"synthetic": 3, "real": 3})


def test_replay_samples_exact_source_quotas_without_replacement_and_is_seeded():
    left = BalancedReplay(capacity=8, seed=19)
    right = BalancedReplay(capacity=8, seed=19)
    for replay in (left, right):
        for action in range(4):
            replay.add(_transition("synthetic", action))
            replay.add(_transition("real", action + 10))

    quotas = {"synthetic": 3, "real": 2}
    left_sample = left.sample(quotas)
    right_sample = right.sample(quotas)

    assert [item.action for item in left_sample] == [item.action for item in right_sample]
    assert [item.state.instance["source"] for item in left_sample].count("synthetic") == 3
    assert [item.state.instance["source"] for item in left_sample].count("real") == 2
    assert len({id(item) for item in left_sample}) == len(left_sample)


def test_replay_rejects_unknown_sources_and_underfilled_samples():
    replay = BalancedReplay(capacity=4, seed=5)
    with pytest.raises(ValueError):
        replay.add(_transition("other", 0))

    replay.add(_transition("synthetic", 0))
    with pytest.raises(ValueError):
        replay.sample({"synthetic": 2, "real": 0})


def test_replay_eviction_removes_oldest_transition_in_each_source():
    replay = BalancedReplay(capacity=6, seed=0)
    for action in range(5):
        replay.add(_transition("synthetic", action))
    replay.add(_transition("real", 20))
    sample = replay.sample({"synthetic": 3, "real": 1})
    assert sorted(t.action for t in sample) == [2, 3, 4, 20]


def test_sampler_cycles_only_available_cells_and_real_alphas():
    records = [i for i in _balanced_instances() if not
               (i["source"] == "synthetic" and i["n"] == 20 and
                i["family"] == "uniform" and i["alpha"] == 0.25)]
    sampler = BalancedSampler(records, seed=9)
    seen = []
    real_alphas = []
    for _ in range(11):
        for item in sampler.next_block():
            if item["n"] == 20 and item["source"] == "synthetic":
                seen.append((item["family"], item["alpha"]))
            if item["n"] == 20 and item["source"] == "real":
                real_alphas.append(item["alpha"])
    assert len(set(seen)) == 11
    assert ("uniform", 0.25) not in seen
    for start in (0, 3, 6):
        assert sorted(real_alphas[start:start + 3]) == list(ALPHAS)
