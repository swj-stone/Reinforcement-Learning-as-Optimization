"""Pooling, masked behavior and optimized inference agree on real policies."""

import random

import pytest
import torch

from experiment.baselines.methods import BaselineSolution
from experiment.rl.environment import KnapsackEnv, State, batch_states
from experiment.rl.model import QNetwork, select_action, solve_instances


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _instance(weights, values, capacity, order=None):
    return {
        "instance_id": "model-fixture", "source": "synthetic", "family": "uniform",
        "alpha": 0.5, "n": len(weights), "weights": weights, "values": values,
        "capacity": capacity,
        "item_order": list(range(len(weights))) if order is None else order,
    }


def _constant_network(skip, include):
    network = QNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.head[-1].bias.copy_(torch.tensor([skip, include]))
    return network


def _ordinary_solution(network, instance):
    env = KnapsackEnv(instance)
    while env.state is not None:
        env.step(select_action(network, env.state, 0.0, random.Random(0)))
    selected = tuple(env.selected_indices)
    return BaselineSolution(selected, sum(instance["values"][i] for i in selected),
                            sum(instance["weights"][i] for i in selected))


def test_padding_cannot_change_either_mean_or_max_pooling():
    network = _constant_network(0, 0)
    # Both active channels equal normalized weight + 1. Padding has a nonzero
    # embedding, so merely supplying zero padding does not make the mean valid.
    with torch.no_grad():
        network.encoder[0].weight[0, 0] = 1
        network.encoder[0].bias[0] = 1
        network.encoder[2].weight[0, 0] = 1
        network.head[0].weight[0, 0] = 1
        network.head[0].weight[1, 64] = 1
        network.head[-1].weight[0, 0] = 1
        network.head[-1].weight[1, 1] = 1
    small = State(_instance([2, 4], [1, 3], 4), 0, 4)
    long = State(_instance([1] * 500, [1] * 500, 500), 0, 500)
    single = network(batch_states([small]))
    batch = batch_states([small, long])
    batch["items"][0, 2:, :] = 1000  # invalid pixels must also be excluded from max
    combined = network(batch)
    torch.testing.assert_close(single, torch.tensor([[1.75, 2.0]]))
    torch.testing.assert_close(combined[:1], single)


def test_processed_items_and_metadata_cannot_change_q_values():
    torch.manual_seed(17)
    network = QNetwork()
    instance = _instance([999, 2, 3], [999, 5, 6], 5)
    state = State(instance, 1, 5)
    observation = batch_states([state])
    altered = {**instance, "source": "real", "instance_id": "new", "family": "other"}
    torch.testing.assert_close(network(observation), network(batch_states([State(altered, 1, 5)])))
    assert observation["items"].shape == (1, 2, 2)


def test_exploration_and_greedy_inference_mask_infeasible_include():
    network = _constant_network(-100, 100)
    state = State(_instance([6], [100], 5), 0, 5)
    for epsilon in (0.0, 0.5, 1.0):
        rng = random.Random(12)
        assert {select_action(network, state, epsilon, rng) for _ in range(100)} == {0}


def test_exploration_can_choose_both_valid_actions_and_q_ties_skip():
    network = _constant_network(2, 2)
    state = State(_instance([5], [100], 5), 0, 5)
    assert select_action(network, state, 0.0, random.Random(10)) == 0
    rng = random.Random(10)
    assert {select_action(network, state, 1.0, rng) for _ in range(100)} == {0, 1}
    assert solve_instances(network, [state.instance]) == [BaselineSolution((), 0, 0)]


def test_solution_preserves_original_indices_exact_capacity_and_integer_totals():
    network = _constant_network(0, 1)
    instance = _instance([9, 4, 2], [18, 8, 4], 6, [2, 0, 1])
    zero = _instance([2, 1], [10, 20], 0, [1, 0])
    assert solve_instances(network, [instance, zero]) == [
        BaselineSolution((1, 2), 12, 6), BaselineSolution((), 0, 0),
    ]
    assert solve_instances(network, []) == []


def test_chunking_keeps_every_row_aligned_with_its_own_capacity_and_input_position():
    network = _constant_network(0, 1)
    instances = [_instance([2, 1], [7, 11], i % 3, [1, 0]) for i in range(130)]
    instances.insert(1, _instance([1], [99], 1))
    expected = [
        BaselineSolution((), 0, 0) if i % 3 == 0 else BaselineSolution((1,), 11, 1)
        for i in range(130)
    ]
    expected.insert(1, BaselineSolution((0,), 99, 1))
    assert solve_instances(network, instances) == expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_inference_rejects_nonfinite_q_values(invalid):
    network = _constant_network(0, invalid)
    instance = _instance([1], [2], 1)
    with pytest.raises(ValueError, match="finite"):
        select_action(network, State(instance, 0, 1), 0, random.Random(0))
    with pytest.raises(ValueError, match="finite"):
        solve_instances(network, [instance])


@pytest.mark.parametrize("seed", [0, 7, 41])
def test_cached_inference_equals_ordinary_policy_for_mixed_sizes(seed):
    torch.manual_seed(seed)
    network = QNetwork()
    rng = random.Random(31)
    instances = []
    for n in (20, 50, 20, 100, 200, 500, 50):
        weights = [rng.randint(1, 20) for _ in range(n)]
        values = [rng.randint(1, 50) for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        instances.append(_instance(weights, values, sum(weights) // 3, order))
    expected = [_ordinary_solution(network, instance) for instance in instances]
    assert solve_instances(network, instances) == expected
    # A cache surviving across calls would become stale after a weight update.
    with torch.no_grad():
        network.encoder[0].bias.add_(1)
        network.head[-1].bias[1].add_(0.2)
    assert solve_instances(network, instances) == [
        _ordinary_solution(network, instance) for instance in instances
    ]


def test_cached_q_values_match_direct_remaining_item_pooling():
    torch.manual_seed(31)
    network = QNetwork()
    instance = _instance([2, 9, 1, 3], [5, 7, 11, 13], 6, [2, 0, 3, 1])
    longer = _instance([1] * 20, [2] * 20, 10)
    initial = batch_states([State(instance, 0, 6), State(longer, 0, 10)])
    with torch.no_grad():
        means, maxima = network.suffix_pools(initial)
        for position, capacity in enumerate([6, 5, 3, 0]):
            batch = batch_states([State(instance, position, capacity)])
            cached = network.head(torch.cat((means[:1, position], maxima[:1, position], batch["context"]), dim=1))
            torch.testing.assert_close(cached, network(batch))
