"""Sequential RL observations retain integer feasibility and original indices."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from experiment.rl.environment import KnapsackEnv, State, batch_states


def _instance(weights, values, capacity, order=None):
    return {
        "instance_id": "environment-fixture",
        "source": "synthetic",
        "family": "uniform",
        "alpha": 0.5,
        "n": len(weights),
        "weights": weights,
        "values": values,
        "capacity": capacity,
        "item_order": list(range(len(weights))) if order is None else order,
    }


def test_episode_uses_saved_order_and_original_reward_denominator():
    instance = _instance([4, 9, 2], [8, 18, 4], 6, [2, 1, 0])
    env = KnapsackEnv(instance)
    first = env.state
    assert first.instance is instance
    with pytest.raises(FrozenInstanceError):
        first.position = 2

    state, reward, done = env.step(1)
    assert (state.position, state.remaining_capacity, reward, done) == (1, 4, 4 / 30, False)
    assert (first.position, first.remaining_capacity) == (0, 6)
    assert tuple(env.selected_indices) == (2,)
    state, reward, done = env.step(0)
    assert (state.position, state.remaining_capacity, reward, done) == (2, 4, 0.0, False)
    state, reward, done = env.step(1)
    assert (state, reward, done) == (None, 8 / 30, True)
    assert env.state is None
    assert tuple(env.selected_indices) == (0, 2)
    with pytest.raises(ValueError):
        env.step(0)


def test_invalid_include_and_unknown_action_do_not_advance_episode():
    env = KnapsackEnv(_instance([7, 2], [14, 4], 6))
    before = env.state
    for action in (1, -1, 2, "include", 0.5):
        with pytest.raises(ValueError):
            env.step(action)
        assert env.state is before
        assert tuple(env.selected_indices) == ()


def test_remaining_items_context_and_padding_are_in_saved_order():
    instance = _instance([4, 9, 2], [8, 18, 4], 6, [2, 1, 0])
    batch = batch_states([State(instance, 1, 4), State(instance, 2, 4)])
    torch.testing.assert_close(
        batch["items"],
        torch.tensor([[[9 / 6, 18 / 30], [4 / 6, 8 / 30]], [[4 / 6, 8 / 30], [0, 0]]]),
    )
    assert batch["valid"].dtype == torch.bool
    assert batch["valid"].tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(
        batch["context"],
        torch.tensor([[9 / 6, 18 / 30, 4 / 6, 2 / 3, math.log(4)],
                      [4 / 6, 8 / 30, 4 / 6, 1 / 3, math.log(4)]]),
    )
    assert batch["action_mask"].dtype == torch.bool
    assert batch["action_mask"].tolist() == [[True, False], [True, True]]


def test_zero_capacity_and_zero_value_denominators_are_finite():
    instance = _instance([2, 5], [0, 0], 0)
    env = KnapsackEnv(instance)
    batch = batch_states([env.state])
    assert torch.isfinite(batch["items"]).all()
    assert torch.isfinite(batch["context"]).all()
    assert batch["items"].tolist() == [[[2, 0], [5, 0]]]
    assert batch["action_mask"].tolist() == [[True, False]]
    assert env.step(0)[1:] == (0.0, False)
    assert env.step(0) == (None, 0.0, True)


def test_integer_feasibility_does_not_round_near_large_weights():
    # float32 rounds both weights to the same number; feasibility must not.
    instance = _instance([2**25 + 1, 2**25], [1, 1], 2**25)
    batch = batch_states([State(instance, 0, 2**25), State(instance, 1, 2**25)])
    assert batch["action_mask"].tolist() == [[True, False], [True, True]]


@pytest.mark.parametrize("n", [20, 50, 100, 200, 500])
def test_episode_takes_exactly_n_steps_even_after_capacity_is_exhausted(n):
    env = KnapsackEnv(_instance([1] * n, [2] * n, 1, list(reversed(range(n)))))
    rewards = []
    for step in range(n):
        batch = batch_states([env.state])
        assert batch["items"].shape == (1, n - step, 2)
        assert batch["action_mask"].tolist() == [[True, step == 0]]
        state, reward, done = env.step(int(step == 0))
        rewards.append(reward)
        assert done == (step == n - 1)
        assert (state is None) == done
    assert sum(rewards) == 1 / n
    assert tuple(env.selected_indices) == (n - 1,)


def test_batch_rejects_terminal_states_instead_of_pooling_an_empty_suffix():
    instance = _instance([1], [2], 1)
    for states in ([], [None], [State(instance, 1, 0)], [State(instance, -1, 0)]):
        with pytest.raises(ValueError):
            batch_states(states)
