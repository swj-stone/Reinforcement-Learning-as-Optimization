"""Saved-order 0/1 knapsack episodes and padded remaining-item observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class State:
    """A replay view of one corpus instance; item arrays are shared, never copied.

    Callers treat the instance dictionary as immutable for the duration of a run.
    A terminal episode is represented by ``None``, not by a State at position n.
    """

    instance: dict
    position: int
    remaining_capacity: int


class KnapsackEnv:
    """Process every item once in the stored permutation, including after C=0."""

    def __init__(self, instance: dict):
        self._instance = instance
        self._selected: list[int] = []
        self._value_denominator = max(sum(instance["values"]), 1)
        self.state: State | None = (
            State(instance, 0, instance["capacity"]) if instance["weights"] else None
        )

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """Return unique original-array indices in the baseline output order."""
        return tuple(sorted(self._selected))

    def step(self, action: int) -> tuple[State | None, float, bool]:
        state = self.state
        if state is None:
            raise ValueError("Cannot step a terminal episode")
        if not isinstance(action, Integral) or action not in (0, 1):
            raise ValueError("Action must be 0 (skip) or 1 (include)")
        index = self._instance["item_order"][state.position]
        remaining = state.remaining_capacity
        reward = 0.0
        if action == 1:
            weight = self._instance["weights"][index]
            if weight > remaining:
                raise ValueError("Include action exceeds remaining capacity")
            remaining -= weight
            self._selected.append(index)
            reward = self._instance["values"][index] / self._value_denominator
        position = state.position + 1
        done = position == len(self._instance["weights"])
        self.state = None if done else State(self._instance, position, remaining)
        return self.state, reward, done


def batch_states(
    states: Sequence[State], device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Encode only unprocessed items, padding suffixes to the largest batch size.

    Float normalization never decides feasibility: the action mask compares the
    original integer weight with the integer remaining capacity. ``valid=True``
    identifies a real unprocessed item, including the current one.
    """
    if not states:
        raise ValueError("A state batch must contain nonterminal observations")
    lengths = []
    for state in states:
        if state is None or not 0 <= state.position < len(state.instance["weights"]):
            raise ValueError("Terminal or invalid positions cannot be batched")
        if state.remaining_capacity < 0:
            raise ValueError("Remaining capacity must be nonnegative")
        lengths.append(len(state.instance["weights"]) - state.position)

    items = np.zeros((len(states), max(lengths), 2), dtype=np.float32)
    valid = np.zeros((len(states), max(lengths)), dtype=np.bool_)
    context = np.empty((len(states), 5), dtype=np.float32)
    action_mask = np.ones((len(states), 2), dtype=np.bool_)
    for row, (state, length) in enumerate(zip(states, lengths)):
        instance = state.instance
        n = len(instance["weights"])
        order = instance["item_order"][state.position:]
        capacity_denominator = max(instance["capacity"], 1)
        value_denominator = max(sum(instance["values"]), 1)
        items[row, :length, 0] = (
            np.asarray(instance["weights"], dtype=np.float64)[order] / capacity_denominator
        )
        items[row, :length, 1] = (
            np.asarray(instance["values"], dtype=np.float64)[order] / value_denominator
        )
        valid[row, :length] = True
        context[row] = (
            items[row, 0, 0], items[row, 0, 1],
            state.remaining_capacity / capacity_denominator,
            length / n, math.log1p(n),
        )
        action_mask[row, 1] = instance["weights"][order[0]] <= state.remaining_capacity
    return {
        "items": torch.from_numpy(items).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "context": torch.from_numpy(context).to(device),
        "action_mask": torch.from_numpy(action_mask).to(device),
    }
