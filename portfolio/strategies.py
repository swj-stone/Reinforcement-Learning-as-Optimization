"""Seven deterministic offline knapsack candidates in original item coordinates.

The 1/e rule is an adapted density-threshold heuristic over a stored permutation.
Its observation prefix is irrevocably rejected. It has no secretary-problem
optimality guarantee and does not turn the frozen offline corpus into online data.

``seeded_greedy8`` and ``swap_repair8`` compare at most eight alternatives with
the density-greedy baseline, so their returned value is never below that baseline.
Their additional construction cost must be included in portfolio timing.
"""
from __future__ import annotations

from functools import cmp_to_key
import math
from typing import Iterable, Mapping, Sequence

from experiment.baselines.methods import BaselineSolution, greedy_solution
from experiment.eval.reference import ratio_order


CANDIDATE_NAMES = (
    "greedy_density",
    "value_first",
    "light_first",
    "density_sqrt",
    "one_e_threshold",
    "seeded_greedy8",
    "swap_repair8",
)


def candidate_names() -> tuple[str, ...]:
    """Stable non-neural candidate order; the calling portfolio supplies DQN."""
    return CANDIDATE_NAMES


def choose_best(solutions: Iterable[BaselineSolution]) -> BaselineSolution:
    """Maximize value; break ties by lower total weight, then sorted indices.

    The empty iterable has the feasible empty solution as its identity. Callers
    provide feasible solutions; this function does not have an instance against
    which it could reconstruct totals or check capacity.
    """
    best = min(solutions, key=lambda result: (
        -result.value, result.weight_used, tuple(sorted(result.selected_indices))), default=None)
    if best is None:
        return BaselineSolution((), 0, 0)
    return BaselineSolution(tuple(sorted(best.selected_indices)), best.value, best.weight_used)


def _solution(instance: Mapping, selected: Iterable[int]) -> BaselineSolution:
    indices = tuple(sorted(selected))
    return BaselineSolution(indices, sum(instance["values"][i] for i in indices),
                            sum(instance["weights"][i] for i in indices))


def _fill(instance: Mapping, order: Sequence[int], selected: Iterable[int] = ()) -> BaselineSolution:
    """Fill remaining capacity in the supplied order, retaining any forced items."""
    chosen = set(selected)
    weights = instance["weights"]
    remaining = instance["capacity"] - sum(weights[i] for i in chosen)
    for i in order:
        if i not in chosen and weights[i] <= remaining:
            chosen.add(i)
            remaining -= weights[i]
    return _solution(instance, chosen)


def _value_order(instance: Mapping) -> list[int]:
    return sorted(range(len(instance["weights"])),
                  key=lambda i: (-instance["values"][i], instance["weights"][i], i))


def _rank_compare(instance: Mapping, *, squared: bool = False, ascending: bool = False):
    """Exact ratio comparison with lower-weight, then lower-index tie breaks."""
    weights, values = instance["weights"], instance["values"]

    def compare(i: int, j: int) -> int:
        vi = values[i] * values[i] if squared else values[i]
        vj = values[j] * values[j] if squared else values[j]
        left, right = vi * weights[j], vj * weights[i]
        if left != right:
            if ascending:
                return -1 if left < right else 1
            return -1 if left > right else 1
        if weights[i] != weights[j]:
            return -1 if weights[i] < weights[j] else 1
        return (i > j) - (i < j)

    return compare


def _one_e_threshold(instance: Mapping) -> BaselineSolution:
    order = instance["item_order"]
    weights, values = instance["weights"], instance["values"]
    observed_count = math.floor(len(weights) / math.e)
    threshold_value, threshold_weight = 0, 1
    for i in order[:observed_count]:
        if values[i] * threshold_weight > threshold_value * weights[i]:
            threshold_value, threshold_weight = values[i], weights[i]
    remaining = instance["capacity"]
    selected = []
    for i in order[observed_count:]:
        if weights[i] <= remaining and values[i] * threshold_weight >= threshold_value * weights[i]:
            selected.append(i)
            remaining -= weights[i]
    return _solution(instance, selected)


def _seeded_greedy8(instance: Mapping) -> BaselineSolution:
    density = ratio_order(instance["weights"], instance["values"])
    seeds = dict.fromkeys(_value_order(instance)[:4] + density[:4])
    alternatives = [greedy_solution(instance)]
    for i in seeds:
        if instance["weights"][i] <= instance["capacity"]:
            alternatives.append(_fill(instance, density, (i,)))
    return choose_best(alternatives)


def _swap_repair8(instance: Mapping) -> BaselineSolution:
    base = greedy_solution(instance)
    selected = set(base.selected_indices)
    candidates = [i for i in _value_order(instance) if i not in selected][:8]
    density = ratio_order(instance["weights"], instance["values"])
    removals = sorted(selected, key=cmp_to_key(_rank_compare(instance, ascending=True)))
    alternatives = [base]
    for candidate in candidates:
        if instance["weights"][candidate] > instance["capacity"]:
            continue
        # Each bounded repair starts from the original greedy set, not the last repair.
        retained = set(selected)
        remaining = instance["capacity"] - base.weight_used
        for i in removals:
            if instance["weights"][candidate] <= remaining:
                break
            retained.remove(i)
            remaining += instance["weights"][i]
        retained.add(candidate)
        alternatives.append(_fill(instance, density, retained))
    return choose_best(alternatives)


def solve_strategy(name: str, instance: Mapping) -> BaselineSolution:
    """Construct one candidate from a valid positive-integer knapsack instance.

    The instance uses the frozen corpus's ``weights``, ``values``, ``capacity``
    and stored ``item_order`` fields. Returned indices always refer to the
    original arrays. All methods except ``one_e_threshold`` ignore item_order.
    """
    if name == "greedy_density":
        return greedy_solution(instance)
    if name == "value_first":
        return _fill(instance, _value_order(instance))
    if name == "light_first":
        order = sorted(range(len(instance["weights"])),
                       key=lambda i: (instance["weights"][i], -instance["values"][i], i))
        return _fill(instance, order)
    if name == "density_sqrt":
        order = sorted(range(len(instance["weights"])), key=cmp_to_key(_rank_compare(instance, squared=True)))
        return _fill(instance, order)
    if name == "one_e_threshold":
        return _one_e_threshold(instance)
    if name == "seeded_greedy8":
        return _seeded_greedy8(instance)
    if name == "swap_repair8":
        return _swap_repair8(instance)
    raise ValueError(f"unknown strategy: {name!r}")
