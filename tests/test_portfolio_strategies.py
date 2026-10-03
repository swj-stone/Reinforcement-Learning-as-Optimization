"""Hand-derived and exhaustive fixtures for the bounded offline portfolio."""
from __future__ import annotations

import random

import pytest

from experiment.baselines.methods import BaselineSolution, greedy_solution
from portfolio.strategies import CANDIDATE_NAMES, candidate_names, choose_best, solve_strategy


def instance(weights, values, capacity, order=None):
    return dict(weights=weights, values=values, capacity=capacity, n=len(weights),
                item_order=list(range(len(weights))) if order is None else order)


def brute_optimum(problem):
    n = len(problem["weights"])
    feasible_values = []
    for bits in range(1 << n):
        selected = [i for i in range(n) if bits & (1 << i)]
        if sum(problem["weights"][i] for i in selected) <= problem["capacity"]:
            feasible_values.append(sum(problem["values"][i] for i in selected))
    return max(feasible_values)


def test_public_candidates_are_callable_and_do_not_include_neural_policy():
    problem = instance([1], [5], 1)
    assert candidate_names() == CANDIDATE_NAMES
    assert len(set(CANDIDATE_NAMES)) == 7
    assert "dqn" not in CANDIDATE_NAMES
    for name in candidate_names():
        assert isinstance(solve_strategy(name, problem), BaselineSolution)
    with pytest.raises(ValueError, match="unknown strategy"):
        solve_strategy("unregistered", problem)


@pytest.mark.parametrize("name,expected", [
    ("greedy_density", BaselineSolution((0, 1), 160, 30)),
    ("value_first", BaselineSolution((1, 2), 220, 50)),
    ("light_first", BaselineSolution((0, 1), 160, 30)),
    ("density_sqrt", BaselineSolution((1, 2), 220, 50)),
    ("seeded_greedy8", BaselineSolution((0, 2), 180, 40)),
    ("swap_repair8", BaselineSolution((0, 2), 180, 40)),
])
def test_analytical_knapsack_distinguishes_the_strategies(name, expected):
    problem = instance([10, 20, 30], [60, 100, 120], 50)
    assert solve_strategy(name, problem) == expected


def test_simple_density_trap_is_fixed_by_seeded_and_swap_candidates():
    problem = instance([6, 5, 5], [13, 10, 10], 10)
    assert greedy_solution(problem).value == 13
    assert solve_strategy("seeded_greedy8", problem) == BaselineSolution((1, 2), 20, 10)
    assert solve_strategy("swap_repair8", problem) == BaselineSolution((1, 2), 20, 10)


@pytest.mark.parametrize("name,problem,expected", [
    ("value_first", instance([3, 2, 2], [8, 8, 8], 3), (1,)),
    ("light_first", instance([2, 2, 3], [4, 5, 100], 2), (1,)),
    ("density_sqrt", instance([1, 4], [2, 4], 4), (0,)),
    ("density_sqrt", instance([1, 1], [2**60, 2**60 + 1], 1), (1,)),
])
def test_rank_ties_and_large_integer_scores_are_exact(name, problem, expected):
    assert solve_strategy(name, problem).selected_indices == expected


def test_one_e_prefix_is_rejected_and_stored_order_controls_observation():
    # n=3, floor(3/e)=1: observe item 1 at density 5, then accept item 0 at density 100.
    problem = instance([1, 2, 3], [100, 10, 12], 5, [1, 0, 2])
    assert solve_strategy("one_e_threshold", problem) == BaselineSolution((0,), 100, 1)
    # The high-value prefix item is irrevocably rejected; equality meets the threshold.
    problem = instance([1, 1, 1], [100, 100, 99], 2)
    assert solve_strategy("one_e_threshold", problem) == BaselineSolution((1,), 100, 1)


def test_one_e_threshold_uses_entire_prefix_and_does_not_peek_at_suffix():
    # floor(6/e)=2, threshold=max(4,5)=5; a later score of 100 never changes acceptance of 6.
    problem = instance([1] * 6, [4, 5, 6, 100, 3, 2], 1)
    assert solve_strategy("one_e_threshold", problem) == BaselineSolution((2,), 6, 1)
    changed_future = dict(problem, values=[4, 5, 6, 1000000, 3, 2])
    assert solve_strategy("one_e_threshold", changed_future) == BaselineSolution((2,), 6, 1)
    # If the second observed density is ignored, item 2 would incorrectly be admitted.
    problem = instance([1] * 6, [4, 8, 6, 9, 3, 2], 1)
    assert solve_strategy("one_e_threshold", problem) == BaselineSolution((3,), 9, 1)


def test_one_e_has_zero_threshold_when_observation_prefix_is_empty():
    assert solve_strategy("one_e_threshold", instance([1, 2], [3, 4], 3)) == BaselineSolution((0, 1), 7, 3)


def test_one_e_continues_after_an_infeasible_qualifying_item():
    problem = instance([1, 20, 2], [1, 100, 4], 2)
    assert solve_strategy("one_e_threshold", problem) == BaselineSolution((2,), 4, 2)


def test_swap_repairs_only_the_top_eight_unselected_items():
    # The ninth unselected item would improve 13 to 20; the bounded top-eight all exceed capacity.
    problem = instance([6] + [100] * 8 + [10], [13] + [1000] * 8 + [20], 10)
    assert solve_strategy("swap_repair8", problem) == BaselineSolution((0,), 13, 6)


def test_seeded_search_is_limited_to_top_value_and_density_seeds():
    # All four highest-value and highest-density seeds are infeasible. Trying every
    # remaining item would find 20, but the bounded portfolio must retain greedy's 13.
    problem = instance([6] + [100] * 8 + [10], [13] + [1000] * 8 + [20], 10)
    assert solve_strategy("seeded_greedy8", problem) == BaselineSolution((0,), 13, 6)


def test_each_swap_repair_starts_from_original_greedy_not_previous_improvement():
    problem = instance([13, 6, 8, 9, 1, 14, 5], [28, 21, 26, 21, 26, 16, 13], 25)
    assert greedy_solution(problem) == BaselineSolution((1, 2, 4, 6), 86, 20)
    # Forcing item 0 yields 88, but the independent item-3 repair yields 94.
    # Cascading from the 88-point repair instead would get stuck at 88.
    assert solve_strategy("swap_repair8", problem) == BaselineSolution((1, 2, 3, 4), 94, 24)


def test_choose_best_uses_value_then_weight_then_canonical_indices():
    low_value = BaselineSolution((0,), 9, 1)
    high_weight = BaselineSolution((1,), 10, 5)
    high_index = BaselineSolution((3, 2), 10, 4)
    expected = BaselineSolution((0, 2), 10, 4)
    solutions = [low_value, high_weight, high_index, expected]
    assert choose_best(solutions) == expected
    assert choose_best(iter(reversed(solutions))) == expected
    assert choose_best([]) == BaselineSolution((), 0, 0)


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
@pytest.mark.parametrize("problem", [instance([], [], 0), instance([2, 3], [6, 5], 0), instance([3, 4], [8, 9], 2)])
def test_empty_and_no_item_fits_return_exact_empty_solution(name, problem):
    assert solve_strategy(name, problem) == BaselineSolution((), 0, 0)


@pytest.mark.parametrize("name", [name for name in CANDIDATE_NAMES if name != "one_e_threshold"])
def test_offline_ranked_strategies_ignore_stored_order(name):
    problem = instance([6, 5, 5, 1], [13, 10, 11, 2], 10)
    changed = dict(problem, item_order=[3, 2, 1, 0])
    assert solve_strategy(name, problem) == solve_strategy(name, changed)


def test_all_strategies_are_feasible_and_bounded_by_exhaustive_optimum():
    rng = random.Random(4821)
    for _ in range(100):
        n = rng.randrange(11)
        weights = [rng.randint(1, 15) for _ in range(n)]
        values = [rng.randint(1, 40) for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        problem = instance(weights, values, rng.randint(0, sum(weights)), order)
        optimum = brute_optimum(problem)
        greedy = greedy_solution(problem)
        for name in CANDIDATE_NAMES:
            result = solve_strategy(name, problem)
            selected = result.selected_indices
            assert selected == tuple(sorted(set(selected)))
            assert all(0 <= i < n for i in selected)
            assert result.value == sum(values[i] for i in selected)
            assert result.weight_used == sum(weights[i] for i in selected) <= problem["capacity"]
            assert result.value <= optimum
            assert result == solve_strategy(name, problem)
            if name in {"seeded_greedy8", "swap_repair8"}:
                assert result.value >= greedy.value
            elif name == "greedy_density":
                assert result == greedy
