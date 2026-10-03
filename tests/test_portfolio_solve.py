import pytest

from portfolio.solve import normalize_instance, best_of_portfolio


def test_best_mode_selects_the_stronger_candidate_and_returns_audit_values():
    instance = normalize_instance({"weights": [10, 20, 30], "values": [60, 100, 120], "capacity": 50})
    solution, arm, values = best_of_portfolio(instance, ["greedy_density", "value_first"], {})
    assert solution.value == 220 and solution.selected_indices == (1, 2)
    assert arm == "value_first"
    assert values == {"greedy_density": 160, "value_first": 220}


@pytest.mark.parametrize("change", [{"weights": [0]}, {"weights": [1.5]},
                                   {"values": [-1]}, {"capacity": -1},
                                   {"item_order": [1]}, {"values": [1, 2]}])
def test_invalid_input_fails_before_any_solver(change):
    with pytest.raises(ValueError):
        normalize_instance({"weights": [1], "values": [1], "capacity": 1, **change})
