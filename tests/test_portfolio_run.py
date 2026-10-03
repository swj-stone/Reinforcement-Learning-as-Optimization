from pathlib import Path

import pytest

from portfolio.bandit import Bandit
from portfolio.run import decide, require_hashes


def test_guarded_policy_runs_greedy_when_selected_rule_loses():
    instance = dict(weights=[1, 1, 1], values=[100, 1, 1], capacity=1,
                    item_order=[0, 1, 2])
    config = {"context_bins": {"n": [], "capacity_fraction": [], "density_cv": [],
                               "max_weight_over_capacity": []}}
    model = Bandit(["one_e_threshold"])
    model.update((0, 0, 0, 0), 0, 0.)
    plain, arm = decide(instance, model, config, {}, guarded=False)
    guarded, guarded_arm = decide(instance, model, config, {}, guarded=True)
    assert plain.value == 0 and arm == "one_e_threshold"
    assert guarded.value == 100 and guarded_arm == "greedy_density"


def test_input_hash_check_rejects_modified_artifact(tmp_path):
    path = tmp_path / "data"
    path.write_text("original")
    from experiment.rl.artifacts import sha256
    expected = {"data": sha256(path)}
    require_hashes(tmp_path, expected)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        require_hashes(tmp_path, expected)
