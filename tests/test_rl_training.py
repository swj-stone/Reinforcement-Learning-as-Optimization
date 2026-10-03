"""Learning-target and end-to-end training tests on small, real tensors."""
from fractions import Fraction
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from experiment.baselines.methods import BaselineSolution
from experiment.rl.environment import State
from experiment.rl.model import QNetwork
from experiment.rl.sampling import Transition
from experiment.rl.training import td_targets, validation_scores, train, is_better


def instance(source="synthetic", key="s"):
    return dict(instance_id=key, source=source, family="uniform" if source == "synthetic" else "online_retail",
                n=3, alpha=0.5, capacity=6, weights=[4, 3, 3], values=[7, 5, 5], item_order=[2, 0, 1])


def small_config():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "experiment_config.yaml").read_text(encoding="utf-8"))
    config["budgets"].update(batch_size=4, replay_capacity=20,
        replay_source_balance={"real": 2, "synthetic": 2}, warmup_transitions=4,
        update_every_transitions=1, target_copy_transitions=4,
        validation_every_transitions=6)
    return config


def test_td_terminal_bypasses_network_and_infeasible_next_include_is_masked():
    model = QNetwork()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.head[-1].bias.data.copy_(torch.tensor([2.0, 1000.0]))
    s = State(instance(), 0, 6)
    blocked = State(s.instance, 1, 1)
    seen = []
    hook = model.register_forward_hook(lambda module, inputs, output: seen.append(len(output)))
    targets = td_targets(model, [Transition(s, 0, 0.25, blocked), Transition(s, 1, 0.5, None)])
    hook.remove()
    assert targets.tolist() == pytest.approx([2.25, 0.5])
    assert seen == [1]
    seen = []
    hook = model.register_forward_hook(lambda module, inputs, output: seen.append(len(output)))
    assert td_targets(model, [Transition(s, 1, 0.5, None)]).item() == 0.5
    hook.remove()
    assert seen == []
    assert not targets.requires_grad


def test_validation_uses_fractional_upper_bound_equal_sources_and_earlier_ties():
    a, b, c = instance(key="a"), instance(key="b"), instance("real", "c")
    refs = {key: {"value": 10, "upper_bound": Fraction(12)} for key in ("a", "b", "c")}
    solutions = [BaselineSolution((1, 2), 10, 6), BaselineSolution((1, 2), 10, 6), BaselineSolution((), 0, 0)]
    scores = validation_scores([a, b, c], solutions, refs)
    assert scores["score"] == pytest.approx(5 / 12)
    assert scores["source_means"]["synthetic"] == pytest.approx(10 / 12)
    assert is_better(0.6, 0.5)
    assert not is_better(0.5, 0.5)
    with pytest.raises(ValueError):
        is_better(float("nan"), 0.5)


def test_small_run_updates_weights_balances_exposure_and_reloads(tmp_path):
    data = [instance(), instance("real", "r")]
    refs = {x["instance_id"]: {"value": 10, "upper_bound": Fraction(12)} for x in data}
    config = small_config()
    run = train(data, data, refs, config, tmp_path / "run", seed=0, transitions=13,
                provenance={"fixture": True}, run_kind="custom")
    assert run["status"] == "completed"
    assert run["transitions"] == 18  # Finish the third balanced six-transition block.
    assert run["exposure"]["transitions_by_source"] == {"synthetic": 9, "real": 9}
    assert run["exposure"]["episodes_by_source_size"] == {"synthetic/3": 3, "real/3": 3}
    assert run["exposure"]["unique_instances_by_source_size"] == {"synthetic/3": 1, "real/3": 1}
    assert run["optimizer_updates"] > 0
    assert math.isfinite(run["best_validation_score"])
    best = torch.load(tmp_path / "run/best.pt", weights_only=True)
    last = torch.load(tmp_path / "run/last.pt", weights_only=True)
    assert best["step"] <= 18
    assert last["step"] == 18
    assert last["training"]["requested_transitions"] == 13
    assert best["training_status"] == last["training_status"] == "completed"
    assert best["training"]["actual_transitions"] == 18
    assert last["training"]["runtime"]["threads"] == 1
    assert "rl/training.py" in last["training"]["code_sha256"]
    model = QNetwork()
    model.load_state_dict(best["model"])
    torch.manual_seed(0)
    initial = QNetwork().state_dict()
    assert any(not torch.equal(initial[k], last["model"][k]) for k in initial)
    assert json.loads((tmp_path / "run/run.json").read_text())["run_kind"] == "custom"
    run2 = train(data, data, refs, config, tmp_path / "repeat", seed=0, transitions=13,
                 provenance={"fixture": True}, run_kind="custom")
    last2 = torch.load(tmp_path / "repeat/last.pt", weights_only=True)
    assert all(torch.equal(last["model"][k], last2["model"][k]) for k in last["model"])
    with pytest.raises(FileExistsError):
        train(data, data, refs, config, tmp_path / "run", seed=0, transitions=13,
              provenance={}, run_kind="custom")


@pytest.mark.parametrize("field,value", [("batch_size", 5), ("replay_capacity", 3),
                                         ("warmup_transitions", -1), ("learning_rate", float("nan"))])
def test_invalid_config_is_rejected_before_creating_run(tmp_path, field, value):
    config = small_config()
    config["budgets"][field] = value
    with pytest.raises(ValueError):
        train([], [], {}, config, tmp_path / "invalid", seed=0, transitions=12,
              provenance={}, run_kind="custom")
    assert not (tmp_path / "invalid").exists()
