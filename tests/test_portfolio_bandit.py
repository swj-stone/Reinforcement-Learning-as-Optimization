import copy

import numpy as np

from portfolio.bandit import Bandit, context_key, fit


BINS = {"n": [30, 75, 150], "capacity_fraction": [.375, .625],
        "density_cv": [.5, 1.5], "max_weight_over_capacity": [.1, .5]}


def test_context_does_not_use_metadata_or_item_order():
    a = dict(weights=[1, 3, 2], values=[5, 8, 4], capacity=3, n=3,
             source="real", family=None, instance_id="a", item_order=[2, 1, 0])
    b = dict(a, weights=[3, 2, 1], values=[8, 4, 5], source="synthetic",
             family="uniform", instance_id="b", item_order=[0, 2, 1])
    assert context_key(a, BINS) == context_key(b, BINS)
    assert len(context_key(dict(weights=[], values=[], capacity=0), BINS)) == 4


def test_only_selected_action_is_updated_and_unseen_context_uses_global():
    model = Bandit(["a", "b"], prior_strength=5)
    model.update((0,), 1, .9)
    assert model.counts[(0,)].tolist() == [0, 1]
    assert model.totals[(0,)].tolist() == [0, .9]
    assert model.predict((999,)) == 1
    before = copy.deepcopy(model.to_dict())
    model.predict((0,))
    assert before == model.to_dict()


def test_complementary_contexts_learn_different_actions_and_roundtrip():
    model = Bandit(["a", "b"])
    for _ in range(50):
        for key, rewards in [((0,), [1, 0]), ((1,), [0, 1])]:
            for action, reward in enumerate(rewards):
                model.update(key, action, reward)
    assert [model.predict((k,)) for k in (0, 1)] == [0, 1]
    restored = Bandit.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    assert restored.predict((1,)) == 1


def test_fit_is_deterministic_balanced_and_keeps_validation_out_of_updates():
    config = dict(arms=["a", "b"], epochs=6, epsilon_start=1., epsilon_end=.1,
                  prior_strength=5., validation_every_epochs=1)
    rewards = np.tile([[1., 0.], [0., 1.]], (100, 1))
    contexts = [(i % 2,) for i in range(200)]
    sources = ["real"] * 100 + ["synthetic"] * 100
    args = (rewards, contexts, sources, np.array([[1., 0.], [0., 1.]]),
            [(0,), (1,)], ["real", "synthetic"], config, 17)
    model, log = fit(*args)
    model2, log2 = fit(*args)
    assert model.to_dict() == model2.to_dict()
    assert log == log2
    assert log["observations"] == 1200
    assert log["source_observations"] == {"real": 600, "synthetic": 600}
    assert log["best_validation_quality"] == 1
    assert sum(sum(row) for row in model.to_dict()["counts"].values()) == log["best_step"]
    assert log["best_step"] <= 1200
