"""Selected-action feedback for a tabular contextual strategy bandit."""
from __future__ import annotations

from bisect import bisect_right
import math

import numpy as np


def context_key(instance: dict, bins: dict) -> tuple[int, ...]:
    """Permutation-invariant input statistics; no labels, IDs or solver outcomes."""
    weights = np.asarray(instance["weights"], dtype=float)
    values = np.asarray(instance["values"], dtype=float)
    if len(weights) != len(values) or np.any(weights <= 0) or np.any(values < 0):
        raise ValueError("Expected positive weights and nonnegative values")
    capacity = instance["capacity"]
    if capacity < 0:
        raise ValueError("Capacity must be nonnegative")
    density = values / weights
    mean = float(density.mean()) if len(density) else 0.
    features = (len(weights), capacity / max(1., float(weights.sum())),
                float(density.std()) / mean if mean else 0.,
                float(weights.max()) / max(1, capacity) if len(weights) else 0.)
    names = ("n", "capacity_fraction", "density_cv", "max_weight_over_capacity")
    return tuple(bisect_right(bins[name], value) for name, value in zip(names, features))


class Bandit:
    def __init__(self, arms: list[str], prior_strength: float = 5.):
        if not arms or len(set(arms)) != len(arms) or prior_strength <= 0:
            raise ValueError("Unique nonempty arms and positive prior strength required")
        self.arms = list(arms)
        self.prior_strength = prior_strength
        self.counts: dict[tuple, np.ndarray] = {}
        self.totals: dict[tuple, np.ndarray] = {}
        self.global_counts = np.zeros(len(arms), dtype=np.int64)
        self.global_totals = np.zeros(len(arms))

    def estimates(self, key: tuple) -> np.ndarray:
        means = np.divide(self.global_totals, self.global_counts,
                          out=np.zeros(len(self.arms)), where=self.global_counts > 0)
        if key not in self.counts:
            return means
        return ((self.totals[key] + self.prior_strength * means)
                / (self.counts[key] + self.prior_strength))

    def predict(self, key: tuple) -> int:
        # Fixed action order breaks ties; prediction never changes the learner.
        return int(np.argmax(self.estimates(key)))

    def update(self, key: tuple, action: int, reward: float) -> None:
        if not 0 <= action < len(self.arms) or not math.isfinite(reward) or not 0 <= reward <= 1 + 1e-12:
            raise ValueError("Invalid action or normalized reward")
        if key not in self.counts:
            self.counts[key] = np.zeros(len(self.arms), dtype=np.int64)
            self.totals[key] = np.zeros(len(self.arms))
        self.counts[key][action] += 1
        self.totals[key][action] += reward
        self.global_counts[action] += 1
        self.global_totals[action] += reward

    def to_dict(self) -> dict:
        return dict(arms=self.arms, prior_strength=self.prior_strength,
                    counts={",".join(map(str, k)): v.tolist() for k, v in self.counts.items()},
                    totals={",".join(map(str, k)): v.tolist() for k, v in self.totals.items()},
                    global_counts=self.global_counts.tolist(), global_totals=self.global_totals.tolist())

    @classmethod
    def from_dict(cls, state: dict) -> Bandit:
        model = cls(state["arms"], state["prior_strength"])
        key = lambda text: tuple(map(int, text.split(",")))
        model.counts = {key(k): np.array(v, dtype=np.int64) for k, v in state["counts"].items()}
        model.totals = {key(k): np.array(v, dtype=float) for k, v in state["totals"].items()}
        model.global_counts = np.array(state["global_counts"], dtype=np.int64)
        model.global_totals = np.array(state["global_totals"], dtype=float)
        return model


def source_mean(values: np.ndarray, sources: list[str]) -> float:
    groups = sorted(set(sources))
    return float(np.mean([np.mean(values[np.array(sources) == source]) for source in groups]))


def fit(rewards: np.ndarray, contexts: list[tuple], sources: list[str],
        validation_rewards: np.ndarray, validation_contexts: list[tuple],
        validation_sources: list[str], config: dict, seed: int) -> tuple[Bandit, dict]:
    """Replay a cached simulator, revealing ONLY the chosen arm to each update.

    Validation selects the saved checkpoint but never contributes an update.
    Every epoch visits each training instance exactly once, alternating sources.
    """
    if rewards.shape != (len(contexts), len(config["arms"])) or len(sources) != len(contexts):
        raise ValueError("Training cache shape mismatch")
    if validation_rewards.shape != (len(validation_contexts), len(config["arms"])):
        raise ValueError("Validation cache shape mismatch")
    groups = {s: np.flatnonzero(np.array(sources) == s) for s in sorted(set(sources))}
    if len(groups) != 2 or len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError("Training requires two equally sized source groups")
    rng = np.random.default_rng(seed)
    model = Bandit(config["arms"], config["prior_strength"])
    best_state, best_score, best_step = None, -math.inf, 0
    step, history = 0, []
    source_counts = {s: 0 for s in groups}
    total_steps = config["epochs"] * len(contexts)
    if total_steps <= 0:
        raise ValueError("Positive training budget required")
    for epoch in range(config["epochs"]):
        order = np.column_stack([rng.permutation(indices) for indices in groups.values()]).ravel()
        for index in order:
            epsilon = config["epsilon_start"] + (config["epsilon_end"] - config["epsilon_start"]) * step / max(1, total_steps - 1)
            action = int(rng.integers(len(model.arms))) if rng.random() < epsilon else model.predict(contexts[index])
            model.update(contexts[index], action, float(rewards[index, action]))
            source_counts[sources[index]] += 1
            step += 1
        if (epoch + 1) % config["validation_every_epochs"] == 0 or epoch + 1 == config["epochs"]:
            actions = np.array([model.predict(key) for key in validation_contexts])
            score = source_mean(validation_rewards[np.arange(len(actions)), actions], validation_sources)
            history.append(dict(epoch=epoch + 1, step=step, validation_quality=score))
            if score > best_score:
                best_score, best_step, best_state = score, step, model.to_dict()
    return Bandit.from_dict(best_state), dict(seed=seed, observations=step,
        source_observations=source_counts, best_step=best_step,
        best_validation_quality=best_score, history=history,
        unique_training_instances=len(contexts), final_action_observations=model.global_counts.tolist())
