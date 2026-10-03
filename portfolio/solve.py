"""Solve a JSON knapsack with value-first portfolio search or learned selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiment.rl.artifacts import sha256
from experiment.rl.training import verify_solution
from portfolio.bandit import Bandit
from portfolio.run import REPO, decide, load_network, read_json, require_hashes, solve_arm
from portfolio.strategies import choose_best


def normalize_instance(instance: dict) -> dict:
    weights, values, capacity = instance["weights"], instance["values"], instance["capacity"]
    if (len(weights) != len(values) or type(capacity) is not int or capacity < 0
            or any(type(w) is not int or w <= 0 for w in weights)
            or any(type(v) is not int or v < 0 for v in values)):
        raise ValueError("Use equally sized positive integer weights and nonnegative integer values/capacity")
    n = len(weights)
    order = instance.get("item_order", list(range(n)))
    if any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
        raise ValueError("item_order must be a permutation of original item indices")
    return dict(instance, n=n, item_order=order)


def best_of_portfolio(instance: dict, arms: list[str], networks: dict):
    candidates = [(arm, solve_arm(arm, instance, networks)) for arm in arms]
    solution = choose_best(s for _, s in candidates)
    arm = next(arm for arm, s in candidates if s == solution)
    return solution, arm, {arm: s.value for arm, s in candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--mode", choices=("best", "learned", "guarded"), default="best")
    parser.add_argument("--training", type=Path, default=REPO / "results/portfolio-v2-training")
    args = parser.parse_args()
    instance = normalize_instance(read_json(args.instance))
    training = read_json(args.training / "training.json")
    if training["status"] != "completed":
        raise ValueError("Training artifacts are incomplete")
    require_hashes(args.training, training["artifacts"])
    require_hashes(REPO, training["core_code"])
    config = training["config"]
    path = REPO / config["dqn_checkpoint"]
    if sha256(path) != config["dqn_checkpoint_sha256"]:
        raise ValueError("Frozen DQN checkpoint changed")
    torch.set_num_threads(1)
    networks = {"dqn_seed0": load_network(path)}
    seed = max(training["runs"], key=lambda row: row["best_validation_quality"])["seed"]
    model = Bandit.from_dict(read_json(args.training / f"seed{seed}.json")["model"])
    start = time.perf_counter()
    if args.mode == "best":
        solution, arm, candidate_values = best_of_portfolio(instance, config["arms"], networks)
    else:
        solution, arm = decide(instance, model, config, networks, guarded=args.mode == "guarded")
        candidate_values = None
    elapsed = time.perf_counter() - start
    verify_solution(instance, solution)
    print(json.dumps(dict(mode=args.mode, chosen_arm=arm, value=solution.value,
        weight_used=solution.weight_used, selected_indices=solution.selected_indices,
        candidate_values=candidate_values, validation_selected_seed=seed if args.mode != "best" else None,
        solver_seconds=elapsed, guarantee="best within the eight candidates" if args.mode == "best" else
        "at least greedy value" if args.mode == "guarded" else "learned expected reward; no instance-wise optimality guarantee"), indent=2))


if __name__ == "__main__":
    main()
