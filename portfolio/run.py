"""Freeze, train and evaluate the follow-up strategy portfolio on the local venv."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiment.baselines.methods import random_solution
from experiment.baselines.seeds import trial_seed
from experiment.eval.reference import fractional_upper_bound, solve as reference_solve
from experiment.rl.artifacts import check_inputs, code_fingerprint, sha256, write_json
from experiment.rl.model import QNetwork, solve_instances
from experiment.rl.training import verify_solution
from portfolio.bandit import Bandit, context_key, fit, source_mean
from portfolio.strategies import choose_best, solve_strategy


REPO = Path(__file__).resolve().parents[1]
COLUMNS = ("instance_id", "split", "method", "value", "weight_used", "selected_indices",
           "latency_s", "chosen_arm", "reference_value", "upper_bound", "certified")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_hashes(root: Path, expected: dict) -> None:
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise ValueError(f"hash mismatch: {name}")


def core_code() -> dict:
    return {f"portfolio/{name}.py": sha256(REPO / "portfolio" / f"{name}.py")
            for name in ("bandit", "strategies", "run", "holdout")}


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    paths = ["portfolio_config.json", "data_manifest.json", "reference_results.csv",
             "baseline_results.csv", "experiment_config.yaml", "EXPERIMENT_REPORT.md",
             "docs/superpowers/specs/2026-10-03-portfolio-design.md"]
    for seed in range(3):
        paths.extend([f"results/dqn-seed{seed}/best.pt", f"results/dqn-seed{seed}/run.json"])
    manifest = read_json(REPO / "data_manifest.json")
    paths.extend("datasets/processed/" + row["path"] for row in manifest["files"])
    config = read_json(REPO / "portfolio_config.json")
    if sha256(REPO / config["dqn_checkpoint"]) != config["dqn_checkpoint_sha256"]:
        raise ValueError("Candidate DQN checkpoint mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(created_at=datetime.now(timezone.utc).isoformat(), config=config,
        originals={name: sha256(REPO / name) for name in paths},
        experiment_code=code_fingerprint(), portfolio_core_code=core_code(),
        rationale="Prospective follow-up after inspecting original DQN results; same real-business test cohort."))
    print(f"Frozen protocol and baseline artifacts: {out}", flush=True)


def check_freeze(path: Path) -> dict:
    frozen = read_json(path)
    require_hashes(REPO, frozen["originals"])
    require_hashes(REPO / "experiment", frozen["experiment_code"])
    require_hashes(REPO, frozen["portfolio_core_code"])
    return frozen


def load_network(path: Path) -> QNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("training_status") != "completed" or checkpoint.get("run_kind") != "full":
        raise ValueError("Only a completed full baseline checkpoint is permitted")
    network = QNetwork()
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network


def solve_arm(name: str, instance: dict, networks: dict):
    if name.startswith("dqn_seed"):
        return solve_instances(networks[name], [instance])[0]
    return solve_strategy(name, instance)


def decide(instance: dict, model: Bandit, config: dict, networks: dict, *, guarded=False):
    action = model.predict(context_key(instance, config["context_bins"]))
    arm = model.arms[action]
    solution = solve_arm(arm, instance, networks)
    if guarded and arm != "greedy_density":
        greedy = solve_strategy("greedy_density", instance)
        if choose_best([solution, greedy]) == greedy:
            return greedy, "greedy_density"
    return solution, arm


def reward_cache(instances: list[dict], config: dict, network: QNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Simulator outcomes; the bandit is given one indexed reward per interaction."""
    start = time.perf_counter()
    outcomes = np.empty((len(instances), len(config["arms"])), dtype=np.int64)
    upper = np.array([float(fractional_upper_bound(i["weights"], i["values"], i["capacity"]))
                      for i in instances])
    for a, name in enumerate(config["arms"]):
        solutions = solve_instances(network, instances) if name.startswith("dqn_seed") else [
            solve_strategy(name, instance) for instance in instances]
        for row, (instance, solution) in enumerate(zip(instances, solutions)):
            verify_solution(instance, solution)
            outcomes[row, a] = solution.value
        print(f"Cached {name}: {len(instances)} cases, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    rewards = np.divide(outcomes, upper[:, None], out=np.ones_like(outcomes, dtype=float),
                        where=upper[:, None] > 0)
    if np.any(rewards > 1 + 1e-12):
        raise ValueError("Reward exceeds fractional bound")
    return rewards, outcomes


def train(out: Path, freeze_path: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    frozen = check_freeze(freeze_path)
    config = frozen["config"]
    data, _, provenance = check_inputs(REPO / "datasets/processed", REPO / "data_manifest.json",
                                       REPO / "reference_results.csv", REPO / "experiment_config.yaml")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    network = load_network(REPO / config["dqn_checkpoint"])
    out.mkdir(parents=True)
    started = time.perf_counter()
    rewards, outcomes, contexts, sources = {}, {}, {}, {}
    for split in ("train", "validation"):
        rewards[split], outcomes[split] = reward_cache(data[split], config, network)
        contexts[split] = [context_key(instance, config["context_bins"]) for instance in data[split]]
        sources[split] = [instance["source"] for instance in data[split]]
    cache_elapsed = time.perf_counter() - started
    np.savez_compressed(out / "reward_cache.npz", train=rewards["train"], validation=rewards["validation"],
                        train_values=outcomes["train"], validation_values=outcomes["validation"])
    write_json(out / "cache_index.json", {split: [i["instance_id"] for i in data[split]] for split in data})
    fixed_scores = {name: source_mean(rewards["validation"][:, a], sources["validation"])
                    for a, name in enumerate(config["arms"])}
    fixed_best = max(config["arms"], key=lambda name: fixed_scores[name])
    runs = []
    for seed in config["seeds"]:
        start = time.perf_counter()
        model, log = fit(rewards["train"], contexts["train"], sources["train"],
                         rewards["validation"], contexts["validation"], sources["validation"], config, seed)
        log["elapsed_seconds"] = time.perf_counter() - start
        write_json(out / f"seed{seed}.json", dict(model=model.to_dict(), training=log,
            config=config, freeze_sha256=sha256(freeze_path), core_code=core_code(), provenance=provenance))
        runs.append(log)
        print(f"Seed {seed}: best validation {log['best_validation_quality']:.8f} at {log['best_step']}; {log['elapsed_seconds']:.2f}s", flush=True)
    check_freeze(freeze_path)
    write_json(out / "training.json", dict(status="completed", config=config, provenance=provenance,
        freeze_sha256=sha256(freeze_path), core_code=core_code(), cache_seconds=cache_elapsed,
        elapsed_seconds=time.perf_counter() - started, fixed_best=fixed_best,
        fixed_validation_scores=fixed_scores, runs=runs,
        artifacts={name: sha256(out / name) for name in ["reward_cache.npz", "cache_index.json", "seed0.json", "seed1.json", "seed2.json"]},
        feedback="Selected action only; all-arm precomputation is simulator cache work included in cache_seconds."))


def evaluate(out: Path, freeze_path: Path, training: Path, holdout: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    frozen = check_freeze(freeze_path)
    config = frozen["config"]
    trained = read_json(training / "training.json")
    if trained["status"] != "completed" or trained["config"] != config or trained["freeze_sha256"] != sha256(freeze_path):
        raise ValueError("Training provenance mismatch")
    require_hashes(training, trained["artifacts"])
    manifest = read_json(holdout / "manifest.json")
    require_hashes(holdout, {entry["path"]: entry["sha256"] for entry in manifest["files"]})
    datasets = {}
    for split in ("main", "e3_size", "e4_tail"):
        path = holdout / split / "instances.jsonl"
        datasets[split] = [dict(instance, split=split) for instance in read_jsonl(path)]
        if len(datasets[split]) != config["holdout"][split]:
            raise ValueError(f"Wrong holdout count: {split}")
    instances = [instance for records in datasets.values() for instance in records]
    if len({i["canonical_hash"] for i in instances}) != len(instances):
        raise ValueError("Duplicate holdout instances")
    models = {seed: Bandit.from_dict(read_json(training / f"seed{seed}.json")["model"]) for seed in config["seeds"]}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    networks = {f"dqn_seed{seed}": load_network(REPO / f"results/dqn-seed{seed}/best.pt") for seed in range(3)}
    for network in networks.values():
        solve_instances(network, instances[:1])
    out.mkdir(parents=True)
    (out / "instances.jsonl").write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in instances), encoding="utf-8")
    rows, references = [], []
    started = time.perf_counter()
    for number, instance in enumerate(instances, 1):
        reference = reference_solve(instance)
        references.append(dict(instance_id=instance["instance_id"], value=reference.value,
                               upper_bound=str(reference.upper_bound), certified=reference.certified,
                               selected_indices=reference.selected_indices, latency_s=reference.runtime_s))

        def record(method, function):
            start = time.perf_counter()
            solution, arm = function()
            elapsed = time.perf_counter() - start
            verify_solution(instance, solution)
            if solution.value > reference.value:
                raise ValueError("Solution exceeds certified optimum")
            rows.append(dict(instance_id=instance["instance_id"], split=instance["split"], method=method,
                value=solution.value, weight_used=solution.weight_used,
                selected_indices=json.dumps(solution.selected_indices), latency_s=elapsed, chosen_arm=arm,
                reference_value=reference.value, upper_bound=float(reference.upper_bound), certified=True))
            return solution

        for trial in range(config["random_trials"]):
            record(f"random_trial{trial}", lambda trial=trial: (
                random_solution(instance, trial_seed(instance["instance_id"], trial, config["random_seed"])), ""))
        for name in [*config["arms"], "dqn_seed1", "dqn_seed2"]:
            record(name, lambda name=name: (solve_arm(name, instance, networks), name))
        name = trained["fixed_best"]
        record("fixed_best", lambda: (solve_arm(name, instance, networks), name))
        for seed, model in models.items():
            record(f"selector_seed{seed}", lambda model=model: decide(instance, model, config, networks))
            record(f"guarded_seed{seed}", lambda model=model: decide(instance, model, config, networks, guarded=True))

        def all_candidates():
            candidates = [(name, solve_arm(name, instance, networks)) for name in config["arms"]]
            best = choose_best(solution for _, solution in candidates)
            name = next(name for name, solution in candidates if solution == best)
            return best, name
        record("portfolio_best", all_candidates)
        if number % 300 == 0:
            print(f"Evaluated {number}/{len(instances)} with exact references: {time.perf_counter()-started:.1f}s", flush=True)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "references.json", {"records": references})
    check_freeze(freeze_path)
    write_json(out / "evaluation.json", dict(status="completed", rows=len(rows), instances=len(instances),
        config=config, config_sha256=sha256(REPO / "portfolio_config.json"), fixed_best=trained["fixed_best"],
        freeze_sha256=sha256(freeze_path), training_sha256=sha256(training / "training.json"),
        holdout_manifest_sha256=sha256(holdout / "manifest.json"), holdout_manifest=manifest,
        holdout_files={f"{s}/instances.jsonl": sha256(holdout / s / "instances.jsonl") for s in datasets},
        elapsed_seconds=time.perf_counter() - started, core_code=core_code(), experiment_code=code_fingerprint(),
        artifacts={name: sha256(out / name) for name in ("results.csv", "instances.jsonl", "references.json")},
        feasibility_violations=0, timing="Warmed CPU, one torch thread, single-instance solver calls including context and solution construction; excludes I/O and DP. Portfolio reruns all eight; guard reruns greedy if needed."))
    print(f"Saved {len(rows)} verified method results to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        if name != "freeze":
            command.add_argument("--freeze", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--training", type=Path, required=True)
            command.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "train":
        train(args.out, args.freeze)
    else:
        evaluate(args.out, args.freeze, args.training, args.holdout)


if __name__ == "__main__":
    main()
