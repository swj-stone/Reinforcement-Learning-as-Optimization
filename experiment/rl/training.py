"""Ordinary DQN with balanced replay and validation-only model selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.eval.metrics import feasibility_violations, normalized_quality
from experiment.rl.artifacts import code_fingerprint, new_run_directory, write_json
from experiment.rl.environment import KnapsackEnv
from experiment.rl.model import QNetwork, batch_states, select_action, solve_instances
from experiment.rl.sampling import BalancedReplay, BalancedSampler, Transition


def td_targets(target, transitions, device="cpu", gamma=1.0):
    """Mask before maximizing; terminal transitions never enter the network."""
    with torch.no_grad():
        values = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        indices = [i for i, t in enumerate(transitions) if t.next_state is not None]
        if indices:
            batch = batch_states([transitions[i].next_state for i in indices], device=device)
            q = target(batch)
            if not torch.isfinite(q).all():
                raise ValueError("non-finite target Q values")
            q = q.masked_fill(~batch["action_mask"], -torch.inf)
            values[indices] += gamma * q.max(dim=1).values
        return values


def validation_scores(instances, solutions, references):
    if len(instances) != len(solutions):
        raise ValueError("validation solution count mismatch")
    by_source, by_cell, gaps = defaultdict(list), defaultdict(list), defaultdict(list)
    trivial = 0
    for instance, solution in zip(instances, solutions):
        verify_solution(instance, solution)
        reference = references[instance["instance_id"]]
        optimum = reference["value"]
        if solution.value > optimum:
            raise ValueError("solution exceeds certified optimum")
        if optimum == 0:
            trivial += 1
            continue
        quality = normalized_quality(solution.value, reference["upper_bound"])
        source = instance["source"]
        by_source[source].append(quality)
        cell = f"{source}/{instance['n']}/{instance['alpha']}/{instance['family']}"
        by_cell[cell].append(quality)
        gaps[source].append(100.0 * (optimum - solution.value) / optimum)
    if set(by_source) != {"synthetic", "real"}:
        raise ValueError("validation needs nontrivial instances from both sources")
    means = {key: math.fsum(values) / len(values) for key, values in by_source.items()}
    cells = {key: math.fsum(values) / len(values) for key, values in by_cell.items()}
    cell_macro = {source: math.fsum(value for key, value in cells.items() if key.startswith(source + "/")) /
                  sum(key.startswith(source + "/") for key in cells) for source in means}
    return {"score": (means["synthetic"] + means["real"]) / 2,
            "source_means": means, "cell_means": cells, "trivial_count": trivial,
            "cell_macro_source_means": cell_macro,
            "cell_macro_score": (cell_macro["synthetic"] + cell_macro["real"]) / 2,
            "exact_gap_by_source": {k: math.fsum(v) / len(v) for k, v in gaps.items()}}


def verify_solution(instance, solution):
    if feasibility_violations(instance, solution.selected_indices):
        raise ValueError(f"infeasible RL solution: {instance['instance_id']}")
    if (sum(instance["values"][i] for i in solution.selected_indices) != solution.value
            or sum(instance["weights"][i] for i in solution.selected_indices) != solution.weight_used):
        raise ValueError("RL solution totals do not match selected original indices")


def is_better(score, previous):
    if not math.isfinite(score):
        raise ValueError("non-finite validation score")
    return previous is None or score > previous


def epsilon_at(step, config):
    fraction = min(1.0, step / config["linear_decay_transitions"])
    return config["initial"] + fraction * (config["final"] - config["initial"])


def validate_config(config, transitions, threads):
    b = config["budgets"]
    for key in ("batch_size", "replay_capacity", "update_every_transitions",
                "target_copy_transitions", "validation_every_transitions"):
        if type(b[key]) is not int or b[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    quotas = b["replay_source_balance"]
    if (set(quotas) != {"synthetic", "real"} or
            any(type(q) is not int or q <= 0 for q in quotas.values()) or
            quotas["synthetic"] != quotas["real"] or
            sum(quotas.values()) != b["batch_size"] or
            b["replay_capacity"] < b["batch_size"] or b["replay_capacity"] % 2):
        raise ValueError("replay requires equal source quotas and sufficient even capacity")
    if config["rl_environment"]["gamma"] != 1 or b["optimizer"] != "adam" or b["loss"] != "huber":
        raise ValueError("this implementation requires gamma=1, Adam and Huber loss")
    e = b["epsilon"]
    if not 0 <= e["final"] <= e["initial"] <= 1 or e["linear_decay_transitions"] <= 0:
        raise ValueError("invalid epsilon schedule")
    if (type(transitions) is not int or transitions <= 0 or type(threads) is not int or threads <= 0
            or type(b["warmup_transitions"]) is not int or b["warmup_transitions"] < 0
            or not math.isfinite(b["learning_rate"]) or b["learning_rate"] <= 0
            or not math.isfinite(b["grad_clip_global_norm"]) or b["grad_clip_global_norm"] <= 0):
        raise ValueError("invalid training budget or optimizer parameters")


def seed_runtime(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _save_checkpoint(path, network, seed, step, score, config, provenance, run_kind, run_metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "model": network.state_dict(), "seed": seed,
                "step": step, "validation_score": score, "config": config,
                "provenance": provenance, "run_kind": run_kind,
                "training_status": "in_progress",
                "training": {key: run_metadata[key] for key in
                             ("requested_transitions", "runtime", "code_sha256")}}, temporary)
    temporary.replace(path)


def train(train_instances, validation_instances, references, config, out: Path, *,
          seed: int, transitions: int, provenance: dict, run_kind="custom",
          device="cpu", threads=1):
    validate_config(config, transitions, threads)
    sampler = BalancedSampler(train_instances, seed)
    # Construct and validate inputs before reserving an output directory.
    seed_runtime(seed, threads)
    online = QNetwork().to(device)
    target = deepcopy(online).eval()
    target.requires_grad_(False)
    b = config["budgets"]
    optimizer = torch.optim.Adam(online.parameters(), lr=b["learning_rate"])
    replay = BalancedReplay(b["replay_capacity"], seed + 100)
    policy_rng = random.Random(seed + 200)
    out = new_run_directory(out)
    start = time.perf_counter()
    step = updates = episodes = 0
    seen, source_steps, size_steps, source_episodes = set(), Counter(), Counter(), Counter()
    size_episodes, cell_episodes, size_seen = Counter(), Counter(), defaultdict(set)
    next_validation = b["validation_every_transitions"]
    best_score, best_step, last_validation = None, None, None
    history = []
    summary = {
        "status": "running", "run_kind": run_kind, "seed": seed,
        "requested_transitions": transitions, "config": config, "provenance": provenance,
        "code_sha256": code_fingerprint(), "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": version("numpy"), "PyYAML": version("PyYAML"),
                    "platform": platform.platform(), "processor": platform.processor(),
                    "device": str(device), "threads": threads,
                    "cuda_device": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None},
        "checkpoint_score": "equal-source mean V / stored fractional upper_bound",
        "test_evaluated": False,
    }
    write_json(out / "run.json", summary)

    def validate(writer, handle):
        nonlocal best_score, best_step, last_validation
        tick = time.perf_counter()
        online.eval()
        solutions = []
        for offset in range(0, len(validation_instances), 128):
            solutions.extend(solve_instances(online, validation_instances[offset:offset + 128], device=device))
        scores = validation_scores(validation_instances, solutions, references)
        elapsed = time.perf_counter() - tick
        improved = is_better(scores["score"], best_score)
        if improved:
            best_score, best_step = scores["score"], step
            _save_checkpoint(out / "best.pt", online, seed, step, best_score, config, provenance, run_kind, summary)
        history.append({"step": step, **scores, "validation_seconds": elapsed})
        writer.writerow({"step": step, "score": scores["score"],
                         "synthetic_score": scores["source_means"]["synthetic"],
                         "real_score": scores["source_means"]["real"],
                         "trivial_count": scores["trivial_count"], "new_best": improved,
                         "validation_seconds": elapsed, "elapsed_seconds": time.perf_counter() - start})
        handle.flush()
        write_json(out / "validation_details.json", {"checkpoints": history})
        print(f"seed={seed} step={step} validation_V/U={scores['score']:.6f} best={best_score:.6f}", flush=True)
        last_validation = step
        online.train()

    try:
        with (out / "training.csv").open("w", encoding="utf-8", newline="") as training_file, \
                (out / "validation.csv").open("w", encoding="utf-8", newline="") as validation_file:
            tw = csv.DictWriter(training_file, fieldnames=["step", "loss", "epsilon", "gradient_norm"], lineterminator="\n")
            vw = csv.DictWriter(validation_file, fieldnames=["step", "score", "synthetic_score", "real_score",
                "trivial_count", "new_best", "validation_seconds", "elapsed_seconds"], lineterminator="\n")
            tw.writeheader()
            vw.writeheader()
            validate(vw, validation_file)
            while step < transitions:
                for instance in sampler.next_block():
                    env = KnapsackEnv(instance)
                    seen.add(instance["instance_id"])
                    group = f"{instance['source']}/{instance['n']}"
                    size_seen[group].add(instance["instance_id"])
                    while env.state is not None:
                        state = env.state
                        epsilon = epsilon_at(step, b["epsilon"])
                        action = select_action(online, state, epsilon, policy_rng, device=device)
                        next_state, reward, done = env.step(action)
                        replay.add(Transition(state, action, reward, next_state))
                        step += 1
                        source_steps[instance["source"]] += 1
                        size_steps[f"{instance['source']}/{instance['n']}"] += 1
                        if step >= b["warmup_transitions"] and step % b["update_every_transitions"] == 0 and replay.ready(b["replay_source_balance"]):
                            samples = replay.sample(b["replay_source_balance"])
                            batch = batch_states([t.state for t in samples], device=device)
                            actions = torch.tensor([t.action for t in samples], dtype=torch.long, device=device)
                            predictions = online(batch).gather(1, actions[:, None]).squeeze(1)
                            targets = td_targets(target, samples, device, config["rl_environment"]["gamma"])
                            loss = F.smooth_l1_loss(predictions, targets)
                            if not torch.isfinite(loss):
                                raise ValueError("non-finite DQN loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(online.parameters(), b["grad_clip_global_norm"], error_if_nonfinite=True)
                            optimizer.step()
                            updates += 1
                            tw.writerow({"step": step, "loss": loss.item(), "epsilon": epsilon, "gradient_norm": norm.item()})
                        if step % b["target_copy_transitions"] == 0:
                            target.load_state_dict(online.state_dict())
                    episodes += 1
                    source_episodes[instance["source"]] += 1
                    size_episodes[group] += 1
                    cell_episodes[f"{group}/{instance['family']}/{instance['alpha']}"] += 1
                    if step >= next_validation:
                        validate(vw, validation_file)
                        while next_validation <= step:
                            next_validation += b["validation_every_transitions"]
                training_file.flush()
            if last_validation != step:
                validate(vw, validation_file)
            _save_checkpoint(out / "last.pt", online, seed, step, history[-1]["score"], config, provenance, run_kind, summary)
            # A best checkpoint may be early, even step zero. Completion refers
            # to the entire run, never to the selected model's own step.
            for name in ("best.pt", "last.pt"):
                path = out / name
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                checkpoint["training_status"] = "completed"
                checkpoint["training"]["actual_transitions"] = step
                temporary = path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(path)
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(transitions=step, optimizer_updates=updates, episodes=episodes,
                       unique_instances_seen=len(seen), best_validation_score=best_score,
                       best_step=best_step, elapsed_seconds=time.perf_counter() - start,
                       exposure={"transitions_by_source": dict(source_steps),
                                 "transitions_by_source_size": dict(size_steps),
                                 "episodes_by_source": dict(source_episodes),
                                 "episodes_by_source_size": dict(size_episodes),
                                 "episodes_by_cell": dict(cell_episodes),
                                 "unique_instances_by_source_size": {key: len(value) for key, value in size_seen.items()}})
        write_json(out / "run.json", summary)
    return summary
