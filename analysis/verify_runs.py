"""Independently verify the three completed formal runs before held-out evaluation.

Usage: python -m analysis.verify_runs --out results/formal-run-verification.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import torch
import yaml

from experiment.rl.artifacts import check_inputs, code_fingerprint, sha256
from experiment.rl.model import QNetwork, solve_instances
from experiment.rl.training import seed_runtime, validation_scores


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=Path("results/formal-experiment-freeze.json"))
    parser.add_argument("--runs", type=Path, nargs=3,
                        default=[Path(f"results/dqn-seed{s}") for s in range(3)])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("experiment_config.yaml").read_text(encoding="utf-8"))
    assert freeze["budget_per_seed"] == config["budgets"]["training_transitions"] == 300000
    assert freeze["seeds"] == config["budgets"]["training_seeds"] == [0, 1, 2]
    assert freeze["code_sha256"] == code_fingerprint(), "experiment source changed after freeze"
    assert all(sha256(Path(name)) == value for name, value in freeze["input_sha256"].items())
    data, references, provenance = check_inputs(Path("datasets/processed"), Path("data_manifest.json"),
        Path("reference_results.csv"), Path("experiment_config.yaml"))
    assert provenance == freeze["provenance"]
    records = []
    for seed, path in zip(freeze["seeds"], args.runs, strict=True):
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        assert run["seed"] == seed and run["status"] == "completed" and run["run_kind"] == "full"
        assert datetime.fromisoformat(run["started_at"]) >= datetime.fromisoformat(freeze["frozen_at_utc"])
        assert run["requested_transitions"] == 300000 and run["transitions"] == 300440
        assert run["code_sha256"] == freeze["code_sha256"] and run["provenance"] == provenance
        assert run["config"] == config
        assert run["episodes"] == 3248 and run["optimizer_updates"] == 73861
        exposure = run["exposure"]
        assert exposure["transitions_by_source"] == {"synthetic": 150220, "real": 150220}
        assert exposure["episodes_by_source"] == {"synthetic": 1624, "real": 1624}
        assert set(exposure["episodes_by_source_size"].values()) == {406}
        assert exposure["transitions_by_source_size"] == {
            f"{source}/{n}": 406 * n for source in ("synthetic", "real") for n in (20, 50, 100, 200)}
        assert sum(exposure["unique_instances_by_source_size"].values()) == run["unique_instances_seen"]
        validation = read_csv(path / "validation.csv")
        steps = [int(r["step"]) for r in validation]
        assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == 300440
        best = max(validation, key=lambda row: float(row["score"]))  # first maximum wins
        assert run["best_step"] == int(best["step"])
        assert run["best_validation_score"] == float(best["score"])
        losses = read_csv(path / "training.csv")
        assert len(losses) == run["optimizer_updates"]
        assert [int(r["step"]) for r in losses] == list(range(5000, 300441, 4))
        for row in losses:
            assert all(math.isfinite(float(row[k])) for k in ("loss", "epsilon", "gradient_norm"))
            assert float(row["loss"]) >= 0 and float(row["gradient_norm"]) >= 0
        checkpoints = {}
        for name in ("best.pt", "last.pt"):
            checkpoint = torch.load(path / name, map_location="cpu", weights_only=True)
            assert checkpoint["training_status"] == "completed" and checkpoint["run_kind"] == "full"
            assert checkpoint["seed"] == seed and checkpoint["provenance"] == provenance
            assert checkpoint["config"] == config
            assert checkpoint["training"]["actual_transitions"] == 300440
            assert checkpoint["training"]["requested_transitions"] == 300000
            assert checkpoint["training"]["code_sha256"] == freeze["code_sha256"]
            assert checkpoint["step"] == (run["best_step"] if name == "best.pt" else 300440)
            checkpoints[name] = sha256(path / name)
        checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        seed_runtime(seed, threads=1)
        network = QNetwork().eval()
        network.load_state_dict(checkpoint["model"])
        # Match training's outer batches exactly: different matrix batch sizes
        # can otherwise perturb an action near a floating-point Q-value tie.
        solutions = []
        for offset in range(0, len(data["validation"]), 128):
            solutions.extend(solve_instances(network, data["validation"][offset:offset + 128], device="cpu"))
        scores = validation_scores(data["validation"], solutions, references)
        assert math.isclose(scores["score"], run["best_validation_score"], rel_tol=0, abs_tol=1e-12)
        records.append({"seed": seed, "path": path.as_posix(), "checkpoint_sha256": checkpoints,
            "run_sha256": sha256(path / "run.json"), "best_step": run["best_step"],
            "reproduced_validation_score": scores["score"], "validation_instances_verified": len(solutions),
            "source_validation_scores": scores["source_means"],
            "training_loss_rows_finite": len(losses), "transitions": run["transitions"],
            "episodes": run["episodes"], "unique_instances_seen": run["unique_instances_seen"],
            "concurrent_elapsed_seconds": run["elapsed_seconds"], "exposure": exposure})
        print(f"Verified seed {seed}: {run['transitions']} transitions, best step {run['best_step']}, "
              f"reproduced validation V/U={scores['score']:.9f}", flush=True)
    record = {"status": "verified", "freeze_sha256": sha256(args.freeze),
        "provenance": provenance, "runs": records,
        "note": "All checks run before formal held-out evaluation. Validation inference only; no tuning."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
