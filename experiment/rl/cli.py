"""Run with ``python -m experiment.rl.cli {check-data,train,evaluate}``."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
import yaml

from experiment.baselines.cli import COLUMNS
from experiment.data.spec import DATASETS_PROCESSED, REPO_ROOT
from experiment.eval.reference import item_set_hash
from experiment.rl.artifacts import check_inputs, code_fingerprint, new_run_directory, sha256, write_json
from experiment.rl.model import QNetwork, solve_instances
from experiment.rl.training import seed_runtime, train, verify_solution

EVALUATION_SPLITS = ("test", "stress/e3_size", "stress/e4_tail")
EXTRA_COLUMNS = ("selected_indices", "checkpoint_sha256", "checkpoint_step", "run_kind")


def build_parser():
    parser = argparse.ArgumentParser(description="Frozen-corpus DQN experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check-data", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--corpus", type=Path, default=DATASETS_PROCESSED)
        command.add_argument("--manifest", type=Path, default=REPO_ROOT / "data_manifest.json")
        command.add_argument("--references", type=Path, default=REPO_ROOT / "reference_results.csv")
        command.add_argument("--config", type=Path, default=REPO_ROOT / "experiment_config.yaml")
        if name != "check-data":
            command.add_argument("--out", required=True, type=Path, help="New output directory; never overwritten")
            command.add_argument("--device", default="cpu")
            command.add_argument("--threads", type=int, default=1)
        if name == "train":
            command.add_argument("--seed", type=int, default=0)
            budget = command.add_mutually_exclusive_group()
            budget.add_argument("--pilot", action="store_true")
            budget.add_argument("--transitions", type=int, help="Recorded custom diagnostic budget")
        elif name == "evaluate":
            command.add_argument("--checkpoint", required=True, type=Path)
            command.add_argument("--allow-diagnostic", action="store_true",
                                 help="Explicitly evaluate a pilot/custom checkpoint, retaining its label")
    return parser


def evaluate(args, config):
    # Only load files created by this project. weights_only avoids arbitrary pickle code.
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    completed = (checkpoint.get("training_status") == "completed" and
                 checkpoint["training"].get("actual_transitions", -1) >=
                 checkpoint["training"]["requested_transitions"])
    kind = checkpoint["run_kind"] if completed else "incomplete"
    if kind != "full" and not args.allow_diagnostic:
        raise ValueError("pilot/custom/incomplete checkpoint: use --allow-diagnostic for a labelled diagnostic evaluation")
    data, references, provenance = check_inputs(args.corpus, args.manifest, args.references,
                                               args.config, EVALUATION_SPLITS)
    for key in ("manifest_sha256", "reference_sha256", "config_sha256"):
        if checkpoint["provenance"].get(key) != provenance[key]:
            raise ValueError(f"checkpoint input mismatch: {key}")
    if checkpoint["config"] != config:
        raise ValueError("checkpoint effective configuration differs")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    seed_runtime(checkpoint["seed"], args.threads)
    network = QNetwork().to(args.device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    out = new_run_directory(args.out)
    digest = sha256(args.checkpoint)
    rows = []
    # Warm-up is excluded, and every timed call still includes its own item encoding.
    first = next((i for records in data.values() for i in records), None)
    if first is not None:
        solve_instances(network, [first], device=args.device)
    for split, instances in data.items():
        for instance in instances:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            solution = solve_instances(network, [instance], device=args.device)[0]
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - start
            verify_solution(instance, solution)
            if solution.value > references[instance["instance_id"]]["value"]:
                raise ValueError("RL solution exceeds certified optimum")
            rows.append({"instance_id": instance["instance_id"], "split": split.split("/")[-1],
                         "method": "dqn", "trial": "", "seed": checkpoint["seed"],
                         "value": solution.value, "weight_used": solution.weight_used,
                         "violations": 0, "item_set_hash": item_set_hash(solution.selected_indices),
                         "latency_s": f"{elapsed:.9f}",
                         "selected_indices": json.dumps(list(solution.selected_indices), separators=(",", ":")),
                         "checkpoint_sha256": digest, "checkpoint_step": checkpoint["step"],
                         "run_kind": kind})
    rows.sort(key=lambda row: row["instance_id"])
    with (out / "rl_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*COLUMNS, *EXTRA_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "evaluation.json", {"rows": len(rows), "checkpoint_sha256": digest,
        "checkpoint_step": checkpoint["step"], "run_kind": kind,
        "seed": checkpoint["seed"], "provenance": provenance,
        "training": checkpoint["training"], "evaluation_code_sha256": code_fingerprint(),
        "device": args.device, "threads": args.threads,
        "latency_scope": "single instance including encoding and solution construction; excludes I/O; warmed-up"})
    print(f"Saved {len(rows)} labelled {kind} RL rows to {out}", flush=True)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command != "check-data" and args.out.exists():
            raise FileExistsError(f"output already exists: {args.out}")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.command == "evaluate":
            evaluate(args, config)
        else:
            splits = ("train", "validation") if args.command == "train" else ("train", "validation", *EVALUATION_SPLITS)
            data, references, provenance = check_inputs(args.corpus, args.manifest, args.references, args.config, splits)
            if args.command == "check-data":
                print(json.dumps({"counts": {k: len(v) for k, v in data.items()}, "provenance": provenance}, indent=2))
            else:
                if args.seed not in config["budgets"]["training_seeds"]:
                    raise ValueError("seed must be one of the configured training_seeds")
                budget = args.transitions if args.transitions is not None else config["budgets"][
                    "pilot_transitions" if args.pilot else "training_transitions"]
                kind = "custom" if args.transitions is not None else "pilot" if args.pilot else "full"
                summary = train(data["train"], data["validation"], references, config, args.out,
                                seed=args.seed, transitions=budget, provenance=provenance,
                                run_kind=kind, device=args.device, threads=args.threads)
                print(json.dumps({k: summary[k] for k in ("status", "transitions", "best_step", "best_validation_score", "elapsed_seconds")}))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"RL command failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
