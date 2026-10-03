"""Check new timings did not change any frozen baseline decision.

python -m analysis.verify_baselines --new results/formal-baseline-results.csv
    --out results/formal-baseline-verification.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from experiment.baselines.cli import COLUMNS
from experiment.rl.artifacts import sha256


def read(path):
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"unexpected columns: {path}")
        for row in reader:
            key = (row["instance_id"], row["method"], row["trial"])
            if key in rows:
                raise ValueError(f"duplicate baseline result: {key}")
            latency = float(row.pop("latency_s"))
            if not math.isfinite(latency) or latency < 0:
                raise ValueError(f"invalid latency: {key}")
            rows[key] = row
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("baseline_results.csv"))
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    original, new = read(args.original), read(args.new)
    if set(original) != set(new):
        raise ValueError("baseline key coverage changed")
    differences = [key for key in original if original[key] != new[key]]
    if differences:
        raise ValueError(f"baseline logical results changed: {differences[:3]}")
    if len(original) != 36300:
        raise ValueError("expected the frozen 36,300-row baseline")
    summary = {"status": "verified", "rows": len(original), "ignored_columns": ["latency_s"],
        "methods": dict(Counter(row["method"] for row in original.values())),
        "distinct_instances": len({key[0] for key in original}),
        "non_timing_differences": 0, "original_sha256": sha256(args.original),
        "new_sha256": sha256(args.new),
        "note": "Every selected value, weight, violation count, item-set hash, trial and seed matches."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
