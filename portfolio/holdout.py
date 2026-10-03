"""Construct a fresh instance holdout without changing the original experiment.

Run only after the follow-up configuration has been frozen::

    python -m portfolio.holdout --out results/portfolio-v2-holdout

New knapsack combinations are disjoint from every original instance. Real lots
still come from the original test-date pools: this is not a new business sample.
This module does not train policies, evaluate methods, or compute reference values.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from experiment.data.assemble.lots import load_index
from experiment.data.assemble.real import build_real_cell, fill_cell
from experiment.data.assemble.stress import generate_skewed_e4
from experiment.data.generate.corpus import build_instance, write_cell
from experiment.data.generate.families import capacity_for, compute_stats
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import ALPHAS, E4_SEED_TOKEN, FAMILIES, SIZES, alpha_token


ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = {"main": "portfolio_v2_test", "e3_size": "portfolio_v2_test/e3",
              "e4_tail": "portfolio_v2_test/e4"}
DEFAULT_QUOTAS = {"main_synthetic_per_cell": 25, "main_real_per_cell": 100,
                  "e3_synthetic_per_cell": 25, "e3_real_per_cell": 100,
                  "e4_per_alpha": 100}
ORIGINAL_FILES = ("train/instances.jsonl", "validation/instances.jsonl", "test/instances.jsonl",
                  "stress/e3_size/instances.jsonl", "stress/e4_tail/instances.jsonl")


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _empty_groups():
    return {"source_rows": set(), "source_dates": set(), "invoices": set()}


def _add_record(groups, record):
    if record["source"] == "real":
        groups["source_rows"].update(record["source_row_ids"])
        groups["source_dates"].add(record["source_date"])
        groups["invoices"].update(record["invoice_nos"])


def _original_inputs(repo):
    """Read all five certified corpus files, checking their frozen byte hashes."""
    manifest_path = repo / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    facts = {entry["path"]: entry for entry in manifest["files"]}
    if len(facts) != len(manifest["files"]) or set(facts) != set(ORIGINAL_FILES):
        raise ValueError("original manifest must contain exactly the five frozen corpus files")
    hashes, seen, ids, groups = {}, set(), set(), {}
    total = 0
    for relative in ORIGINAL_FILES:
        path = repo / "datasets/processed" / relative
        digest = _sha256(path)
        if digest != facts[relative]["sha256"]:
            raise ValueError(f"original corpus SHA-256 mismatch: {relative}")
        hashes[relative] = digest
        label = Path(relative).parts[-2]
        groups[label] = _empty_groups()
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["canonical_hash"] in seen or record["instance_id"] in ids:
                    raise ValueError("original corpus contains duplicate instance IDs or canonical hashes")
                seen.add(record["canonical_hash"])
                ids.add(record["instance_id"])
                _add_record(groups[label], record)
                count += 1
        if count != facts[relative]["lines"]:
            raise ValueError(f"original corpus line count mismatch: {relative}")
        total += count

    lot_fact = manifest["provenance"]["lots_clean"]
    if lot_fact["path"] != "datasets/raw/real/lots_clean.csv":
        raise ValueError("unexpected original cleaned-lots path")
    lots_path = repo / lot_fact["path"]
    lots_hash = _sha256(lots_path)
    if lots_hash != lot_fact["sha256"]:
        raise ValueError("original cleaned-lots SHA-256 mismatch")
    fractions = tuple(manifest["split_policy"]["fractions"])
    if fractions != (.7, .85):
        raise ValueError("original chronological split fractions changed")
    index = load_index(lots_path, fractions=fractions)
    if len(index.lots) != lot_fact["rows"]:
        raise ValueError("original cleaned-lots row count mismatch")
    if len({lot.source_row_id for lot in index.lots}) != len(index.lots):
        raise ValueError("original cleaned lots have duplicate source-row IDs")
    for n in (*SIZES, 500):
        if not index.eligible("test", n):
            raise ValueError(f"no original test-source date can supply n={n} distinct lots")
    provenance = {"original_manifest_sha256": _sha256(manifest_path),
                  "original_corpus_sha256": hashes, "lots_clean_sha256": lots_hash,
                  "lots_clean_path": lot_fact["path"]}
    return manifest, index, seen, ids, groups, total, provenance


def _synthetic(suite, family, n, alpha, instance_index, attempt):
    stream = NAMESPACES[suite]
    if suite != "e4_tail":
        result = build_instance(stream, family, n, alpha, instance_index, attempt)
    else:
        seed = derive_seed(stream, family, n, alpha, instance_index, attempt, "items")
        weights, values = generate_skewed_e4(n, seed)
        capacity = capacity_for(alpha, weights)
        order_seed = derive_seed(stream, family, n, alpha, instance_index, attempt, "order")
        result = {"alpha": alpha, "canonical_hash": canonical_hash(n, capacity, weights, values),
                  "capacity": capacity, "family": family, "n": n, "seed": seed,
                  "item_order": np.random.default_rng(order_seed).permutation(n).tolist(),
                  "resample_attempt": attempt, "source": "synthetic",
                  "stats": compute_stats(weights, values),
                  "weights": weights.tolist(), "values": values.tolist()}
    name = stream.replace("/", "_")
    result["instance_id"] = f"syn-{name}-{family}-n{n}-{alpha_token(alpha)}-i{instance_index:04d}"
    result.update(split=suite, generation_namespace=stream)
    return result


def _source_pools(index):
    groups = {}
    for split, dates in index.pools.items():
        group = _empty_groups()
        for day, lots in dates.items():
            group["source_dates"].add(day)
            group["source_rows"].update(lot.source_row_id for lot in lots)
            group["invoices"].update(lot.invoice_no for lot in lots)
        groups[split] = group
    return groups


def _overlap(records, original_groups, source_pools):
    own = _empty_groups()
    for record in records:
        _add_record(own, record)
    result = {"unique_source_rows": len(own["source_rows"]),
              "unique_source_dates": len(own["source_dates"]), "unique_invoices": len(own["invoices"])}
    for label, reference in (("original_instances", original_groups), ("original_source_pools", source_pools)):
        result[label] = {split: {field: len(own[field] & group[field]) for field in own}
                         for split, group in reference.items()}
        for split in ("train", "validation"):
            if any(result[label][split].values()):
                raise ValueError(f"follow-up real data overlap original {split}: {label}")
    if any(own[field] - source_pools["test"][field] for field in own):
        raise ValueError("follow-up real data include records outside original test-source pools")
    return result


def build_holdout(out: Path, repo: Path, *, quotas: dict | None = None) -> dict:
    """Save deterministic new combinations; refuse any already existing output.

    Default counts are 2,400 main, 600 E3 and 300 E4 instances. Optional per-cell
    quotas support small diagnostic corpora; non-default quotas are labelled as
    diagnostic in the manifest and must retain the 50:50 source balance.
    No output is created until inputs, generation, and overlap audits succeed.
    """
    out, repo = Path(out), Path(repo)
    if out.exists():
        raise FileExistsError(f"holdout output already exists: {out}")
    effective = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    if set(effective) != set(DEFAULT_QUOTAS) or any(type(value) is not int or value <= 0 for value in effective.values()):
        raise ValueError("quotas require all five declared positive integer per-cell counts")
    for prefix in ("main", "e3"):
        if effective[f"{prefix}_real_per_cell"] != len(FAMILIES) * effective[f"{prefix}_synthetic_per_cell"]:
            raise ValueError(f"{prefix} quotas must preserve equal synthetic/real instance counts")
    original_manifest, index, seen, old_ids, original_groups, original_count, provenance = _original_inputs(repo)
    original_hashes = frozenset(seen)
    data = {suite: [] for suite in NAMESPACES}
    for suite, sizes, prefix in (("main", SIZES, "main"), ("e3_size", (500,), "e3")):
        for family in FAMILIES:
            for n in sizes:
                for alpha in ALPHAS:
                    for instance_index in range(effective[f"{prefix}_synthetic_per_cell"]):
                        data[suite].extend(fill_cell(
                            lambda attempt, f=family, n=n, a=alpha, i=instance_index: _synthetic(suite, f, n, a, i, attempt),
                            quota=1, seen=seen))
        for n in sizes:
            for alpha in ALPHAS:
                records = build_real_cell(index, NAMESPACES[suite].replace("/", "_"), "test", n, alpha,
                                          effective[f"{prefix}_real_per_cell"], seen,
                                          seed_stream=NAMESPACES[suite])
                for record in records:
                    record.update(split=suite, generation_namespace=NAMESPACES[suite])
                data[suite].extend(records)
    for alpha in ALPHAS:
        for instance_index in range(effective["e4_per_alpha"]):
            data["e4_tail"].extend(fill_cell(
                lambda attempt, a=alpha, i=instance_index: _synthetic("e4_tail", E4_SEED_TOKEN, 200, a, i, attempt),
                quota=1, seen=seen))
    all_records = [record for records in data.values() for record in records]
    new_hashes = {record["canonical_hash"] for record in all_records}
    new_ids = {record["instance_id"] for record in all_records}
    if (len(new_hashes) != len(all_records) or new_hashes & original_hashes
            or len(new_ids) != len(all_records) or new_ids & old_ids):
        raise ValueError("holdout instance identities are not disjoint")
    source_pools = _source_pools(index)
    overlaps = {suite: _overlap(records, original_groups, source_pools) for suite, records in data.items()}
    overlaps["all_new_real"] = _overlap(all_records, original_groups, source_pools)
    counts, cells = {}, []
    for suite, records in data.items():
        source_counts = Counter(r["source"] for r in records)
        counts[suite] = {"synthetic": source_counts["synthetic"], "real": source_counts["real"], "total": len(records)}
        grouped = Counter((r["source"], r["family"] or "real", r["n"], r["alpha"]) for r in records)
        cells.extend(dict(split=suite, source=src, family=fam, n=n, alpha=alpha, count=count)
                     for (src, fam, n, alpha), count in sorted(grouped.items()))
    provenance.update(python=platform.python_version(), numpy=np.__version__,
        builder_sha256=_sha256(Path(__file__)),
        generator_code_sha256={p.relative_to(ROOT).as_posix(): _sha256(p)
                               for p in sorted((ROOT / "experiment/data").rglob("*.py"))})
    manifest = {"schema_version": 1, "namespaces": dict(NAMESPACES), "quotas": effective,
        "run_kind": "full" if effective == DEFAULT_QUOTAS else "diagnostic",
        "counts": counts, "cells": cells, "provenance": provenance,
        "original_test_already_inspected": True, "independent_business_sample": False,
        "dependence_notice": "New canonical knapsack combinations reuse the original test-source dates and may reuse source rows/invoices. This is an instance holdout, not an independent business sample.",
        "source_pool": "original test partition only; whole-invoice earliest-date 70/15/15 chronological split",
        "eligible_test_dates": {str(n): len(index.eligible("test", n)) for n in (*SIZES, 500)},
        "tail_thresholds": original_manifest["tail_thresholds"], "overlap": overlaps,
        "dedup": {"original_instances": original_count, "new_instances": len(all_records),
                  "distinct_new_canonical_hashes": len(new_hashes), "original_canonical_overlap": 0,
                  "collisions_resampled": sum(r["resample_attempt"] for r in all_records),
                  "max_resample_attempt": max(r["resample_attempt"] for r in all_records)},
        "files": []}
    # mkdir is the final no-overwrite gate, including a concurrently created path.
    out.mkdir(parents=True, exist_ok=False)
    for suite, records in data.items():
        relative = f"{suite}/instances.jsonl"
        digest = write_cell(out / relative, records)
        manifest["files"].append({"path": relative, "sha256": digest, "lines": len(records),
                                  "bytes": (out / relative).stat().st_size})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = build_holdout(args.out, args.repo)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as error:
        print(f"Holdout construction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out), "counts": manifest["counts"],
                      "independent_business_sample": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
