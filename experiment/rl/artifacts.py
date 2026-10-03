"""Read frozen inputs and bind runs to their actual bytes."""
from __future__ import annotations

import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_references(path: Path, required_ids: set[str]) -> dict[str, dict]:
    references = {}
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["instance_id"]
            if key in seen:
                raise ValueError(f"duplicate reference ID: {key}")
            seen.add(key)
            if key not in required_ids:
                continue
            if row["certified"].lower() != "true":
                raise ValueError(f"uncertified reference: {key}")
            value, upper = int(row["value"]), Fraction(row["upper_bound"])
            if value < 0 or upper < value:
                raise ValueError(f"invalid reference bound: {key}")
            references[key] = {"value": value, "upper_bound": upper}
    missing = required_ids - references.keys()
    if missing:
        raise ValueError(f"missing reference IDs: {sorted(missing)[:3]}")
    return references


def check_inputs(
    corpus: Path, manifest_path: Path, reference_path: Path, config_path: Path,
    splits: tuple[str, ...] = ("train", "validation"),
) -> tuple[dict[str, list[dict]], dict[str, dict], dict]:
    """Verify only requested partitions; training never opens test instances."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    facts = {entry["path"]: entry for entry in manifest["files"]}
    data, corpus_hashes, ids = {}, {}, set()
    for split in splits:
        relative = f"{split}/instances.jsonl"
        if relative not in facts:
            raise ValueError(f"partition missing from manifest: {relative}")
        path = Path(corpus) / relative
        digest = sha256(path)
        if digest != facts[relative]["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        with path.open(encoding="utf-8") as handle:
            instances = [json.loads(line) for line in handle if line.strip()]
        if len(instances) != facts[relative]["lines"]:
            raise ValueError(f"line count mismatch: {relative}")
        for instance in instances:
            if instance["instance_id"] in ids:
                raise ValueError(f"duplicate instance ID: {instance['instance_id']}")
            ids.add(instance["instance_id"])
        data[split] = instances
        corpus_hashes[relative] = digest
    references = load_references(reference_path, ids)
    provenance = {
        "manifest_sha256": sha256(manifest_path),
        "reference_sha256": sha256(reference_path),
        "config_sha256": sha256(config_path),
        "corpus_sha256": corpus_hashes,
    }
    return data, references, provenance


def code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*.py"))}


def new_run_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
