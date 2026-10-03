"""Exercise the public command entry points with a self-contained corpus."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiment.rl.cli import main
from tests.test_rl_training import instance, small_config


def write_fixture(tmp_path):
    root = tmp_path / "data"
    facts, refs = [], []
    for split in ("train", "validation", "test", "stress/e3_size", "stress/e4_tail"):
        records = [instance(s, f"{split}-{s}") for s in ("synthetic", "real")]
        file = root / split / "instances.jsonl"
        file.parent.mkdir(parents=True)
        file.write_text("".join(json.dumps(i) + "\n" for i in records), encoding="utf-8")
        facts.append({"path": f"{split}/instances.jsonl", "lines": 2,
                      "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        refs.extend(i["instance_id"] for i in records)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": facts}), encoding="utf-8")
    reference = tmp_path / "references.csv"
    reference.write_text("instance_id,certified,value,upper_bound\n" +
                         "".join(f"{i},True,10,12\n" for i in refs), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(small_config()), encoding="utf-8")
    return ["--corpus", str(root), "--manifest", str(manifest),
            "--references", str(reference), "--config", str(config)]


def test_cli_train_evaluate_and_refuse_overwrite_or_stale_inputs(tmp_path):
    common = write_fixture(tmp_path)
    out = tmp_path / "run"
    assert main(["check-data", *common]) == 0
    assert main(["train", *common, "--out", str(out), "--transitions", "12"]) == 0
    assert main(["train", *common, "--out", str(out), "--transitions", "12"]) == 1
    checkpoint = out / "best.pt"
    target = tmp_path / "evaluation"
    # A diagnostic/custom checkpoint must never pass as the formal experiment.
    args = ["evaluate", *common, "--checkpoint", str(checkpoint), "--out", str(target)]
    assert main(args) == 1
    assert main([*args, "--allow-diagnostic"]) == 0
    with (target / "rl_results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {r["split"] for r in rows} == {"test", "e3_size", "e4_tail"}
    assert all(r["method"] == "dqn" and int(r["violations"]) == 0 for r in rows)
    assert all(int(r["value"]) <= 10 for r in rows)
    assert all(len(r["checkpoint_sha256"]) == 64 for r in rows)
    assert main([*args, "--allow-diagnostic"]) == 1
    interrupted = torch.load(checkpoint, weights_only=True)
    interrupted["run_kind"] = "full"
    interrupted["training_status"] = "in_progress"
    interrupted_path = tmp_path / "interrupted.pt"
    torch.save(interrupted, interrupted_path)
    interrupted_args = ["evaluate", *common, "--checkpoint", str(interrupted_path),
                        "--out", str(tmp_path / "interrupted-evaluation")]
    assert main(interrupted_args) == 1
    assert main([*interrupted_args, "--allow-diagnostic"]) == 0
    metadata = json.loads((tmp_path / "interrupted-evaluation/evaluation.json").read_text())
    assert metadata["run_kind"] == "incomplete"
    config = Path(common[-1])
    config.write_text(config.read_text() + "\n# changed\n", encoding="utf-8")
    assert main(["evaluate", *common, "--checkpoint", str(checkpoint),
                 "--out", str(tmp_path / "stale"), "--allow-diagnostic"]) == 1
    assert not (tmp_path / "stale").exists()
