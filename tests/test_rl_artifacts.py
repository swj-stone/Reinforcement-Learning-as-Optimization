"""Provenance failures must stop a run before training."""
import hashlib
import json

import pytest

from experiment.rl.artifacts import check_inputs, load_references, new_run_directory


def fixture_files(tmp_path):
    root = tmp_path / "corpus"
    files = []
    for split in ("train", "validation"):
        path = root / split / "instances.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"instance_id": split}) + "\n", encoding="utf-8")
        files.append({"path": f"{split}/instances.jsonl", "lines": 1,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    refs = tmp_path / "refs.csv"
    refs.write_text("instance_id,certified,value,upper_bound\ntrain,True,10,12\nvalidation,True,10,12\n", encoding="utf-8")
    return root, manifest, refs, config


def test_manifest_hash_and_line_count_are_verified(tmp_path):
    args = fixture_files(tmp_path)
    data, refs, provenance = check_inputs(*args)
    assert set(data) == {"train", "validation"}
    assert refs["validation"]["upper_bound"] == 12
    assert len(provenance["manifest_sha256"]) == 64
    (args[0] / "train/instances.jsonl").write_text('{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        check_inputs(*args)


def test_reference_missing_duplicate_and_invalid_bounds_rejected(tmp_path):
    args = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        load_references(args[2], {"absent"})
    with args[2].open("a", encoding="utf-8") as f:
        f.write("validation,True,10,12\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_references(args[2], {"validation"})
    args[2].write_text("instance_id,certified,value,upper_bound\nx,True,10,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bound"):
        load_references(args[2], {"x"})


def test_run_directory_never_overwrites_even_an_empty_existing_dir(tmp_path):
    out = tmp_path / "run"
    new_run_directory(out)
    marker = out / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        new_run_directory(out)
    assert marker.read_text() == "keep"
