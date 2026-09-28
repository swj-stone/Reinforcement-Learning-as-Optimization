"""两个协议产物的字段、真实性与格式。

`upstream_provenance` 与两个写出函数已在 Task 5 交付；本任务补 manifest 的
派生段与完整的 config。
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiment.data import spec
from experiment.data.assemble import manifest as M


def _tmp(tmp_path: Path) -> dict:
    """一份最小但完整的 manifest 输入。"""
    corpus = tmp_path / "processed" / "train"
    corpus.mkdir(parents=True, exist_ok=True)
    path = corpus / "instances.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    return dict(
        cells=[{"alpha": 0.25, "n": 20, "real": 500, "split": "train", "synthetic": 500}],
        counts={"train": {"real": 6000, "synthetic": 6000, "total": 12000}},
        dedup={
            "collisions_resampled": 0,
            "distinct_canonical_hashes": 17400,
            "max_attempt_used": 0,
            "total_instances": 17400,
        },
        files=[
            {
                "bytes": 16,
                "lines": 2,
                "path": "train/instances.jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
        generated_at="2026-09-28T00:00:00Z",
        lot_index=None,
        out_dir=tmp_path / "processed",
        provenance={"stub": True},
        schema_version=1,
        stress={"e3_size": {"real": 300, "synthetic": 300}, "e4_tail": {"real": 0, "synthetic": 300}},
        tail={
            "basis": "train-split lots only",
            "quantile": 0.99,
            "quantity_w": "1234",
            "unit_price": "5.00",
        },
    )


# --- manifest 的字段 --------------------------------------------------------


def test_manifest_has_the_sections_the_plan_requires(tmp_path):
    """§10 要求 manifest 载有源校验和、清洗规则、日期、种子、计数、单位、
    源/组 ID 与切分/哈希审计。"""
    man = M.build_manifest(**_tmp(tmp_path))
    assert set(man) >= {
        "schema_version", "generated_at", "provenance", "split_policy",
        "splits", "streams", "cells", "counts", "stress", "tail_thresholds",
        "dedup", "files", "output_dir",
    }


def test_manifest_carries_the_provenance_it_was_given(tmp_path):
    man = M.build_manifest(**_tmp(tmp_path))
    assert man["provenance"] == {"stub": True}


def test_streams_record_both_the_key_and_the_declared_master_seed(tmp_path):
    """§8 的 4001/5001 是声明标识，真正的隔离是字符串记号。两者都要记录，
    这个关系才看得出来。"""
    streams = M.build_manifest(**_tmp(tmp_path))["streams"]
    assert streams["size"]["seed_key"]["split"] == spec.STRESS_SIZE_STREAM
    assert streams["size"]["declared_master_seed"] == 4001
    assert streams["size"]["dataset"] == spec.E3_NAME
    assert streams["tail"]["seed_key"]["split"] == spec.STRESS_TAIL_STREAM
    assert streams["tail"]["seed_key"]["family"] == spec.E4_SEED_TOKEN
    assert streams["tail"]["declared_master_seed"] == 5001
    assert streams["core"]["seed_key"]["family"] == spec.REAL_SEED_TOKEN


def test_tail_thresholds_are_carried_through_unchanged(tmp_path):
    facts = _tmp(tmp_path)
    assert M.build_manifest(**facts)["tail_thresholds"] == facts["tail"]


def test_manifest_is_json_serialisable(tmp_path):
    json.dumps(M.build_manifest(**_tmp(tmp_path)), sort_keys=True, ensure_ascii=False)


def test_write_manifest_uses_lf_and_no_bom(tmp_path):
    path = tmp_path / "data_manifest.json"
    M.write_manifest(path, M.build_manifest(**_tmp(tmp_path)))
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["schema_version"] == 1


def test_write_manifest_is_byte_stable_for_the_same_input(tmp_path):
    man = M.build_manifest(**_tmp(tmp_path))
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    M.write_manifest(a, man)
    M.write_manifest(b, man)
    assert a.read_bytes() == b.read_bytes()


# --- config 的字段 ----------------------------------------------------------


def _config(tmp_path: Path, **kwargs) -> dict:
    kwargs.setdefault("requirements", tmp_path / "requirements.txt")
    if not kwargs["requirements"].is_file():
        kwargs["requirements"].write_text("numpy==2.5.3\n", encoding="utf-8")
    return M.build_config(out_dir=tmp_path, generated_at="2026-09-28T00:00:00Z", **kwargs)


def test_config_carries_the_sections_the_plan_requires(tmp_path):
    """§10：生成器公式、配额、RL 设置、预算、评估规则、固定包版本。"""
    doc = _config(tmp_path)
    assert set(doc) >= {
        "schema_version", "generated_at", "generators", "cap_ratios", "sizes",
        "quotas", "real", "stress", "budgets", "evaluation", "packages", "runtime",
    }


def test_config_packages_are_parsed_from_requirements_not_hardcoded(tmp_path):
    """硬编码的版本会在依赖升级后悄悄说谎。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==9.9.9\n\n# 注释\npandas==1.2.3\n", encoding="utf-8")
    doc = _config(tmp_path, requirements=req)
    assert doc["packages"]["numpy"] == "9.9.9"
    assert doc["packages"]["pandas"] == "1.2.3"
    assert "注释" not in doc["packages"]


def test_config_packages_follow_an_edited_requirements_file(tmp_path):
    """改一个版本号，产物必须跟着变——证明它真是解析出来的。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.1.1\n", encoding="utf-8")
    assert _config(tmp_path, requirements=req)["packages"]["numpy"] == "1.1.1"
    req.write_text("numpy==2.2.2\n", encoding="utf-8")
    assert _config(tmp_path, requirements=req)["packages"]["numpy"] == "2.2.2"


def test_config_rejects_an_unpinned_requirement(tmp_path):
    """半钉住的依赖就是复现性漏洞，宁可报错也不静默记录。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy>=1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numpy"):
        M.build_config(out_dir=tmp_path, generated_at="x", requirements=req)


def test_config_records_the_frozen_quotas_and_sizes(tmp_path):
    doc = _config(tmp_path)
    assert doc["quotas"]["synthetic"] == dict(spec.SYNTHETIC_QUOTA)
    assert doc["quotas"]["real"] == dict(spec.REAL_QUOTA)
    assert doc["sizes"] == list(spec.SIZES)
    assert doc["cap_ratios"] == list(spec.ALPHAS)
    assert doc["real"]["split_fractions"] == list(spec.SPLIT_DATE_FRACTIONS)


def test_config_e4_generator_uses_the_variant_parameters(tmp_path):
    doc = _config(tmp_path)
    assert doc["generators"]["skewed_e4"]["k"] == spec.E4_SKEW_K
    assert doc["generators"]["skewed_e4"]["heavy_weight"] == list(
        spec.E4_SKEW_HEAVY_WEIGHT_RANGE
    )
    assert doc["generators"]["skewed_e4"]["heavy_density"] == list(
        spec.E4_SKEW_HEAVY_DENSITY_RANGE
    )


def test_config_records_the_frozen_families_unchanged(tmp_path):
    """主语料的 skewed 参数必须与 spec 一致——E4 是变体，不是改版。"""
    skewed = _config(tmp_path)["generators"]["skewed"]
    assert skewed["heavy_weight"] == list(spec.SKEW_HEAVY_WEIGHT_RANGE)
    assert skewed["heavy_density"] == list(spec.SKEW_HEAVY_DENSITY_RANGE)
    assert skewed["base_weight"] == list(spec.SKEW_BASE_WEIGHT_RANGE)
    assert skewed["base_density"] == list(spec.SKEW_BASE_DENSITY_RANGE)


def test_config_records_the_training_budget_and_evaluation_rules(tmp_path):
    doc = _config(tmp_path)
    assert doc["budgets"]["training_transitions"] == 300000
    assert doc["budgets"]["training_seeds"] == [0, 1, 2]
    assert doc["evaluation"]["methods"] == ["random", "greedy", "dqn"]
    assert doc["evaluation"]["random_trials_per_instance"] == 10
    assert doc["evaluation"]["evaluation_seed"] == 6001
    assert doc["evaluation"]["bootstrap"]["resamples"] == 1000


def test_write_config_roundtrips_as_yaml(tmp_path):
    doc = _config(tmp_path)
    path = tmp_path / "experiment_config.yaml"
    M.write_config(path, doc)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert yaml.safe_load(raw.decode("utf-8")) == doc


def test_write_config_is_byte_stable_and_sorts_keys(tmp_path):
    doc = _config(tmp_path)
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    M.write_config(a, doc)
    M.write_config(b, doc)
    assert a.read_bytes() == b.read_bytes()
    text = a.read_text(encoding="utf-8")
    assert text.index("cap_ratios") < text.index("generators") < text.index("packages")


def test_write_config_keeps_chinese_readable(tmp_path):
    """allow_unicode=True：中文标签不该被转义成 \\uXXXX 而变得不可读。"""
    path = tmp_path / "c.yaml"
    M.write_config(path, {"note": "训练预算"})
    assert "训练预算" in path.read_text(encoding="utf-8")
