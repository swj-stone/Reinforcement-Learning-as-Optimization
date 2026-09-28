"""金样本：钉住装配产物的**派生内容**与两个协议产物的**字节格式**。

为什么单独要一个模块。逐任务的审查反复测出同一种盲区：身份、计数、50:50、
种子、流隔离都有断言，而

* **派生内容**——切分边界、每 split 的合格日期与实测占比、每个语料文件的
  sha256/行数/字节数——一条断言都没有；改一个取整方向、少算一个 split，
  整套测试依然全绿；
* **入库产物的字节**——缩进、键序、行尾、BOM——同样无人过问。

`golden/assemble_sections.json` 由一次真实装配生成、人工核对后入库，本模块
逐段比对。它**不提供"缺失即生成"的回退**：那种回退会让金样本在文件丢失时
静默通过，而本项目的验收要求「全绿且无 skipped」（设计 §11.8）。

夹具是 10 天 × 600 批次。conftest 的 10 × 24 夹具装不了：E3 的真实半边要从
单个测试日期里抽出 500 个批次，而 24 个批次在 n=50 就停了。
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from experiment.data import spec
from experiment.data.assemble import manifest as MAN
from experiment.data.assemble import merge as MERGE
from experiment.data.generate.corpus import run_generation
from tests.conftest import write_lots_csv

GOLDEN = Path(__file__).parent / "golden" / "assemble_sections.json"

DAYS = 10
LOTS_PER_DAY = 600

#: 金样本覆盖的 manifest 段；其余三段由 :func:`test_only_the_clock_and_this_runs_paths_are_unpinned` 盯着。
PINNED = (
    "cells",
    "counts",
    "dedup",
    "files",
    "schema_version",
    "split_policy",
    "splits",
    "streams",
    "stress",
    "tail_thresholds",
)


def _rows() -> list[tuple]:
    """与 conftest 同构的批次表，但每天 600 条，够装 n=500。"""
    rows = []
    row_id = 0
    for day_index in range(DAYS):
        day = (date(2011, 1, 1) + timedelta(days=day_index)).isoformat()
        for slot in range(LOTS_PER_DAY):
            rows.append(
                (
                    row_id,
                    100000 + day_index * 1000 + slot,
                    day,
                    slot + 1,
                    "2.50",
                    (slot + 1) * 250,
                )
            )
            row_id += 1
    return rows


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    """跑一次**生产形态**的装配：全部 12 个 cell、含 E3/E4。"""
    root = tmp_path_factory.mktemp("frozen")
    lots = write_lots_csv(root / "collect" / "lots_clean.csv", _rows())
    # 装配会拿 collect_report.json 的 kept_rows 与批次表实际行数对账，
    # 所以夹具必须让它有得查（与 conftest.lots_csv 同一约定）。
    (root / "collect" / "collect_report.json").write_bytes(
        json.dumps(
            {
                "kept_rows": DAYS * LOTS_PER_DAY,
                "schema_version": 1,
                "source": {
                    "url": "https://example.invalid/fixture.zip",
                    "xlsx_bytes": 123,
                    "xlsx_sha256": "0" * 64,
                },
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    generation = root / "generation"
    run_generation(
        out_dir=generation, quota_override={"train": 1, "validation": 1, "test": 1}
    )
    manifest = MERGE.run_assembly(
        out_dir=root / "processed",
        lots_path=lots,
        generation_dir=generation,
        manifest_path=root / "data_manifest.json",
        config_path=root / "experiment_config.yaml",
        cells=None,
        synthetic_quota={split: 1 for split in spec.SPLITS},
        real_quota={split: len(spec.FAMILIES) for split in spec.SPLITS},
        include_stress=True,
    )
    return {"root": root, "manifest": manifest, "lots": lots}


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


# --- 派生内容 ---------------------------------------------------------------


@pytest.mark.parametrize("section", PINNED)
def test_derived_section_matches_the_golden(assembled, section):
    got = assembled["manifest"][section]
    if section == "cells":
        # 段内顺序不是协议的一部分；内容才是。
        got = sorted(got, key=lambda cell: (cell["split"], cell["n"], cell["alpha"]))
    assert got == _golden()[section]


def test_only_the_clock_and_this_runs_paths_are_unpinned(assembled):
    """金样本漏钉的段必须**只有**那三段，否则"钉住"会退化成"没钉"。

    把排除项写死在这里：将来若有人把 `generated_at` 挪走、或让 `output_dir`
    变成相对路径，这条会先失败，而不是让金样本悄悄少钉一段。
    """
    manifest = assembled["manifest"]
    golden = _golden()
    assert set(manifest) - set(golden) == {"generated_at", "output_dir", "provenance"}
    assert set(golden) <= set(manifest)


def test_the_two_stress_sets_match_the_frozen_shape(assembled):
    """§8.5：E3 两个来源各 300，E4 只有合成半边。行数由金样本的 files 段钉住，
    这里再钉住"为什么"——配额来自 spec，而不是从某个数字反推。"""
    stress = assembled["manifest"]["stress"]
    assert stress["e3_size"] == {
        "real": spec.E3_REAL_PER_ALPHA * len(spec.ALPHAS),
        "synthetic": spec.E3_SYNTHETIC_PER_FAMILY * len(spec.FAMILIES) * len(spec.ALPHAS),
    }
    assert stress["e4_tail"]["real"] == 0


# --- 入库凭据里的路径（修正 a）---------------------------------------------


def test_paths_are_posix_and_relative_inside_the_repo(assembled):
    """绝对路径会把本机目录树写进**入库**的凭据里。

    夹具在仓库外，所以端到端只能验证"仓库外保持绝对、且是 POSIX 形式"；
    仓库内那一支走不到，必须直接量 `_display_path`。
    """
    manifest = assembled["manifest"]
    assert "\\" not in manifest["output_dir"]
    assert Path(manifest["output_dir"]).is_absolute()
    for key in ("lots_clean", "collect_report", "generation_manifest"):
        path = manifest["provenance"][key]["path"]
        assert "\\" not in path, path
        assert Path(path).is_absolute(), path

    inside = spec.REPO_ROOT / "datasets" / "raw" / "collect" / "lots_clean.csv"
    assert MAN._display_path(inside) == "datasets/raw/collect/lots_clean.csv"
    # 绑定 `REPO_ROOT` 本身必须是绝对路径，否则相对化会静默失效。
    assert spec.REPO_ROOT.is_absolute()
    assert MAN._display_path(spec.REPO_ROOT) == "."


# --- 两个协议产物的字节 -----------------------------------------------------


def test_manifest_bytes_are_the_canonical_serialisation(assembled):
    """格式也是协议的一部分：缩进 2、键排序、不转义非 ASCII、LF、无 BOM、末尾一个换行。"""
    raw = (assembled["root"] / "data_manifest.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert raw.startswith(b'{\n  "cells"')  # 缩进 2，且首键是排序后的第一个
    expected = (
        json.dumps(
            assembled["manifest"], indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n"
    )
    assert raw == expected.encode("utf-8")


def test_config_bytes_are_the_canonical_block_yaml(assembled):
    raw = (assembled["root"] / "experiment_config.yaml").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")
    assert not text.lstrip().startswith(b"{".decode())  # 块状，不是 flow
    doc = yaml.safe_load(text)
    assert text.splitlines()[0].split(":")[0] == sorted(doc)[0]


# --- §6 的正文（修正 c）-----------------------------------------------------


def test_config_carries_section_6_prose_as_well_as_its_table(assembled):
    """§6 的表格进 `budgets`；表格之外的正文进 `rl_environment`。

    只抄表格会让 §10 要求的"RL 设置"名不副实——表格里既没有 `gamma`，
    也没有网络结构，而这两项恰恰决定了能不能复现训练。
    """
    doc = yaml.safe_load(
        (assembled["root"] / "experiment_config.yaml").read_text(encoding="utf-8")
    )
    assert "rl_environment" in doc
    rl = doc["rl_environment"]
    assert rl["gamma"] == 1
    assert "v_i / sum_j(v_j)" in rl["reward_shaping"]
    assert "two-layer" in rl["network"]["encoder"]
    assert "64-unit" in rl["network"]["encoder"]
    assert "128-unit" in rl["network"]["head"]
    assert "two Q-value outputs" in rl["network"]["head"]
    assert rl["network"]["excluded_from_inputs"] == (
        "IDs and source labels are not prediction features"
    )
    assert rl["network"]["terminal"] == "terminal states bypass next-state evaluation"
    assert rl["budget_escalation"]["to_transitions"] == 1000000
    assert rl["budget_escalation"]["applies_to"] == "all three seeds"
    assert rl["checkpoint_selection"]["never_select_on"] == "test performance"
    assert "masked in exploration, inference" in rl["action_masking"]
    assert "balanced block" in rl["sampling"]
    # `budgets` 仍是那张表：两边不是同一份内容，别把正文塞进 budgets。
    assert doc["budgets"]["pilot_transitions"] == 50000
    assert "gamma" not in doc["budgets"]
