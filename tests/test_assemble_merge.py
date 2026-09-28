"""合并顺序、逐 cell 50:50、全局去重、字节稳定与端到端。

所有测试走密闭夹具：夹具语料的 cell 集合与真实语料一致（n ∈ SIZES），
因此这里只出现 n=20 这类真实存在的 cell。
"""

import hashlib
import json
from pathlib import Path

import pytest

from experiment.data import spec
from experiment.data.assemble import lots as L
from experiment.data.assemble import merge as M
from experiment.data.spec import FAMILIES

CELL = (20, 0.25)

#: cell 级的配额是**标量**。``assemble_split`` 的 ``synthetic_quota`` 是
#: 「每个合成族每 cell 的条数」；夹具每个 cell 只提供 1 条，故取 1。
#: ``real_quota`` 必须等于族数 × 每族条数，否则 50:50 闸门会报错。
#:
#: 前缀 ``CELL_`` 是刻意加的：``spec.REAL_QUOTA``（= ``{"train": 500, …}``）说的是
#: 「每个 split 的真实条数」，与本处的「每个 cell 4 条」同名而不同物，量级还差两个
#: 数量级。不带前缀，读的人很容易把 4 当成 500。
CELL_SYN_QUOTA = 1
CELL_REAL_QUOTA = len(FAMILIES) * CELL_SYN_QUOTA

#: run 级的配额是**按 split 索引的映射**，只给 ``run_assembly`` 用。
#: 两个层级同名不同形，混用会让 ``read_cell(path)[:quota]`` 拿字典去切片。
QUOTAS = {
    "synthetic_quota": {split: CELL_SYN_QUOTA for split in spec.SPLITS},
    "real_quota": {split: CELL_REAL_QUOTA for split in spec.SPLITS},
}


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


def _read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


# --- 合成半边是「读回来」而不是「重新生成」----------------------------------


def test_read_cell_preserves_file_order(fixture_corpus):
    path = fixture_corpus / "train" / "uniform_n20_a25.jsonl"
    instances = M.read_cell(path)
    assert [i["instance_id"] for i in instances] == ["syn-train-uniform-n20-a25-i0000"]


def test_synthetic_cell_path_points_into_the_generation_tree(fixture_corpus):
    path = M.synthetic_cell_path(fixture_corpus, "train", "uniform", 20, 0.25)
    assert path.is_file()
    assert path.name == "uniform_n20_a25.jsonl"


def test_synthetic_half_is_identical_to_the_generation_corpus(fixture_corpus, index):
    """合并文件里的合成记录必须与生成阶段的语料逐字节同源。

    「重新生成一遍」与「读回来」在数学上等价，但前者会让合并产物依赖当前
    代码而非已审计的语料。这条断言钉住后者。
    """
    source = M.read_cell(M.synthetic_cell_path(fixture_corpus, "train", "uniform", 20, 0.25))
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    picked = [i for i in merged if i["instance_id"] == source[0]["instance_id"]]
    assert len(picked) == 1
    assert picked[0] == source[0]


# --- 顺序与 50:50 -----------------------------------------------------------


def test_cell_order_is_families_then_real(index, fixture_corpus):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    assert [i["source"] for i in merged] == ["synthetic"] * 4 + ["real"] * 4
    assert [i["family"] for i in merged[:4]] == list(FAMILIES)
    assert all(i["family"] is None for i in merged[4:])


def test_every_cell_is_fifty_fifty(index, fixture_corpus):
    _, cells = M.assemble_split(
        "train", index, fixture_corpus, cells=[(20, 0.25), (20, 0.50)], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA
    )
    assert len(cells) == 2
    for cell in cells:
        assert cell["synthetic"] == cell["real"] == 4


def test_raises_when_the_two_sides_do_not_match(index, fixture_corpus):
    """50:50 检查必须是活的：配额不匹配时装配必须报错，而不是静默产出。"""
    with pytest.raises(ValueError, match="50:50"):
        M.assemble_split(
            "train", index, fixture_corpus, cells=[CELL],
            synthetic_quota=1, real_quota=5,
        )


def test_real_instances_use_the_family_count_as_the_quota(index, fixture_corpus):
    """真实半边没有族维度，其条数必须等于合成半边（族数 × 每族条数）。

    每族条数只能取 ``CELL_SYN_QUOTA``：夹具每个 cell 只有 1 条合成实例
    （``run_generation(quota_override=1)``），要求 3 条就会先被 50:50 闸门拦下。
    两边都断言，才能同时钉住「合成半边真的按族数产出」与「真实半边对齐它」。
    """
    merged, _ = M.assemble_split(
        "train", index, fixture_corpus, cells=[CELL],
        synthetic_quota=CELL_SYN_QUOTA, real_quota=len(FAMILIES) * CELL_SYN_QUOTA,
    )
    assert sum(1 for i in merged if i["source"] == "synthetic") == len(FAMILIES) * CELL_SYN_QUOTA
    assert sum(1 for i in merged if i["source"] == "real") == len(FAMILIES) * CELL_SYN_QUOTA


# --- 全局去重 ---------------------------------------------------------------


def test_all_hashes_in_the_merged_output_are_unique(index, fixture_corpus):
    merged, _ = M.assemble_split(
        "train", index, fixture_corpus, cells=[(20, 0.25), (20, 0.50)], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA
    )
    hashes = [i["canonical_hash"] for i in merged]
    assert len(set(hashes)) == len(hashes)


def test_seen_accumulates_across_cells_and_splits(index, fixture_corpus):
    seen: set[str] = set()
    M.assemble_split("train", index, fixture_corpus, cells=[CELL], seen=seen, synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    M.assemble_split("validation", index, fixture_corpus, cells=[CELL], seen=seen, synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    assert len(seen) == 16


def test_synthetic_hash_colliding_with_seen_raises(index, fixture_corpus):
    """合成半边一旦与已装配的实例重复，必须报错而不是被静默吞掉。

    传入上一次装配的全部哈希即可制造这种冲突——这条断言证明那道检查是活的。
    """
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    with pytest.raises(ValueError, match="合成"):
        M.assemble_split(
            "train",
            index,
            fixture_corpus,
            cells=[CELL],
            seen={i["canonical_hash"] for i in merged},
            synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA,
        )


# --- 上游 provenance --------------------------------------------------------


def _upstream_tree(tmp_path: Path) -> tuple[Path, Path]:
    """一份带 collect_report.json 的批次表与一个带 manifest.json 的生成目录。"""
    real = tmp_path / "real"
    real.mkdir(parents=True, exist_ok=True)
    lots = real / "lots_clean.csv"
    lots.write_bytes(b"source_row_id,invoice_no\n0,1\n")
    (real / "collect_report.json").write_bytes(
        json.dumps(
            {
                "kept_rows": 1,
                "source": {
                    "url": "https://example.invalid/x.zip",
                    "xlsx_bytes": 42,
                    "xlsx_sha256": "a" * 64,
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    gen = tmp_path / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "manifest.json").write_bytes(
        json.dumps({"generator_fingerprint": "b" * 64, "schema_version": 1}).encode("utf-8")
    )
    return lots, gen


def test_upstream_provenance_hashes_every_file_it_cites(tmp_path):
    lots, gen = _upstream_tree(tmp_path)
    prov = M.upstream_provenance(lots, gen)
    assert prov["lots_clean"]["sha256"] == hashlib.sha256(lots.read_bytes()).hexdigest()
    assert prov["lots_clean"]["bytes"] == len(lots.read_bytes())
    assert prov["lots_clean"]["columns"] == list(spec.LOTS_COLUMNS)
    assert prov["collect_report"]["sha256"] == hashlib.sha256(
        (lots.parent / "collect_report.json").read_bytes()
    ).hexdigest()
    assert prov["generation_manifest"]["sha256"] == hashlib.sha256(
        (gen / "manifest.json").read_bytes()
    ).hexdigest()


def test_upstream_provenance_reads_the_values_instead_of_copying_them(tmp_path):
    """`url`/`xlsx_sha256` 等必须来自 collect_report.json 的**内容**。

    写死一份副本就会在上游换文件后继续说谎——这正是本阶段的病灶。
    """
    lots, gen = _upstream_tree(tmp_path)
    prov = M.upstream_provenance(lots, gen)
    assert prov["workbook"]["url"] == "https://example.invalid/x.zip"
    assert prov["workbook"]["bytes"] == 42
    assert prov["workbook"]["sha256"] == "a" * 64
    assert prov["generation_manifest"]["generator_fingerprint"] == "b" * 64
    assert prov["lots_clean"]["rows"] == 1
    assert prov["units"] == {"value": "value_pence", "weight": "quantity_w"}
    assert prov["cleaning_rules"] == list(spec.CLEANING_RULES)


def test_upstream_provenance_follows_an_edited_report(tmp_path):
    """改掉报告里的 url，provenance 必须跟着变。"""
    lots, gen = _upstream_tree(tmp_path)
    report = lots.parent / "collect_report.json"
    doc = json.loads(report.read_text(encoding="utf-8"))
    doc["source"]["url"] = "https://example.invalid/CHANGED.zip"
    report.write_bytes(json.dumps(doc, sort_keys=True).encode("utf-8"))
    assert M.upstream_provenance(lots, gen)["workbook"]["url"].endswith("CHANGED.zip")


def test_upstream_provenance_reports_missing_files_as_none(tmp_path):
    """上游文件缺失时摘要为 None，不假装有。"""
    prov = M.upstream_provenance(tmp_path / "nothing.csv", tmp_path / "nowhere")
    assert prov["lots_clean"]["sha256"] is None
    assert prov["collect_report"]["sha256"] is None
    assert prov["generation_manifest"]["generator_fingerprint"] is None


# --- 落盘 -------------------------------------------------------------------


def test_write_split_writes_lf_jsonl_and_returns_the_digest(index, fixture_corpus, tmp_path):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    path = tmp_path / "train" / "instances.jsonl"
    digest = M.write_split(path, merged)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert digest == hashlib.sha256(raw).hexdigest()
    assert len(_read(path)) == 8


def test_write_split_is_byte_identical_on_rerun(index, fixture_corpus, tmp_path):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    a, b = tmp_path / "a" / "instances.jsonl", tmp_path / "b" / "instances.jsonl"
    M.write_split(a, merged)
    M.write_split(b, merged)
    assert a.read_bytes() == b.read_bytes()


# --- 端到端 -----------------------------------------------------------------


def _run(tmp_path, lots_csv, fixture_corpus, **overrides):
    kwargs = dict(
        out_dir=tmp_path / "processed",
        lots_path=lots_csv,
        generation_dir=fixture_corpus,
        manifest_path=tmp_path / "data_manifest.json",
        config_path=tmp_path / "experiment_config.yaml",
        cells=[CELL],
        include_stress=False,
        **QUOTAS,
    )
    kwargs.update(overrides)
    return M.run_assembly(**kwargs)


def test_run_assembly_writes_every_artifact(lots_csv, fixture_corpus, tmp_path):
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    for split in ("train", "validation", "test"):
        assert (tmp_path / "processed" / split / "instances.jsonl").is_file()
    assert (tmp_path / "data_manifest.json").is_file()
    assert (tmp_path / "experiment_config.yaml").is_file()
    assert set(manifest["counts"]) == {"train", "validation", "test"}
    assert manifest["counts"]["train"] == {"synthetic": 4, "real": 4, "total": 8}


def test_run_assembly_is_deterministic_across_runs(lots_csv, fixture_corpus, tmp_path):
    _run(tmp_path, lots_csv, fixture_corpus, out_dir=tmp_path / "one")
    _run(tmp_path, lots_csv, fixture_corpus, out_dir=tmp_path / "two")
    for split in ("train", "validation", "test"):
        assert (tmp_path / "one" / split / "instances.jsonl").read_bytes() == (
            tmp_path / "two" / split / "instances.jsonl"
        ).read_bytes()


def test_run_assembly_records_real_file_hashes(lots_csv, fixture_corpus, tmp_path):
    """三个文件记录都必须能从磁盘上的真文件独立复算。

    摘要与字节数来自文件本身；行数来自**解析**出来的记录条数，而不是再数一遍
    换行符——后者与 ``_file_record`` 的实现是同一个表达式，永远相等，什么也
    证明不了。解析计数会抓到多行 JSON、多余空行这类真实的写入事故。
    """
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    assert len(manifest["files"]) == 3
    for record in manifest["files"]:
        path = tmp_path / "processed" / record["path"]
        raw = path.read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["bytes"] == len(raw)
        assert record["lines"] == len(_read(path))


def test_dedup_block_is_measured_from_the_written_corpus(lots_csv, fixture_corpus, tmp_path):
    """去重统计必须能从落盘内容独立复算出来。

    四个数各有独立来源：总数来自写出的行数、唯一数来自哈希集合、碰撞数与
    最大重抽次数来自每条记录的 resample_attempt。**任何一个都不许是与另一个
    同源的表达式**——那正是本项目反复出现的「常量对常量」病灶。
    """
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    written = [
        inst
        for record in manifest["files"]
        for inst in _read(tmp_path / "processed" / record["path"])
    ]
    dedup = manifest["dedup"]
    assert dedup["total_instances"] == len(written)
    assert dedup["distinct_canonical_hashes"] == len({i["canonical_hash"] for i in written})
    assert dedup["collisions_resampled"] == sum(i["resample_attempt"] for i in written)
    assert dedup["max_attempt_used"] == max(i["resample_attempt"] for i in written)


def test_run_assembly_enforces_the_family_source_invariant(lots_csv, fixture_corpus, tmp_path):
    """设计文档 §10 的不变量：family is None <=> source == "real"。"""
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    written = [
        inst
        for record in manifest["files"]
        for inst in _read(tmp_path / "processed" / record["path"])
    ]
    assert written
    for inst in written:
        assert (inst["family"] is None) == (inst["source"] == "real")


def test_run_assembly_reports_training_only_tail_thresholds(lots_csv, fixture_corpus, tmp_path):
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    threshold = manifest["tail_thresholds"]
    assert threshold["basis"] == "train-split lots only"
    assert threshold["quantile"] == 0.99
    # 夹具 train 段 7 天的 quantity 为 1..24，阈值必然落在这个区间内
    assert 1 <= int(threshold["quantity_w"]) <= 24


def test_run_assembly_rejects_a_lots_table_that_contradicts_the_collect_report(
    lots_csv, fixture_corpus, tmp_path
):
    """上游报告与实际批次表对不上时必须报错。

    `kept_rows` 与 CSV 的实际行数是两个独立来源；不一致意味着批次表被替换过
    而报告没跟着更新，此时装配出的语料无法溯源。
    """
    report = lots_csv.parent / "collect_report.json"
    doc = json.loads(report.read_text(encoding="utf-8"))
    doc["kept_rows"] = 999999
    report.write_bytes(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    with pytest.raises(ValueError, match="999999"):
        _run(tmp_path, lots_csv, fixture_corpus)
