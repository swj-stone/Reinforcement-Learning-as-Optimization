"""把合成语料与真实实例合并成冻结的混合语料，并编排整条装配流程。

合并文件的排列顺序是冻结语义，且要让 50:50 肉眼可核：
``for n: for alpha: 四个族各 quota 条 -> 真实 quota 条``。

依据 EXPERIMENT_PLAN.md 第 4 节与设计文档 §8、§9。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from experiment.data.assemble.lots import LotIndex, load_index, tail_thresholds
from experiment.data.assemble.manifest import (
    build_config,
    build_manifest,
    upstream_provenance,
    write_config,
    write_manifest,
)
from experiment.data.assemble.real import build_real_cell
from experiment.data.assemble.stress import build_e3, build_e4
from experiment.data.generate.corpus import write_cell
from experiment.data.spec import (
    DATA_MANIFEST_PATH,
    DATASETS_PROCESSED,
    EXPERIMENT_CONFIG_PATH,
    E3_NAME,
    E4_NAME,
    FAMILIES,
    GENERATION_DIR,
    REAL_DIR,
    REAL_QUOTA,
    SCHEMA_VERSION,
    SPLITS,
    SYNTHETIC_QUOTA,
    TAIL_QUANTILE,
    capacity_cells,
    cell_filename,
)


def synthetic_cell_path(
    generation_dir: Path, split: str, family: str, n: int, alpha: float
) -> Path:
    """合成语料里一个 cell 的文件路径。"""
    return Path(generation_dir) / split / cell_filename(family, n, alpha)


def read_cell(path: Path) -> list[dict]:
    """读回一个合成 cell 的实例，保持文件行序。

    刻意是「读回来」而不是「重新生成」：前者保证合并文件里的合成半边与已
    审计的原始语料逐字节同源，后者会让它依赖当前代码。
    """
    instances: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(json.loads(line))
    return instances


def assemble_split(
    split: str,
    lot_index: LotIndex,
    generation_dir: Path,
    cells: Sequence[tuple[int, float]] | None = None,
    synthetic_quota: int | None = None,
    real_quota: int | None = None,
    seen: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """拼出一个 split 的全部实例，并返回逐 cell 的 50:50 审计。

    Args:
        cells: 要装配的 ``(n, alpha)`` 序列；``None`` 表示全部 12 个。测试可
            只传一个 cell 以便在密闭夹具上跑。
        synthetic_quota: 每个合成族每 cell 的条数；``None`` 取冻结配额
        real_quota: 每 cell 真实条数；``None`` 取冻结配额
        seen: 跨 split 的全局哈希集合，就地更新

    Returns:
        ``(instances, cells_audit)``

    Raises:
        ValueError: 某 cell 两侧计数不等（第 4 节的 50:50），或合成半边与
            已装配的实例哈希重复。
    """
    book = set() if seen is None else seen
    syn_quota = SYNTHETIC_QUOTA[split] if synthetic_quota is None else synthetic_quota
    real_q = REAL_QUOTA[split] if real_quota is None else real_quota
    plan = list(capacity_cells() if cells is None else cells)

    instances: list[dict] = []
    audit: list[dict] = []

    for n, alpha in plan:
        synthetic: list[dict] = []
        for family in FAMILIES:
            path = synthetic_cell_path(generation_dir, split, family, n, alpha)
            synthetic.extend(read_cell(path)[:syn_quota])

        hashes = {inst["canonical_hash"] for inst in synthetic}
        if len(hashes) != len(synthetic):
            raise ValueError(f"{split}/n={n}/alpha={alpha} 的合成 cell 内部有重复实例")
        if hashes & book:
            raise ValueError(
                f"{split}/n={n}/alpha={alpha} 的合成实例与已装配的实例重复"
            )
        book |= hashes

        real = build_real_cell(lot_index, split, split, n, alpha, real_q, book)

        if len(synthetic) != len(real):
            raise ValueError(
                f"{split}/n={n}/alpha={alpha} 不满足 50:50："
                f"合成 {len(synthetic)} 条，真实 {len(real)} 条"
            )

        instances.extend(synthetic)
        instances.extend(real)
        audit.append(
            {
                "alpha": alpha,
                "n": n,
                "real": len(real),
                "split": split,
                "synthetic": len(synthetic),
            }
        )

    return instances, audit


def write_split(path: Path, instances: list[dict]) -> str:
    """落盘一个 split 的实例，返回内容的 SHA-256。

    直接复用生成阶段的 ``write_cell``：序列化约定必须与合成语料完全一致，
    否则同一个实例在两个文件里的字节会不同。
    """
    return write_cell(path, instances)


def _file_record(path: Path, root: Path) -> dict:
    """一个语料文件的审计记录：相对路径、摘要、字节数、行数。"""
    raw = Path(path).read_bytes()
    return {
        "bytes": len(raw),
        "lines": raw.count(b"\n"),
        "path": Path(path).relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _resample_stats(instances: Sequence[dict]) -> tuple[int, int]:
    """``(碰撞次数, 最大重抽次数)``。

    每条记录的 ``resample_attempt`` 就是它经历过的碰撞数，因此求和即总碰撞数。
    这两个数是从产物本身量出来的，不是另算一份可能与之脱节的常量。
    """
    return (
        sum(inst["resample_attempt"] for inst in instances),
        max((inst["resample_attempt"] for inst in instances), default=0),
    )


def run_assembly(
    out_dir: Path = DATASETS_PROCESSED,
    lots_path: Path = REAL_DIR / "lots_clean.csv",
    generation_dir: Path = GENERATION_DIR,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    cells: Sequence[tuple[int, float]] | None = None,
    synthetic_quota: Mapping[str, int] | None = None,
    real_quota: Mapping[str, int] | None = None,
    include_stress: bool = True,
) -> dict:
    """装配全部语料并写出两个协议产物，返回 manifest。

    遍历顺序是冻结语义：先三个核心 split（train -> validation -> test），
    再 E3、最后 E4。``seen`` 跨全部五个数据集共享，因此「全局唯一」是一条
    真断言，而不是三次各自为政的检查。

    Args:
        include_stress: 为 ``False`` 时跳过 E3/E4，供快速端到端测试使用
            （E3 的 n=500 在密闭夹具上无法满足）
    """
    manifest_path = DATA_MANIFEST_PATH if manifest_path is None else Path(manifest_path)
    config_path = EXPERIMENT_CONFIG_PATH if config_path is None else Path(config_path)
    out_dir = Path(out_dir)

    lot_index = load_index(lots_path)
    provenance = upstream_provenance(lots_path, generation_dir)

    declared = provenance["lots_clean"]["rows"]
    actual = len(lot_index.lots)
    if declared is not None and declared != actual:
        raise ValueError(
            f"collect_report.json 记录 {declared} 行，但批次表实际读出 {actual} 行；"
            "上游产物与其报告不一致，装配结果无法溯源"
        )

    book: set[str] = set()
    counts: dict[str, dict] = {}
    cells_audit: list[dict] = []
    files: list[dict] = []
    total_instances = 0
    collisions = 0
    max_attempt = 0

    for split in SPLITS:
        instances, audit = assemble_split(
            split,
            lot_index,
            generation_dir,
            cells,
            None if synthetic_quota is None else synthetic_quota[split],
            None if real_quota is None else real_quota[split],
            book,
        )
        path = out_dir / split / "instances.jsonl"
        write_split(path, instances)
        files.append(_file_record(path, out_dir))

        synthetic = sum(cell["synthetic"] for cell in audit)
        real = sum(cell["real"] for cell in audit)
        counts[split] = {"real": real, "synthetic": synthetic, "total": synthetic + real}
        cells_audit.extend(audit)

        total_instances += len(instances)
        hits, worst = _resample_stats(instances)
        collisions += hits
        max_attempt = max(max_attempt, worst)

    stress_counts: dict[str, dict] = {}
    if include_stress:
        for name, instances in (
            (E3_NAME, build_e3(lot_index, book)),
            (E4_NAME, build_e4(book)),
        ):
            path = out_dir / "stress" / name / "instances.jsonl"
            write_split(path, instances)
            files.append(_file_record(path, out_dir))
            stress_counts[name] = {
                "real": sum(1 for i in instances if i["source"] == "real"),
                "synthetic": sum(1 for i in instances if i["source"] == "synthetic"),
            }
            total_instances += len(instances)
            hits, worst = _resample_stats(instances)
            collisions += hits
            max_attempt = max(max_attempt, worst)

    train_lots = [lot for pool in lot_index.pools["train"].values() for lot in pool]

    manifest = build_manifest(
        cells=cells_audit,
        counts=counts,
        dedup={
            "collisions_resampled": collisions,
            "distinct_canonical_hashes": len(book),
            "max_attempt_used": max_attempt,
            "total_instances": total_instances,
        },
        files=files,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        lot_index=lot_index,
        out_dir=out_dir,
        provenance=provenance,
        schema_version=SCHEMA_VERSION,
        stress=stress_counts,
        tail=tail_thresholds(train_lots, TAIL_QUANTILE),
    )
    write_manifest(manifest_path, manifest)
    write_config(
        config_path,
        build_config(out_dir=out_dir, generated_at=manifest["generated_at"]),
    )
    return manifest
