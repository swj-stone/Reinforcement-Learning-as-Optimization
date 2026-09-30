"""``reference_results.csv`` 的生成入口。

用法::

    python -m experiment.eval.cli
    python -m experiment.eval.cli --corpus datasets/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from experiment.data.spec import (
    DATASETS_PROCESSED,
    E3_NAME,
    E4_NAME,
    REPO_ROOT,
    SPLITS,
)
from experiment.eval.reference import format_upper_bound, solve

#: 产物放在**仓库根目录并入库**，与 data_manifest.json / experiment_config.yaml
#: 一致：datasets/ 整体被 gitignore，把审计凭据放在那里等于换台机器就无从证明
#: 数据未被调换。
#:
#: 刻意**不**写进 spec.py：generate/corpus.py 的 generator_fingerprint() 会把
#: spec.py 的字节纳入哈希，而该指纹已写进入库的 data_manifest.json；往里加一个
#: 常量会让「重新生成是否一致」这个复核动作失真。
DEFAULT_OUT: Path = REPO_ROOT / "reference_results.csv"

#: 列序即产物的一部分。
COLUMNS: tuple[str, ...] = (
    "instance_id",
    "certified",
    "value",
    "upper_bound",
    "item_set_hash",
    "solver_time_s",
)


def corpus_files(root: Path) -> list[Path]:
    """语料的五个文件，顺序固定：三个 split 之后是 E3、E4。"""
    root = Path(root)
    files = [root / split / "instances.jsonl" for split in SPLITS]
    files.append(root / "stress" / E3_NAME / "instances.jsonl")
    files.append(root / "stress" / E4_NAME / "instances.jsonl")
    return files


def read_instances(files: Sequence[Path]) -> list[dict]:
    instances: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    instances.append(json.loads(line))
    return instances


def run(corpus_root: Path, out_path: Path) -> dict:
    """求解全部实例并落盘，返回一份汇总。

    途中会校验 ``U >= V*`` 并对违例直接报错——这是设计 §12 的「上界性质」
    在**真实语料**上的执行点；CI 里跑的是同阶形状的合成语料（测试不读被
    gitignore 的 datasets/）。
    """
    rows = []
    for instance in read_instances(corpus_files(corpus_root)):
        result = solve(instance)
        if result.upper_bound < result.value:
            raise AssertionError(
                f"{instance['instance_id']}: 上界 {result.upper_bound} "
                f"< 最优值 {result.value}"
            )
        rows.append(
            {
                "certified": "true" if result.certified else "false",
                "instance_id": instance["instance_id"],
                "item_set_hash": result.item_set_hash,
                "solver_time_s": f"{result.runtime_s:.6f}",
                "upper_bound": format_upper_bound(result.upper_bound),
                "value": str(result.value),
            }
        )

    # 行序是产物的一部分：按 instance_id 升序，确定且与语料的文件顺序无关。
    rows.sort(key=lambda row: row["instance_id"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 关掉 Python 的换行翻译，lineterminator="\n" 固定为 LF。
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "total_solver_time_s": sum(float(row["solver_time_s"]) for row in rows),
        "trivial_zero": sum(1 for row in rows if row["value"] == "0"),
        "out": out_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为冻结语料生成精确参考解与上界")
    parser.add_argument(
        "--corpus", type=Path, default=DATASETS_PROCESSED, help="冻结语料根目录"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="reference_results.csv 的位置"
    )
    return parser


def main() -> int:
    # 与 generate/collect/assemble 的入口一致：重定向到文件时 Windows 用 ANSI
    # 代码页编码 stdout，中文标签会抛 UnicodeEncodeError。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    start = time.perf_counter()
    summary = run(args.corpus, args.out)

    print(f"语料根目录: {args.corpus}")
    print(f"产物:       {summary['out']}")
    print(f"实例数:     {summary['rows']}")
    print(f"平凡解:     {summary['trivial_zero']}（已从百分比分母剔除）")
    print(f"求解总耗时: {summary['total_solver_time_s']:.1f} s")
    print(f"墙钟总耗时: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
