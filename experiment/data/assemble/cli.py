"""冻结语料装配入口。

用法::

    python -m experiment.data.assemble.cli
    python -m experiment.data.assemble.cli --out datasets/processed
    python -m experiment.data.assemble.cli --cell 20x0.25 --synthetic-quota 1 --no-stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiment.data.assemble.merge import run_assembly
from experiment.data.spec import (
    DATA_MANIFEST_PATH,
    DATASETS_PROCESSED,
    EXPERIMENT_CONFIG_PATH,
    GENERATION_DIR,
    REAL_DIR,
    SPLITS,
)


def parse_cell(text: str) -> tuple[int, float]:
    """解析 ``20x0.25`` 形式的 cell 说明。

    用 ``x`` 而不是逗号：Windows 的 shell 与 argparse 对逗号分隔的多值参数
    处理各不相同，而 ``x`` 是一个不需要转义的整体记号。
    """
    size, sep, alpha = text.partition("x")
    if not sep:
        raise argparse.ArgumentTypeError(f"cell 需形如 20x0.25，收到 {text!r}")
    try:
        return int(size), float(alpha)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cell 需形如 20x0.25，收到 {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="装配冻结的混合背包语料")
    parser.add_argument(
        "--out", type=Path, default=DATASETS_PROCESSED, help="语料输出根目录"
    )
    parser.add_argument(
        "--lots", type=Path, default=REAL_DIR / "lots_clean.csv", help="清洗后的批次表"
    )
    parser.add_argument(
        "--generation", type=Path, default=GENERATION_DIR, help="合成语料根目录"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATA_MANIFEST_PATH, help="data_manifest.json 的位置"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_CONFIG_PATH,
        help="experiment_config.yaml 的位置",
    )
    parser.add_argument(
        "--cell",
        type=parse_cell,
        action="append",
        default=None,
        help="只装配指定的 n x alpha，可重复；默认全部 12 个",
    )
    parser.add_argument(
        "--synthetic-quota",
        type=int,
        default=None,
        help="覆盖每族每 cell 的合成条数（仅供小样本试跑）",
    )
    parser.add_argument(
        "--real-quota",
        type=int,
        default=None,
        help="覆盖每 cell 的真实条数（仅供小样本试跑）",
    )
    parser.add_argument("--no-stress", action="store_true", help="跳过 E3/E4")
    return parser


def main() -> int:
    # 与 generate/collect 的入口一致：重定向到文件时 Windows 用 ANSI 代码页
    # 编码 stdout，中文标签会抛 UnicodeEncodeError；此时语料已经写完，只会
    # 留下一个失败的退出码和空日志。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    synthetic_quota = (
        None if args.synthetic_quota is None else {s: args.synthetic_quota for s in SPLITS}
    )
    real_quota = None if args.real_quota is None else {s: args.real_quota for s in SPLITS}

    manifest = run_assembly(
        out_dir=args.out,
        lots_path=args.lots,
        generation_dir=args.generation,
        manifest_path=args.manifest,
        config_path=args.config,
        cells=args.cell,
        synthetic_quota=synthetic_quota,
        real_quota=real_quota,
        include_stress=not args.no_stress,
    )

    print(f"输出目录: {args.out}")
    print(f"manifest: {args.manifest}")
    print(f"config:   {args.config}")
    for split, count in manifest["counts"].items():
        print(
            f"  {split}: 合成 {count['synthetic']} + 真实 {count['real']} "
            f"= {count['total']}（逐 cell 50:50）"
        )
    for name, count in manifest["stress"].items():
        print(f"  {name}: 合成 {count['synthetic']} + 真实 {count['real']}")
    print(f"实例总数: {manifest['dedup']['total_instances']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
