"""UCI Online Retail 采集入口。

用法::

    python -m experiment.data.collect.cli
    python -m experiment.data.collect.cli --force-download
    python -m experiment.data.collect.cli --out datasets/raw/real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiment.data.collect.run import run_collection
from experiment.data.spec import REAL_DIR


def main() -> int:
    # 重定向到文件时 Windows 用 ANSI 代码页（如 cp1252/cp437）编码 stdout，
    # 中文标签会抛 UnicodeEncodeError；此时产物已经写完，只会留下一个失败
    # 的退出码和空日志。统一切到 UTF-8，与落盘约定一致。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="采集并清洗 UCI Online Retail")
    parser.add_argument(
        "--out",
        type=Path,
        default=REAL_DIR,
        help="输出根目录（默认 datasets/raw/real）",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="已有的工作簿路径；给定则不下载",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="忽略已存在的工作簿，强制重新下载",
    )
    args = parser.parse_args()

    report = run_collection(
        out_dir=args.out,
        workbook=args.workbook,
        force_download=args.force_download,
    )

    print(f"输出目录: {args.out}")
    print(f"原始行数: {report['raw_rows']}")
    print(f"幸存行数: {report['kept_rows']}")
    print("逐规则丢弃:")
    for rule, count in report["cleaning"]["dropped_by_reason"].items():
        print(f"  {rule}: {count}")
    print(f"恒等式成立: {report['cleaning']['identity_holds']}")
    print(f"工作簿 sha256: {report['source']['xlsx_sha256']}")
    print(f"批次表 sha256: {report['source']['lots_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
