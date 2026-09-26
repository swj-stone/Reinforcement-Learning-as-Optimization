"""合成语料生成入口。

用法::

    python -m experiment.data.generate.cli
    python -m experiment.data.generate.cli --out datasets/raw/generation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiment.data.generate.corpus import run_generation
from experiment.data.spec import GENERATION_DIR


def main() -> int:
    # 重定向到文件时 Windows 用 ANSI 代码页（如 cp1252/cp437）编码 stdout，
    # 中文标签会抛 UnicodeEncodeError；此时语料已经写完，只会留下一个失败
    # 的退出码和空日志。统一切到 UTF-8，与落盘约定一致。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="生成合成背包语料")
    parser.add_argument(
        "--out",
        type=Path,
        default=GENERATION_DIR,
        help="输出根目录（默认 datasets/raw/generation）",
    )
    args = parser.parse_args()

    manifest = run_generation(out_dir=args.out)

    print(f"输出目录: {args.out}")
    print(f"cell 数: {len(manifest['cells'])}")
    print(f"实例总数: {manifest['dedup']['total_instances']}")
    print(f"重抽次数: {manifest['dedup']['collisions_resampled']}")
    print(f"最大 attempt: {manifest['dedup']['max_resample_attempt']}")
    print(f"generator fingerprint: {manifest['generator_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
