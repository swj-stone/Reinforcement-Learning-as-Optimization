"""合成语料生成入口。

用法::

    python -m experiment.data.generate.cli
    python -m experiment.data.generate.cli --out datasets/raw/generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.data.generate.corpus import run_generation
from experiment.data.spec import GENERATION_DIR


def main() -> int:
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
