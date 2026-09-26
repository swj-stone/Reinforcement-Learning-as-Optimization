"""实验规格的唯一真源。

本模块的常量一经确定即视为冻结：改动其中任何一项都会改变已生成语料的
内容与哈希，等同于使既有产物失效。

依据 EXPERIMENT_PLAN.md 第 2 节。
"""

from __future__ import annotations

import math
from pathlib import Path

# --- 冻结的生成维度 ---------------------------------------------------------

FAMILIES: tuple[str, ...] = ("uniform", "normal", "correlated", "skewed")
SIZES: tuple[int, ...] = (20, 50, 100, 200)
ALPHAS: tuple[float, ...] = (0.25, 0.50, 0.75)
SPLITS: tuple[str, ...] = ("train", "validation", "test")

MASTER_SEEDS: dict[str, int] = {"train": 1001, "validation": 2001, "test": 3001}

#: 每个 (family, n, alpha) cell 的实例数
SYNTHETIC_QUOTA: dict[str, int] = {"train": 125, "validation": 25, "test": 25}

# --- 生成语义常量 -----------------------------------------------------------

#: 偏斜族中稀有索引的比例（EXPERIMENT_PLAN.md 第 2 节）
SKEW_FRACTION: float = 0.05
#: 偏斜族稀有项的权重范围，含端点
SKEW_HEAVY_WEIGHT_RANGE: tuple[int, int] = (200, 500)
#: 偏斜族稀有项的密度范围，含端点
SKEW_HEAVY_DENSITY_RANGE: tuple[int, int] = (30, 60)
#: 常规项的权重与密度范围，含端点
SKEW_BASE_WEIGHT_RANGE: tuple[int, int] = (1, 20)
SKEW_BASE_DENSITY_RANGE: tuple[int, int] = (1, 5)

#: 跨 split 哈希冲突时的重抽上限，超过即报错而非死循环
MAX_RESAMPLE_ATTEMPT: int = 1000

#: manifest 的模式版本
SCHEMA_VERSION: int = 1

# --- 输出路径 ---------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS_RAW: Path = REPO_ROOT / "datasets" / "raw"
GENERATION_DIR: Path = DATASETS_RAW / "generation"
REAL_DIR: Path = DATASETS_RAW / "real"


# --- 派生辅助 ---------------------------------------------------------------


def alpha_key(alpha: float) -> str:
    """容量比的稳定字符串键，用于种子派生。

    显式格式化为两位小数，使「本应相同、却因浮点运算而近似」的两个值落到同一个
    键上：``0.1 + 0.2`` 的 ``repr`` 是 ``0.30000000000000004``，直接 ``str()``
    会与 ``0.3`` 派生出两个不同的种子，``f"{...:.2f}"`` 则都得到 ``0.30``。
    """
    return f"{alpha:.2f}"


def alpha_token(alpha: float) -> str:
    """容量比的文件名记号：0.25 -> ``a25``。

    用整数百分比而非 ``a0.25``，避免文件名中的点被工具按扩展名切分。
    """
    return f"a{round(alpha * 100):02d}"


def skew_k(n: int) -> int:
    """偏斜族的稀有索引个数：``max(1, floor(0.05n + 0.5))``。"""
    return max(1, math.floor(SKEW_FRACTION * n + 0.5))


def cell_filename(family: str, n: int, alpha: float) -> str:
    """cell 文件名，如 ``uniform_n20_a25.jsonl``。"""
    return f"{family}_n{n}_{alpha_token(alpha)}.jsonl"


def all_cells() -> list[tuple[str, int, float]]:
    """按冻结顺序枚举全部 48 个 cell。

    顺序即去重时的判定顺序，是冻结语义的一部分：family 在外层、n 居中、alpha
    在内层（即下方列表推导的嵌套序）。改动这个嵌套序会改变哪些实例被重抽。

    注意本函数不含 split 维度：先 train、后 validation、最后 test 的遍历顺序由
    ``generate`` 模块的去重循环负责，不在返回值里。
    """
    return [(f, n, a) for f in FAMILIES for n in SIZES for a in ALPHAS]
