"""四个合成数据族的纯抽样函数。

本模块只做抽样，不碰文件系统、不含 I/O、不持全局状态。

**抽样顺序是冻结语义的一部分**：每个族内部的 rng 调用次序一经确定即不可
更改，否则会改变已生成语料的全部内容。各族顺序见函数注释。
"""

from __future__ import annotations

import math

import numpy as np

from experiment.data.spec import (
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    skew_k,
)

IntArray = np.ndarray


def round_half_up(x):
    """四舍五入（.5 进位），接受标量或数组。

    刻意不用 ``np.rint``：后者是银行家舍入，把 2.5 舍为 2。设计文档第 6 节
    选定 half-up，与 EXPERIMENT_PLAN.md 第 3.1 节对价格的取整要求一致。

    标量输入返回 ``int``，数组输入返回 ``np.int64`` 数组。
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.floor(arr + 0.5).astype(np.int64)
    return int(out) if arr.ndim == 0 else out


def generate_items(family: str, n: int, seed: int) -> tuple[IntArray, IntArray]:
    """生成一个实例的权重与价值数组。

    Args:
        family: ``uniform`` / ``normal`` / ``correlated`` / ``skewed``
        n: 物品数
        seed: 由 :func:`experiment.data.generate.seeds.derive_seed` 派生

    Returns:
        ``(weights, values)``，均为 ``int64`` 且长度为 ``n``。
    """
    rng = np.random.default_rng(seed)
    return _FAMILY_FUNCS[family](rng, n)


def capacity_for(alpha: float, weights: IntArray) -> int:
    """``C = floor(alpha * sum(w))``。

    三个 alpha 取值都是二进制精确值，浮点乘积对本项目的量级无误差；
    仍显式下取整以匹配规格。
    """
    return math.floor(alpha * int(weights.sum()))


# --- 各族实现（顺序即冻结语义）---------------------------------------------


def _uniform(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, v。"""
    w = rng.integers(1, 101, size=n, dtype=np.int64)
    v = rng.integers(1, 101, size=n, dtype=np.int64)
    return w, v


def _normal(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w 的正态抽样、v 的正态抽样。

    先取整、后裁剪，与 EXPERIMENT_PLAN.md 第 2 节的措辞顺序一致。
    """
    w = np.clip(round_half_up(rng.normal(50.0, 15.0, size=n)), 1, 100).astype(np.int64)
    v = np.clip(round_half_up(rng.normal(50.0, 15.0, size=n)), 1, 100).astype(np.int64)
    return w, v


def _correlated(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, e。``v = max(1, w + e)``。"""
    w = rng.integers(1, 101, size=n, dtype=np.int64)
    e = rng.integers(-10, 11, size=n, dtype=np.int64)
    v = np.maximum(1, w + e).astype(np.int64)
    return w, v


def _skewed(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, d, idx_w, 替换权重, idx_d, 替换密度。

    三处易错点（设计文档第 6 节）：
    1. ``idx_w`` 与 ``idx_d`` 独立抽取，二者**可以重叠**；
    2. ``v = w * d`` 用**替换之后**的 w 与 d；
    3. 极端值一律保留，不做裁剪。
    """
    lo_w, hi_w = SKEW_BASE_WEIGHT_RANGE
    lo_d, hi_d = SKEW_BASE_DENSITY_RANGE
    heavy_lo_w, heavy_hi_w = SKEW_HEAVY_WEIGHT_RANGE
    heavy_lo_d, heavy_hi_d = SKEW_HEAVY_DENSITY_RANGE

    w = rng.integers(lo_w, hi_w + 1, size=n, dtype=np.int64)
    d = rng.integers(lo_d, hi_d + 1, size=n, dtype=np.int64)

    k = skew_k(n)

    idx_w = rng.choice(n, size=k, replace=False)
    w[idx_w] = rng.integers(heavy_lo_w, heavy_hi_w + 1, size=k, dtype=np.int64)

    idx_d = rng.choice(n, size=k, replace=False)
    d[idx_d] = rng.integers(heavy_lo_d, heavy_hi_d + 1, size=k, dtype=np.int64)

    v = (w * d).astype(np.int64)
    return w, v


_FAMILY_FUNCS = {
    "uniform": _uniform,
    "normal": _normal,
    "correlated": _correlated,
    "skewed": _skewed,
}
