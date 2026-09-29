"""两个应力集：E3（未见过的规模）与 E4（更强的不平衡）。

E3 用 ``n=500`` 检验规模泛化，保留 50:50 来源平衡；E4 用 ``n=200`` 的
强不平衡合成实例检验极值表现。依据 EXPERIMENT_PLAN.md 第 8 节。

**两条保留流的隔离靠字符串记号，不靠 ``STRESS_MASTER_SEEDS`` 里的数字。**
``derive_seed`` 的键里没有那个数字——这是设计文档 §12.1 记录的已知事实。
"""

from __future__ import annotations

import numpy as np

from experiment.data.assemble.lots import LotIndex
from experiment.data.assemble.real import build_real_cell, build_real_instance, fill_cell
from experiment.data.generate.families import capacity_for, compute_stats, generate_items
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    ALPHAS,
    E3_NAME,
    E3_REAL_PER_ALPHA,
    E3_SIZE,
    E3_SYNTHETIC_PER_FAMILY,
    E4_COUNT_PER_ALPHA,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    FAMILIES,
    REAL_SEED_TOKEN,
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    alpha_token,
)


def generate_skewed_e4(
    n: int,
    seed: int,
    with_density: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E4 的强不平衡变体生成器（第 8 节）。

    与 ``generate.families._skewed`` 的关系是**另立变体，不是改版**：主语料的
    skewed 半边已被金样本钉死，改它会作废全部既有语料。这里按同一套配方重写，
    只换参数：

    ==========  =============================  ======================
    参数         主语料 skewed                   E4 变体
    ==========  =============================  ======================
    稀有索引数    ``max(1, floor(0.05n+0.5))``   ``k = 2``
    稀有权重      ``U(200,500)``                 ``U(500,2000)``
    稀有密度      ``U(30,60)``                   ``U(100,200)``
    ==========  =============================  ======================

    rng 调用次序与 ``_skewed`` 完全一致（w, d, idx_w, 替换权重, idx_d,
    替换密度），这个次序本身是冻结语义。两处易错点同样适用：两组稀有索引
    独立抽取、**可以重叠**；``v = w * d`` 用替换**之后**的 w 与 d。

    Args:
        with_density: 为 ``True`` 时额外返回密度数组。密度无法从 ``v = w*d``
            反推，而测试需要直接检验稀有密度的个数与范围。
    """
    rng = np.random.default_rng(seed)
    lo_w, hi_w = SKEW_BASE_WEIGHT_RANGE
    lo_d, hi_d = SKEW_BASE_DENSITY_RANGE
    heavy_lo_w, heavy_hi_w = E4_SKEW_HEAVY_WEIGHT_RANGE
    heavy_lo_d, heavy_hi_d = E4_SKEW_HEAVY_DENSITY_RANGE

    w = rng.integers(lo_w, hi_w + 1, size=n, dtype=np.int64)
    d = rng.integers(lo_d, hi_d + 1, size=n, dtype=np.int64)

    k = E4_SKEW_K

    idx_w = rng.choice(n, size=k, replace=False)
    w[idx_w] = rng.integers(heavy_lo_w, heavy_hi_w + 1, size=k, dtype=np.int64)

    idx_d = rng.choice(n, size=k, replace=False)
    d[idx_d] = rng.integers(heavy_lo_d, heavy_hi_d + 1, size=k, dtype=np.int64)

    values = (w * d).astype(np.int64)
    if with_density:
        return w, values, d
    return w, values


def _finish(
    instance_id: str,
    family: str | None,
    source: str,
    n: int,
    alpha: float,
    order_seed: int,
    weights: np.ndarray,
    values: np.ndarray,
    extra: dict,
) -> dict:
    """把一份 (weights, values) 补成完整的合成实例记录。

    两条合成分支共用，避免把 13 个字段抄两遍。``order_seed`` 是 **order 流**
    的种子，不是 items 流的——调用方必须传对，否则排列会与主集的约定不一致。
    """
    capacity = capacity_for(alpha, weights)
    record = {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": family,
        "instance_id": instance_id,
        "item_order": np.random.default_rng(order_seed).permutation(n).tolist(),
        "n": n,
        "source": source,
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }
    record.update(extra)
    return record


def build_e3_synthetic(
    family: str,
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E3_SIZE,
) -> dict:
    """E3 的一条合成实例。

    ``n`` 默认是冻结的 500；测试可传小值以便在密闭夹具上跑。种子键的 split
    槽位是保留流记号 ``stream_size``，与主集的任何 cell 都不重合。
    """
    seed = derive_seed(STRESS_SIZE_STREAM, family, n, alpha, index, attempt, "items")
    weights, values = generate_items(family, n, seed)
    order_seed = derive_seed(STRESS_SIZE_STREAM, family, n, alpha, index, attempt, "order")
    return _finish(
        f"syn-{E3_NAME}-{family}-n{n}-{alpha_token(alpha)}-i{index:04d}",
        family,
        "synthetic",
        n,
        alpha,
        order_seed,
        weights,
        values,
        {"resample_attempt": attempt, "seed": seed},
    )


def build_e3_real(
    lot_index: LotIndex,
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E3_SIZE,
) -> dict:
    """E3 的一条真实实例：从 **test** 的日期池抽，种子在保留流上。

    第 8 节要求 E3 真实只用合格的 test 源日期，且与主集的 canonical hash
    不相交——后者由 ``seen`` 的全局性保证，不是本函数的事。
    """
    return build_real_instance(
        lot_index, E3_NAME, "test", n, alpha, index, attempt, seed_stream=STRESS_SIZE_STREAM
    )


def build_e4_synthetic(
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E4_SIZE,
) -> dict:
    """E4 的一条合成实例。族记号是 ``skewed_e4``，不在 FAMILIES 中。"""
    seed = derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, n, alpha, index, attempt, "items")
    weights, values = generate_skewed_e4(n, seed)
    order_seed = derive_seed(
        STRESS_TAIL_STREAM, E4_SEED_TOKEN, n, alpha, index, attempt, "order"
    )
    return _finish(
        f"syn-{E4_NAME}-{E4_SEED_TOKEN}-n{n}-{alpha_token(alpha)}-i{index:04d}",
        E4_SEED_TOKEN,
        "synthetic",
        n,
        alpha,
        order_seed,
        weights,
        values,
        {"resample_attempt": attempt, "seed": seed},
    )


def build_e3(
    lot_index: LotIndex | None,
    seen: set[str] | None = None,
    n: int = E3_SIZE,
    synthetic_per_family: int = E3_SYNTHETIC_PER_FAMILY,
    real_per_alpha: int = E3_REAL_PER_ALPHA,
) -> list[dict]:
    """E3 全集：合成在前（族 × 比率），真实在后（比率）。

    ``lot_index`` 为 ``None`` 时只造合成部分，供不涉及真实数据的测试使用。
    """
    book = set() if seen is None else seen
    instances: list[dict] = []
    for family in FAMILIES:
        for alpha in ALPHAS:
            for index in range(synthetic_per_family):
                instances.extend(
                    fill_cell(
                        lambda attempt, f=family, a=alpha, i=index: build_e3_synthetic(
                            f, a, i, attempt, n
                        ),
                        quota=1,
                        seen=book,
                    )
                )
    if lot_index is not None:
        for alpha in ALPHAS:
            instances.extend(
                build_real_cell(
                    lot_index,
                    E3_NAME,
                    "test",
                    n,
                    alpha,
                    real_per_alpha,
                    book,
                    seed_stream=STRESS_SIZE_STREAM,
                )
            )
    return instances


def build_e4(
    seen: set[str] | None = None,
    n: int = E4_SIZE,
    per_alpha: int = E4_COUNT_PER_ALPHA,
) -> list[dict]:
    """E4 全集：每比率 ``per_alpha`` 条合成实例。"""
    book = set() if seen is None else seen
    instances: list[dict] = []
    for alpha in ALPHAS:
        for index in range(per_alpha):
            instances.extend(
                fill_cell(
                    lambda attempt, a=alpha, i=index: build_e4_synthetic(a, i, attempt, n),
                    quota=1,
                    seen=book,
                )
            )
    return instances
