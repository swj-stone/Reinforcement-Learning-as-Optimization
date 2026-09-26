"""实例的规范化哈希，用于跨 split 去重。

对 ``(w_i, v_i)`` 对排序后再哈希，使物品顺序被打乱的副本得到相同哈希——
EXPERIMENT_PLAN.md 第 2 节明确要求识别 ``copies with reordered items``。
"""

from __future__ import annotations

import hashlib

import numpy as np


def canonical_hash(n: int, capacity: int, weights: np.ndarray, values: np.ndarray) -> str:
    """返回实例的 SHA-256 十六进制摘要。

    参与哈希的内容：物品数、容量、以及**排序后**的 ``(w_i, v_i)`` 对序列。
    物品顺序与实例 ID 不参与，因此重排副本与不同来源的相同实例会被识别为重复。
    """
    pairs = sorted(zip(weights.tolist(), values.tolist()))
    body = ";".join(f"{w},{v}" for w, v in pairs)
    payload = f"{n}|{capacity}|{body}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
