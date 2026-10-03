"""试验种子的派生（设计 §4）。

一个随机试验的身份是三元组 ``(evaluation_seed, instance_id, trial)``。本模块
把它映射成一个确定的 64 位整数种子。

**刻意不用 Python 的 ``hash()``**：``hash(str)`` 每个进程用不同的随机盐
（PYTHONHASHSEED），进程内自洽、跨进程不稳定。而 ``baseline_results.csv`` 是要
入库的产物，"换个进程就变"等于复现性直接失效。BLAKE2b 没有这个问题。
"""

from __future__ import annotations

import hashlib

#: 摘要取 8 字节 → 种子恰好落在 64 位整数范围内，不必再掩码。
#:
#: 取 4 字节也"够用"，但 33,000 次抽取在 2**32 空间里的生日碰撞概率约 12%
#: （在 2**64 里约 3e-11）。碰撞意味着两次"独立"的试验抽到同一串硬币，
#: 取平均时它们是重复计数。取满 8 字节没有额外成本。
_DIGEST_BYTES = 8


def trial_seed(instance_id: str, trial: int, base: int) -> int:
    """``(base, instance_id, trial)`` → 稳定的 64 位种子。

    键序刻意写成 ``base | instance_id | trial``（与实参顺序不同）：三段用 ``|``
    分隔。语料的 ``instance_id`` 不含 ``|``（实测 3,300/3,300），
    ``tests/test_baselines_corpus.py`` 有一条断言钉住语料的这一点——但那条引用
    **并不承重**：``base`` 与 ``trial`` 是十进制整数、分隔符恰好两个，按**首个**与
    **末个** ``|`` 切分即可唯一还原三段，所以即便某个 ``instance_id`` 含 ``|``，
    也不会与另一组三元组撞成同一个键（探针实测：16,000 组含 ``|`` 的
    ``(base, id, trial)`` 无一碰撞）。那条语料断言挡的是"语料里混进了意外的
    ``|``"这个事实，而不是派生算法的正确性。

    种子是**键的函数**，不依赖任何全局计数器，也不依赖语料的文件顺序：同一次
    试验在两次运行里、在两台机器上、在并行测试中都是同一个种子。
    """
    payload = "|".join((str(base), instance_id, str(trial)))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES)
    return int.from_bytes(digest.digest(), "big")
