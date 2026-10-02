"""两个基线方法的构造（设计 §2）。

本模块是**纯函数**：实例进、解出。不读文件、不写文件、不看时钟——计时归
``cli.py``。这样"方法是否正确"与"产物是否正确"可以分开测，两个方法也能在测试
里被直接调用，不必造配置文件或临时目录。
"""

from __future__ import annotations

import random
from typing import Mapping, NamedTuple, Sequence

from experiment.eval.reference import ratio_order


class BaselineSolution(NamedTuple):
    """一个基线的解：设计 §8 的三列原始观测。

    **不复用 A 的 ``ReferenceSolution``**：它带 ``upper_bound`` 与 ``certified``，
    基线产不出这两个量，硬塞进去只能填假值——而一个填了假值的字段迟早会被当成
    真值读。
    """

    #: 被选物品索引，**按 ID 升序**。
    #:
    #: 升序不是为了好看：``item_set_hash`` 对输入次序本就不敏感，而升序让"两次
    #: 运行逐格比对"这种最朴素的复核也能直接用。
    selected_indices: tuple[int, ...]

    #: 被选集合的价值总和，**原始单位**（不是奖励整形后的值）。
    value: int

    #: 被选集合占用的容量。可行的解满足 ``<= instance["capacity"]``。
    weight_used: int


def _totals(instance: Mapping, selected: Sequence[int]) -> tuple[int, int]:
    """由集合现算 ``(value, weight_used)``。

    刻意不把这两个数作为独立状态维护：三份数据各自漂移是本项目的旧病，而它们
    必须永远一致。
    """
    values, weights = instance["values"], instance["weights"]
    return (
        sum(values[i] for i in selected),
        sum(weights[i] for i in selected),
    )


def greedy_solution(instance: Mapping) -> BaselineSolution:
    """经典 0/1 贪心：按 ``v/w`` 降序扫描，装得下就装。

    依据 `EXPERIMENT_PLAN.md` 第 5 节。排序用 ``ratio_order``——**与分数背包上界
    ``U`` 共用同一个函数**，理由见设计 §6：两份实现会在某次修订后分歧，而症状是
    "上界与贪心用了不同的次序"。

    **装不下时跳过继续，不是停下。** ``break`` 是更"自然"的写法，而且它在多数
    实例上给出同样的答案——只有在一个轻物品被重物品挡在后面时才暴露。计划 §5
    专门写了 "continue after a rejected item"，所以它由合成实例钉住（本文件的
    ``test_a_rejected_item_does_not_stop_the_scan``）。

    **不使用 ``item_order``**：它自己排序。实例里那份存储排列只对 Random 有意义。
    """
    weights = instance["weights"]
    remaining = instance["capacity"]
    selected: list[int] = []
    for i in ratio_order(weights, instance["values"]):
        if weights[i] > remaining:
            continue
        selected.append(i)
        remaining -= weights[i]
    selected.sort()
    return BaselineSolution(tuple(selected), *_totals(instance, selected))


def random_solution(instance: Mapping, seed: int) -> BaselineSolution:
    """随机基线：沿实例自带的 ``item_order`` 走一遍，逐件掷硬币。

    ``item_order`` 是实例记录里的**存储排列**（计划 §5 的 "stored item
    permutation"）：`generate/corpus.py` 与 `assemble/{real,stress}.py` 都由一条
    独立的 order 随机流生成它。它在 10 次试验之间**固定不变**，所以一次试验是沿
    它走，而不是沿索引 ``0..n-1`` 走——后者产出的仍是一个确定的、可复现的 CSV，
    只是它不是计划描述的那个实验。

    **只有装得下的物品才铸币**（设计 §2.2 的读法 (a)）。读法 (b)（每件都掷、
    装不下才丢弃）同样确定、同样跨版本可复现，所以"种子可复现"这条验收**区分不了**
    两者；区分靠 ``tests/test_baselines_random.py`` 里的对照实现。

    **平均由 D 做，本函数不取最好**：计划 §5 的 "Average outcomes rather than
    selecting the best trial" 是一句禁令，best-of-ten 同样确定、同样可复现，
    只是数字好看得多。

    随机源的用法（设计 §2.2，三条都是可复现性的依据）：
      - 种子传 **int**，不传 ``str``/``bytes``：``random.seed`` 对 ``str``/``bytes``
        的哈希经历过 ``version=`` 语义变更；
      - 用 ``random()`` 而不是 ``getrandbits()``：标准库保证 ``random()`` 在给定
        兼容种子时**跨版本产出同一序列**；
      - 用**实例化**的 ``Random(seed)``，不用模块级的 ``random.seed()``：后者是
        全局状态，会把"这一次试验"绑定到调用顺序上，并让并行的测试互相污染。
    """
    weights = instance["weights"]
    rng = random.Random(seed)
    remaining = instance["capacity"]
    selected: list[int] = []
    for i in instance["item_order"]:
        if weights[i] > remaining:
            continue  # 装不下：跳过，**不铸币**
        if rng.random() < 0.5:  # 装得下才铸币
            selected.append(i)
            remaining -= weights[i]
    selected.sort()
    return BaselineSolution(tuple(selected), *_totals(instance, selected))
