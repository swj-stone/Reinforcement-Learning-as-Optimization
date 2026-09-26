import math

import numpy as np
import pytest

from experiment.data.generate.families import (
    capacity_for,
    generate_items,
    round_half_up,
)

N_REPEAT = 2000
#: 用于验证正态族两侧裁剪确实被触发：N(50,15) 越过 [1,100] 的概率约 5e-4，
#: 2000 次抽样有约 1/3 的概率一次都不越界，必须放大样本才能稳定断言。
N_REPEAT_BIG = 200_000


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.5, 1), (1.5, 2), (2.5, 3), (-0.5, 0), (-1.5, -1),
        (0.4, 0), (0.6, 1), (3, 3), (100.5, 101),
    ],
)
def test_round_half_up_scalar(value, expected):
    out = round_half_up(value)
    assert out == expected
    # docstring 承诺「标量输入返回 int」。np.int64 不是 int 的子类，缺了
    # int() 解包这一行 assert out == expected 照样绿——该承诺原本无人看守。
    assert isinstance(out, int)


def test_round_half_up_differs_from_bankers_rounding():
    """np.rint 是银行家舍入（2.5 -> 2），half-up 必须给 3。"""
    assert round_half_up(2.5) == 3
    assert int(np.rint(2.5)) == 2


def test_round_half_up_array():
    out = round_half_up(np.array([0.5, 1.5, 2.5]))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int64
    assert out.tolist() == [1, 2, 3]


def test_uniform_bounds_are_inclusive():
    w, v = generate_items("uniform", N_REPEAT, seed=12345)
    assert w.min() == 1 and w.max() == 100
    assert v.min() == 1 and v.max() == 100


def test_uniform_is_dtype_int64():
    w, v = generate_items("uniform", 20, seed=1)
    assert w.dtype == np.int64 and v.dtype == np.int64
    assert w.shape == (20,) and v.shape == (20,)


def test_uniform_draw_order_is_frozen():
    """w 先抽、v 后抽。顺序颠倒会改变每一个 uniform 实例，而边界、dtype、
    独立性三个测试全都照旧通过——必须靠硬编码期望值才守得住。

    期望值写成字面量而非「在测试里把公式重算一遍」：重算只能守住公式一致，
    守不住 numpy 抽样流本身漂移（Task 11 把 np.__version__ 纳入指纹哈希，
    正是因为升级 numpy 会改变抽样）。
    """
    w, v = generate_items("uniform", 6, seed=7)
    assert w.tolist() == [95, 63, 69, 90, 58, 78]
    assert v.tolist() == [84, 23, 6, 31, 29, 88]


def test_uniform_weights_and_values_are_independent():
    w, v = generate_items("uniform", N_REPEAT, seed=999)
    corr = np.corrcoef(w.astype(float), v.astype(float))[0, 1]
    assert abs(corr) < 0.1


def test_capacity_is_floor_of_alpha_times_total_weight():
    w, _ = generate_items("uniform", 100, seed=7)
    for alpha in (0.25, 0.50, 0.75):
        assert capacity_for(alpha, w) == math.floor(alpha * int(w.sum()))
    # 上面那个实例 sum(w)=5400，是 4 的倍数，三个 alpha 的乘积全是整数，
    # floor/round/ceil/int 四者同值——测试名里的 "floor" 在结构上守不住。
    # 手工构造总和为 5 的实例（非 4 的倍数），令乘积的小数部分非零：
    #   0.25*5 = 1.25 -> floor 1（ceil 给 2）
    #   0.75*5 = 3.75 -> floor 3（round 给 4、ceil 给 4）
    # 这把两个期望值写死；容量决定每个实例的可行集，取整模式一变，
    # 8400 个实例的约束就全变了，而别处没有任何测试能发现。
    odd = np.array([1, 2, 2], dtype=np.int64)
    assert int(odd.sum()) == 5
    assert capacity_for(0.25, odd) == 1
    assert capacity_for(0.75, odd) == 3


def test_capacity_of_alpha_one_equals_total_weight():
    w, _ = generate_items("uniform", 50, seed=8)
    assert capacity_for(1.0, w) == int(w.sum())


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        generate_items("nope", 20, seed=1)
