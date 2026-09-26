import math

import numpy as np
import pytest

from experiment.data.generate.families import (
    capacity_for,
    generate_items,
    round_half_up,
)

N_REPEAT = 2000
#: 用于验证正态族两侧裁剪确实被触发。判据是**取整之后**的值，故门槛落在两个半
#: 整数上，而不是边界本身：`max == 100` 要求原始抽样 >= 99.5（概率 4.83e-4），
#: `min == 1` 要求 < 1.5（概率 6.12e-4）。单个 2000 抽样的样本至少缺一侧的概率
#: 约 0.5625（蒙特卡洛实测 0.556），必须放大样本才能稳定断言。
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


def test_normal_is_clipped_to_1_100():
    w, v = generate_items("normal", N_REPEAT, seed=4242)
    assert w.min() >= 1 and w.max() <= 100
    assert v.min() >= 1 and v.max() <= 100


def test_normal_is_centred_near_50():
    w, _ = generate_items("normal", N_REPEAT, seed=4243)
    assert 48.0 <= w.mean() <= 52.0


def test_normal_reaches_both_clip_bounds():
    """正态(50,15)在 20 万次抽样里必然越过两侧边界，否则裁剪没被触发。

    w 与 v 必须**各自**断言。二者是同分布的两次独立抽样，但只有被断言的那个的
    np.clip 才被守住：把 v 那一行的 clip 删掉，当时全套 51 个测试照样绿——因为在
    test_normal_is_clipped_to_1_100 用的 4242/2000 上，裁剪前的 v 恰好落在
    [2, 96]（w 则越界 2 次），那条 v 断言从来不可能失败。
    """
    w, v = generate_items("normal", N_REPEAT_BIG, seed=4244)
    assert w.min() == 1
    assert w.max() == 100
    assert v.min() == 1
    assert v.max() == 100


def test_correlated_v_equals_max_one_w_plus_e():
    """v - w 就是未被抬升时的 e，其范围必须落在 [-10, 10]。"""
    w, v = generate_items("correlated", N_REPEAT, seed=555)
    diff = v - w
    assert diff.min() >= -10
    assert diff.max() <= 10


def test_correlated_weights_span_full_range():
    w, _ = generate_items("correlated", N_REPEAT, seed=556)
    assert w.min() == 1 and w.max() == 100


def test_correlated_hits_the_floor_of_one():
    """去掉 max(1, ...) 地板后，seed=557 的 2000 次抽样里有 50 个 w+e < 1（最低 -9），
    下面第一行断言随即变红——它才是真正守住地板的那个。
    """
    w, v = generate_items("correlated", N_REPEAT, seed=557)
    assert (v >= 1).all()
    # 这行只是「地板确实被够到过」的抽样 sanity check，对地板本身不敏感：
    # 含地板与不含地板的 v 都能取到 1（例如 w=2、e=-1 时 w+e 本就等于 1）。
    assert (v == 1).any()
    # w=1 处 e 取值限于 [-10,10]，故 v = max(1, 1+e) 只能落在 [1, 11]。该上界同样守不住
    # 地板（去掉地板后仍是 11），它守的是 e 的取值范围不被放宽。
    assert v[w == 1].max() <= 11


def test_normal_draw_order_is_frozen():
    """w 先抽、v 后抽。二者同分布、完全可交换，所以边界、中心、裁剪三处断言
    在顺序颠倒后全部照旧通过——只有硬编码值守得住。

    期望值按规格文本独立重算（normal(50,15) -> half-up -> clip[1,100]），
    不是把实现跑一遍抄下来。
    """
    w, v = generate_items("normal", 6, seed=7)
    assert w.tolist() == [50, 54, 46, 37, 43, 35]
    assert v.tolist() == [51, 70, 43, 41, 57, 55]


def test_correlated_formula_is_frozen():
    """v = max(1, w + e)，且 w 先抽、e 后抽。

    必须把 v 的值也写死：e 对称于 0，故 w - e 的分布与 w + e 完全相同，
    「差落在 [-10,10]」与「能触到 1」两条断言对减号照样全绿，w 更是一字不变。
    这正是本测试名断言的 "+e" 唯一被守住的地方。
    """
    w, v = generate_items("correlated", 6, seed=7)
    assert w.tolist() == [95, 63, 69, 90, 58, 78]
    assert v.tolist() == [102, 57, 60, 86, 53, 86]
