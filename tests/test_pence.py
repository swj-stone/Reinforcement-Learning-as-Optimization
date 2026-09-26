import math
from decimal import Decimal

import numpy as np
import pytest

from experiment.data.collect.clean import to_pence


def test_basic_conversion():
    assert to_pence(2.55, 6) == 1530


def test_integer_quantity_and_price():
    assert to_pence(1.0, 1) == 100


def test_single_item_price():
    assert to_pence(0.42, 3) == 126


def test_uses_half_up_not_bankers():
    """2.125 * 100 = 212.5，half-up 给 213，银行家舍入给 212。"""
    assert to_pence(2.125, 1) == 213


def test_uses_decimal_not_binary_float():
    """本测试是 Decimal 语义的判别式。

    1.005 的二进制表示略小于 1.005，浮点路径 1.005*100 = 100.49999999999999，
    floor(+0.5) 得 100；十进制路径 Decimal("1.005")*100 = 100.5，half-up 得 101。
    若本测试失败，说明实现退回了浮点运算。
    """
    assert to_pence(1.005, 1) == 101
    assert math.floor(1.005 * 1 * 100 + 0.5) == 100


def test_accepts_numpy_scalars():
    assert to_pence(np.float64(2.55), np.int64(6)) == 1530


def test_tiny_price_rounds_to_zero():
    assert to_pence(0.001, 1) == 0


def test_returns_python_int():
    result = to_pence(1.25, 2)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


def test_result_is_exact_not_approximate():
    """1000 件单价 0.001 应为 100 分，不能有浮点尾差。"""
    assert to_pence(0.001, 1000) == 100
