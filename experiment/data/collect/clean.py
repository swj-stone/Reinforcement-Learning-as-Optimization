"""UCI Online Retail 的清洗规则。

依据 EXPERIMENT_PLAN.md 第 3.1 节。本模块是**纯函数**：只接受和返回
DataFrame，不做任何 I/O，因此每条规则都能用内存中的手搓样本测到。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

#: 中间计算的十进制精度。工作簿的价格最多 3 位小数、数量最多 5 位，
#: 28 位远超所需，仅用于杜绝任何隐式截断。
_DECIMAL_PRECISION = 28

_ONE = Decimal(1)
_HUNDRED = Decimal(100)


def to_pence(unit_price, quantity) -> int:
    """``round_half_up(unit_price * quantity * 100)``，全程十进制。

    刻意不走浮点：1.005 的二进制表示略小于 1.005，浮点路径会得到
    100.49999999999999 而被舍成 100 分，十进制路径得到准确的 100.5 而进位为
    101 分。这类偏差在几十万行的合计里会累积成可观测的差。

    Args:
        unit_price: 单价，英镑。接受 float / numpy 浮点 / 字符串数字。
        quantity: 件数，必须是整数。

    Returns:
        整数分。
    """
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        product = Decimal(str(unit_price)) * Decimal(str(quantity)) * _HUNDRED
    return int(product.quantize(_ONE, rounding=ROUND_HALF_UP))
