"""UCI Online Retail 的清洗规则。

依据 EXPERIMENT_PLAN.md 第 3.1 节。本模块是**纯函数**：只接受和返回
DataFrame，不做任何 I/O，因此每条规则都能用内存中的手搓样本测到。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

import pandas as pd

from experiment.data.spec import REQUIRED_COLUMNS, SOURCE_COLUMNS, STOCK_CODE_RE

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


def _as_code_string(series: pd.Series) -> pd.Series:
    """把发票号/股票代码规范成字符串，且不带浮点尾巴。

    工作簿里这些列一旦混入空值，pandas 会整列读成 float，``85123`` 就变成
    ``85123.0`` —— 那会被规则 7 的正则判为非法代码而整列误删。这里把整数值
    还原成 ``"85123"``，同时保留 ``"85123A"`` / ``"C536379"`` 这类文本值。
    """
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    integral = numeric.notna() & (numeric % 1 == 0)
    as_int = numeric.astype("Int64").astype("string")
    return text.mask(integral, as_int)


def apply_cleaning(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按 EXPERIMENT_PLAN.md 第 3.1 节的顺序清洗，返回幸存者与归因报告。

    规则是**顺序流水线**：每条规则只在上一条的幸存者上判定。归因取首次命中，
    因此 ``raw_rows == kept_rows + sum(dropped_by_reason.values())`` 恒成立，
    每一行的去向都可单独追溯。

    Returns:
        ``(kept, report)``。``kept`` 含新增列 ``quantity_w`` / ``unit_price`` /
        ``value_pence`` 与 ``source_row_id``；``report`` 含 ``raw_rows``、
        ``kept_rows``、``dropped_by_reason``。
    """
    frame = raw.reset_index(drop=True).copy()
    frame["source_row_id"] = range(len(frame))

    reason = pd.Series("", index=frame.index, dtype="object")
    dropped: dict[str, int] = {}

    def mark(mask: pd.Series, rule: str) -> None:
        """只在尚未归因的行上落这条规则，返回本次新增的丢弃数。"""
        fired = mask.fillna(False).astype(bool) & (reason == "")
        reason.loc[fired] = rule
        dropped[rule] = int(fired.sum())

    # 规则 2：取消发票。大小写不敏感的前缀 C。
    invoice_str = _as_code_string(frame["InvoiceNo"])
    mark(invoice_str.str.upper().str.startswith("C", na=False), "cancellation_invoice")

    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    price = pd.to_numeric(frame["UnitPrice"], errors="coerce")

    # 规则 3：数量或单价非正。NaN 与 0 的比较为 False，非数值行留给规则 6。
    mark((quantity <= 0) | (price <= 0), "nonpositive_quantity_or_price")

    # 规则 4：必需字段缺失。CustomerID 刻意不在必需之列。
    mark(frame[list(REQUIRED_COLUMNS)].isna().any(axis=1), "missing_required_field")

    parsed_date = pd.to_datetime(frame["InvoiceDate"], errors="coerce")

    # 规则 5：时间戳无法解析。文本型日期通过规则 4（不是 NaN），在此被拦下。
    mark(parsed_date.isna(), "invalid_timestamp")

    # 规则 6：数量必须是正整数。小数与非数值一律丢弃，不做静默取整。
    mark(quantity.isna() | (quantity % 1 != 0), "non_integer_quantity")

    # 规则 7：股票代码形态。
    stock_str = _as_code_string(frame["StockCode"])
    mark(~stock_str.str.match(STOCK_CODE_RE, na=False), "invalid_stock_code")

    # 规则 8：全字段精确重复。只在幸存者之间比较——流水线语义下，已被前序
    # 规则丢弃的行不参与去重，否则会把它后面的同内容行一并带走。
    alive = reason == ""
    duplicate = pd.Series(False, index=frame.index)
    duplicate.loc[alive] = frame.loc[alive].duplicated(
        subset=list(SOURCE_COLUMNS), keep="first"
    )
    mark(duplicate, "exact_duplicate_record")

    # 规则 9：换算成分后为 0。单价极小（如 0.001）时半进位会落到 0 分，
    # 这类批次对背包没有贡献，且会让价值为 0 的物品混进实例。
    value_pence = pd.Series(0, index=frame.index, dtype="int64")
    survivors = reason == ""
    # 一个幸存者都没有时（空表，或被前序规则清空）右侧是空列表，
    # 而向 int64 切片赋空列表会抛 LossySetitemError，所以必须先判空。
    if survivors.any():
        value_pence.loc[survivors] = [
            to_pence(p, q) for p, q in zip(price.loc[survivors], quantity.loc[survivors])
        ]
    mark(value_pence == 0, "zero_value_pence")

    # source_row_id 必须一起带出来：它是 lots_clean.csv 的首列，也是事后把
    # 每一批回溯到工作簿原始行的唯一凭据。
    kept_mask = reason == ""
    kept = frame.loc[kept_mask, ["source_row_id", *SOURCE_COLUMNS]].copy()
    kept["invoice_no"] = invoice_str.loc[kept_mask].astype(object)
    kept["stock_code"] = stock_str.loc[kept_mask].astype(object)
    kept["invoice_date"] = parsed_date.loc[kept_mask].dt.strftime("%Y-%m-%d %H:%M:%S")
    kept["quantity_w"] = quantity.loc[kept_mask].astype("int64")
    kept["unit_price"] = price.loc[kept_mask].astype(float)
    kept["value_pence"] = value_pence.loc[kept_mask].astype("int64")

    return kept, {
        "raw_rows": len(frame),
        "kept_rows": int(kept_mask.sum()),
        "dropped_by_reason": dropped,
    }
