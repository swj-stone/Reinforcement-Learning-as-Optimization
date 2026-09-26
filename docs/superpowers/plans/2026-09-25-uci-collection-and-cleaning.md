# UCI 采集与清洗 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 下载 UCI Online Retail 工作簿，按 EXPERIMENT_PLAN.md 第 3.1 节的规则清洗成"批次表"，落到 `datasets/raw/real/`，并产出可审计的 `collect_report.json`。

**Architecture:** 清洗是**纯函数**——输入一个 DataFrame，输出清洗后的 DataFrame 与逐规则丢弃计数，完全不碰文件系统，因此每个规则都能用内存里的手搓样本测到；I/O 与网络被隔离在 `download.py` 与薄薄一层 `run_collection()` 里。规则按顺序流水线执行，每条只处理上一条的幸存者。

**Tech Stack:** Python 3.12、pandas、openpyxl、`decimal.Decimal`、标准库 `urllib.request`、pytest

**设计依据:** [../specs/2026-09-25-raw-data-collection-and-generation-design.md](../specs/2026-09-25-raw-data-collection-and-generation-design.md)

**前置依赖:** 本计划假定 `2026-09-25-synthetic-corpus-generation.md` 的 Task 2 已完成，即 `experiment/data/spec.py`、`experiment/data/__init__.py`、`tests/__init__.py`、`pyproject.toml` 均已存在。本计划只向 `spec.py` **追加**采集相关常量，不修改已有内容。

**工作目录:** 所有相对路径均以仓库根目录为基准。

> **执行须知（PowerShell）：** 本计划的 PowerShell 片段全部假定当前目录已是仓库根。
> 若非如此，`New-Item` / `Remove-Item` / `Copy-Item` 会**静默地在错误位置**执行或以
> 非终止错误失败，而后续的 `"已创建"` 之类打印照常输出，造成成功假象。
> 每条 PowerShell 命令前先 `Set-Location <仓库根>`，且判定成败要看**复查**（`Test-Path`、
> `Get-FileHash`），不要看创建命令有没有报错。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `experiment/data/spec.py` | **修改**：追加 UCI 来源常量、源列名、必需列名、清洗规则名 |
| `experiment/data/collect/__init__.py` | 包标记 |
| `experiment/data/collect/clean.py` | `to_pence()`、`apply_cleaning()`、`build_lots()`、`write_lots_csv()` |
| `experiment/data/collect/download.py` | `fetch_zip()`、`extract_entry()`、`sha256_file()`、`ensure_workbook()` |
| `experiment/data/collect/run.py` | `run_collection()`：把上面两者串起来并写报告 |
| `experiment/data/collect/cli.py` | 命令行入口 |
| `tests/test_pence.py` | 分币换算的十进制语义 |
| `tests/test_cleaning.py` | 八条规则的逐条归因与总量恒等式 |
| `tests/test_lots.py` | 输出表结构与 CSV 落盘 |
| `tests/test_collect_io.py` | 下载/解压/校验的离线部分 |
| `tests/test_collect_integration.py` | 真实工作簿的端到端校验（未下载则跳过） |

---

## Task 1: 采集常量与分币换算

**Files:**
- Modify: `experiment/data/spec.py`（在 `# --- 输出路径` 之前插入一节）
- Create: `experiment/data/collect/__init__.py`
- Create: `tests/test_pence.py`

- [ ] **Step 1: 建包标记**

```powershell
$p = 'experiment\data\collect\__init__.py'
New-Item -ItemType Directory -Force (Split-Path $p) | Out-Null
if (-not (Test-Path $p)) { New-Item -ItemType File $p | Out-Null; "created $p" }
```
Expected: 打印 `created experiment\data\collect\__init__.py`。

- [ ] **Step 2: 写失败测试**

`tests/test_pence.py`：
```python
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
```

- [ ] **Step 3: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_pence.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.collect.clean'`

- [ ] **Step 4: 向 spec.py 追加常量**

在 `experiment/data/spec.py` 中，把 `# --- 输出路径 ---` 那一节之前插入：
```python
# --- UCI Online Retail 采集 --------------------------------------------------

UCI_URL: str = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_ZIP_ENTRY: str = "Online Retail.xlsx"
UCI_SHEET_NAME: str = "Online Retail"
#: 冻结的工作簿字节数。UCI 若替换了文件，此校验会立刻失败，而不是静默换数据。
UCI_XLSX_BYTES: int = 23715344
#: 仓库中原始工作簿的落盘名
UCI_XLSX_NAME: str = "online_retail.xlsx"
UCI_ZIP_NAME: str = "online_retail.zip"
#: 部分镜像会拒绝没有 User-Agent 的请求
USER_AGENT: str = "rl-knapsack-seminar/0.1 (+educational use)"

#: 工作簿的全部源列，顺序即规则 8「全字段重复」的比较字段
SOURCE_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)

#: 缺失即丢弃的列。CustomerID 刻意不在其中：它经常为空，且不进模型特征。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
)

#: 保留的股票代码形态：五位数字，可带字母后缀
STOCK_CODE_RE: str = r"^\d{5}[A-Za-z]*$"

#: 清洗流水线的规则名，顺序即执行顺序，也是报告中的键顺序
CLEANING_RULES: tuple[str, ...] = (
    "cancellation_invoice",
    "nonpositive_quantity_or_price",
    "missing_required_field",
    "invalid_timestamp",
    "non_integer_quantity",
    "invalid_stock_code",
    "exact_duplicate_record",
    "zero_value_pence",
)

#: lots_clean.csv 的列顺序
LOTS_COLUMNS: tuple[str, ...] = (
    "source_row_id",
    "invoice_no",
    "stock_code",
    "invoice_date",
    "quantity_w",
    "unit_price",
    "value_pence",
)
```

- [ ] **Step 5: 写实现**

`experiment/data/collect/clean.py`：
```python
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
```

- [ ] **Step 6: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_pence.py -v`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add experiment/data/spec.py experiment/data/collect tests/test_pence.py
git commit -m "feat(data): UCI 采集常量与十进制分币换算"
```

---

## Task 2: 清洗流水线的前四条规则

**Files:**
- Modify: `experiment/data/collect/clean.py`（追加 `apply_cleaning` 骨架与前四条：规则 1 是 `source_row_id` 赋值，规则 2–4 是丢弃规则）
- Create: `tests/test_cleaning.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cleaning.py`：
```python
"""八条规则的逐条归因测试。

样本表是手搓的：每一行恰好触发一条规则（或全部通过），因此丢弃计数是
精确值而不是范围。这样一旦规则顺序被改动，测试会立刻指出是哪一条。
"""

import pandas as pd
import pytest

from experiment.data.collect.clean import apply_cleaning
from experiment.data.spec import CLEANING_RULES, SOURCE_COLUMNS

KEPT_ID = 0
KEPT_ID_2 = 1
KEPT_ID_MISSING_CUSTOMER = 5
CANCELLATION_ID = 2
NONPOSITIVE_QTY_ID = 3
NONPOSITIVE_PRICE_ID = 4
MISSING_STOCK_ID = 6
BAD_TIMESTAMP_ID = 7
FRACTIONAL_QTY_ID = 8
NONNUMERIC_QTY_ID = 9
BAD_STOCK_ID = 10
DUPLICATE_ID = 11
ZERO_PENCE_ID = 12


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = [
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536365, "71053", "WHITE METAL LANTERN", 6, "2010-12-01 08:26:00", 3.39, 17850.0, "United Kingdom"),
        ("C536379", "D", "Discount", -1, "2010-12-01 09:41:00", 27.50, 14527.0, "United Kingdom"),
        (536367, "84879", "ASSORTED COLOUR BIRD", -4, "2010-12-01 09:34:00", 1.69, 13047.0, "United Kingdom"),
        (536368, "84879", "ADJUSTABLE", 1, "2010-12-01 09:34:00", 0.0, 13047.0, "United Kingdom"),
        (536369, "22139", "RETROSPOT", 2, "2010-12-01 09:34:00", 1.25, None, "United Kingdom"),
        (536370, None, "?", 1, "2010-12-01 09:34:00", 1.0, 13047.0, "United Kingdom"),
        (536371, "22139", "RETROSPOT", 1, "not-a-date", 1.25, 13047.0, "United Kingdom"),
        (536372, "22139", "RETROSPOT", 1.5, "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536373, "22139", "RETROSPOT", "abc", "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536374, "ABC", "RETROSPOT", 1, "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536375, "22139", "RETROSPOT", 1, "2010-12-01 09:34:00", 0.001, 13047.0, "United Kingdom"),
    ]
    return pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))


@pytest.fixture
def dropped(raw_frame) -> dict[str, int]:
    _, report = apply_cleaning(raw_frame)
    return report["dropped_by_reason"]


@pytest.fixture
def kept(raw_frame) -> pd.DataFrame:
    frame, _ = apply_cleaning(raw_frame)
    return frame


def test_source_row_id_is_the_original_position(raw_frame):
    frame, _ = apply_cleaning(raw_frame)
    assert frame["source_row_id"].tolist() == [KEPT_ID, KEPT_ID_2, KEPT_ID_MISSING_CUSTOMER]


def test_cancellation_is_dropped(dropped):
    assert dropped["cancellation_invoice"] == 1


def test_cancellation_is_attributed_before_nonpositive(dropped):
    """C 开头的发票同时数量为负；归因必须落在取消发票上，而不是数量非正。"""
    assert dropped["cancellation_invoice"] == 1
    assert dropped["nonpositive_quantity_or_price"] == 2


def test_nonpositive_quantity_and_price_are_dropped(dropped):
    assert dropped["nonpositive_quantity_or_price"] == 2


def test_missing_required_field_is_dropped(dropped):
    assert dropped["missing_required_field"] == 1


def test_missing_customer_id_is_not_a_missing_required_field(dropped, kept):
    """CustomerID 允许为空：样本里那一行必须在幸存者中。"""
    assert dropped["missing_required_field"] == 1
    assert KEPT_ID_MISSING_CUSTOMER in kept["source_row_id"].tolist()


def test_later_rules_do_not_fire_on_this_sample(dropped):
    for rule in ("invalid_timestamp", "non_integer_quantity", "invalid_stock_code",
                 "exact_duplicate_record", "zero_value_pence"):
        assert dropped[rule] == 0, rule


def test_report_keys_are_the_frozen_rules(raw_frame):
    _, report = apply_cleaning(raw_frame)
    assert tuple(report["dropped_by_reason"]) == CLEANING_RULES
    assert report["raw_rows"] == 13
    assert report["kept_rows"] == 3


def test_empty_frame_is_handled():
    frame, report = apply_cleaning(pd.DataFrame(columns=list(SOURCE_COLUMNS)))
    assert frame.empty
    assert report["raw_rows"] == 0
    assert report["kept_rows"] == 0
    assert all(count == 0 for count in report["dropped_by_reason"].values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_cleaning.py -v`
Expected: FAIL，`ImportError: cannot import name 'apply_cleaning'`

- [ ] **Step 3: 写实现**

在 `experiment/data/collect/clean.py` 的 `to_pence` 之后追加：
```python
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

    kept_mask = reason == ""
    return frame.loc[kept_mask].copy(), {
        "raw_rows": len(frame),
        "kept_rows": int(kept_mask.sum()),
        "dropped_by_reason": dropped,
    }
```

同时在文件顶部补 import：
```python
import pandas as pd

from experiment.data.spec import REQUIRED_COLUMNS
```

并在 `apply_cleaning` 之前追加私有帮手：
```python
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
```

- [ ] **Step 4: 运行测试，确认失败恰好是预期的三个**

Run: `venv\Scripts\python.exe -m pytest tests/test_cleaning.py -v`
Expected: **6 passed, 3 failed**。失败的必须是这三个：

- `test_source_row_id_is_the_original_position` — 此刻幸存者仍是 `[0, 1, 5, 7, 8, 9, 10, 11, 12]`。样本第 7–12 行只有规则 5–9 才拦得住，而本任务只实现规则 2–4，所以它们仍然被保留
- `test_later_rules_do_not_fire_on_this_sample` — `KeyError: 'invalid_timestamp'`，因为报告尚无规则 5–9 的键
- `test_report_keys_are_the_frozen_rules` — 报告键不全（此时只有 3 个），`kept_rows` 也还是 9 而不是 3

其余 6 个（四条归因、CustomerID 不算缺失、空表处理）必须已经通过。**若失败的不是这三个，先停下来排查，不要继续。**

- [ ] **Step 5: 暂不提交**

本任务与 Task 3 共同构成一次可提交的完整改动，在 Task 3 结束时一并提交。

---

## Task 3: 后五条规则与总量恒等式

**Files:**
- Modify: `experiment/data/collect/clean.py`（补齐规则 5–9 与返回值）
- Modify: `tests/test_cleaning.py`（删除一个过时用例，再追加测试）

- [ ] **Step 1: 删除过时用例，再追加失败测试**

先删除 `tests/test_cleaning.py` 中的这个用例（任务二留下的脚手架，在文件中倒数第三个）：

    def test_later_rules_do_not_fire_on_this_sample(dropped):
        for rule in ("invalid_timestamp", "non_integer_quantity", "invalid_stock_code",
                     "exact_duplicate_record", "zero_value_pence"):
            assert dropped[rule] == 0, rule

把它和它后面的两个空行一并删除，使前后两个用例之间恰好保留两个空行（PEP 8）。它前面已有两个空行，删掉后面那组才刚好剩下一组。

它断言规则 5–9 在样本上一律不触发，而这只在「这五条规则尚未实现」时成立；本任务恰好要实现它们，所以它必然变红。它编码的是当时的开发进度，不是系统的性质——按设计删除，不要改断言去迁就它。

然后在同一文件末尾追加：
```python
def test_invalid_timestamp_is_dropped(dropped):
    assert dropped["invalid_timestamp"] == 1


def test_fractional_quantity_is_dropped_not_rounded(dropped, kept):
    """1.5 件必须被丢弃，而不是静默取整成 1 或 2。"""
    assert dropped["non_integer_quantity"] == 2
    assert FRACTIONAL_QTY_ID not in kept["source_row_id"].tolist()


def test_nonnumeric_quantity_is_dropped(dropped):
    """非数值数量归入「非整数数量」，而不是「字段缺失」。"""
    assert dropped["non_integer_quantity"] == 2
    assert dropped["missing_required_field"] == 1


def test_invalid_stock_code_is_dropped(dropped):
    assert dropped["invalid_stock_code"] == 1


def test_duplicate_record_keeps_the_smallest_source_row_id(dropped, kept):
    assert dropped["exact_duplicate_record"] == 1
    assert DUPLICATE_ID not in kept["source_row_id"].tolist()
    assert KEPT_ID in kept["source_row_id"].tolist()


def test_zero_value_pence_is_dropped(dropped):
    assert dropped["zero_value_pence"] == 1


def test_identity_holds(raw_frame):
    """总账恒等式：原始行数 = 幸存行数 + 各规则丢弃数之和。"""
    _, report = apply_cleaning(raw_frame)
    dropped = report["dropped_by_reason"]
    assert sum(dropped.values()) == 10
    assert report["kept_rows"] == 3
    assert report["raw_rows"] == 13
    assert report["raw_rows"] == report["kept_rows"] + sum(dropped.values())


def test_kept_rows_are_exactly_the_expected_three(kept):
    assert kept["source_row_id"].tolist() == [
        KEPT_ID, KEPT_ID_2, KEPT_ID_MISSING_CUSTOMER
    ]


def test_kept_rows_carry_computed_pence(kept):
    by_id = kept.set_index("source_row_id")
    assert by_id.loc[KEPT_ID, "value_pence"] == 1530      # 2.55 * 6
    assert by_id.loc[KEPT_ID_2, "value_pence"] == 2034    # 3.39 * 6
    assert by_id.loc[KEPT_ID_MISSING_CUSTOMER, "value_pence"] == 250   # 1.25 * 2


def test_quantity_w_is_integer_typed(kept):
    assert kept["quantity_w"].tolist() == [6, 6, 2]
    assert all(isinstance(x, int) for x in kept["quantity_w"].tolist())


def test_duplicate_is_judged_among_survivors_only():
    """一条已被前序规则丢弃的行，不应把它后面的同内容行「带走」。

    构造：第 0 行是取消发票，第 1 行与它逐字段相同。第 0 行按规则 2 丢弃；
    第 1 行若按「与原始表比较」会被判重复而丢弃，按「只与幸存者比较」则应保留。
    流水线语义要求后者。
    """
    rows = [
        ("C536379", "85123A", "X", -1, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        ("C536379", "85123A", "X", -1, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
    ]
    frame, report = apply_cleaning(pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)))
    assert report["dropped_by_reason"]["cancellation_invoice"] == 2
    assert report["dropped_by_reason"]["exact_duplicate_record"] == 0
    assert report["kept_rows"] == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_cleaning.py -v`
Expected: FAIL，`KeyError: 'invalid_timestamp'` 等

- [ ] **Step 3: 写实现**

把 `experiment/data/collect/clean.py` 中 `apply_cleaning` 的末尾（`return frame, {...}` 那一段）替换为：
```python
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
```

并把顶部 import 补为：
```python
import pandas as pd

from experiment.data.spec import REQUIRED_COLUMNS, SOURCE_COLUMNS, STOCK_CODE_RE
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_cleaning.py -v`
Expected: **19 passed**（任务二留下的 8 个 + 本任务新增的 11 个）。若仍有一个 `test_later_rules_do_not_fire_on_this_sample` 失败，说明 Step 1 里那个过时用例还没删。

- [ ] **Step 5: 全套测试**

Run: `venv\Scripts\python.exe -m pytest -v`
Expected: 全部 PASS（含计划一的所有测试）

- [ ] **Step 6: Commit**

```bash
git add experiment/data/collect/clean.py tests/test_cleaning.py
git commit -m "feat(data): UCI 清洗流水线的八条规则与逐条归因"
```

---

## Task 4: 输出表结构与 CSV 落盘

**Files:**
- Modify: `experiment/data/collect/clean.py`（追加 `build_lots`、`write_lots_csv`）
- Create: `tests/test_lots.py`

- [ ] **Step 1: 写失败测试**

`tests/test_lots.py`：
```python
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiment.data.collect.clean import apply_cleaning, build_lots, write_lots_csv
from experiment.data.spec import LOTS_COLUMNS, SOURCE_COLUMNS


@pytest.fixture
def lots() -> pd.DataFrame:
    rows = [
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536369, "22139", "RETROSPOT", 2, "2010-12-01 09:34:00", 1.25, None, "United Kingdom"),
    ]
    frame, _ = apply_cleaning(pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)))
    return build_lots(frame)


def test_lots_has_exactly_the_frozen_columns(lots):
    assert tuple(lots.columns) == LOTS_COLUMNS


def test_customer_and_descriptive_text_are_excluded(lots):
    """EXPERIMENT_PLAN.md 第 3.1 节：客户标识与描述文本不进模型特征。"""
    for column in ("CustomerID", "customer_id", "Description", "description", "Country"):
        assert column not in lots.columns


def test_lots_values(lots):
    assert lots["invoice_no"].tolist() == ["536365", "536369"]
    assert lots["stock_code"].tolist() == ["85123A", "22139"]
    assert lots["quantity_w"].tolist() == [6, 2]
    assert lots["value_pence"].tolist() == [1530, 250]


def test_invoice_date_is_iso_text(lots):
    assert lots["invoice_date"].tolist() == [
        "2010-12-01 08:26:00", "2010-12-01 09:34:00"
    ]


def test_source_row_id_is_preserved(lots):
    assert lots["source_row_id"].tolist() == [0, 1]


def test_write_lots_csv_creates_lf_terminated_file(tmp_path, lots):
    path = tmp_path / "lots_clean.csv"
    write_lots_csv(path, lots)
    raw = path.read_bytes()
    assert b"\r" not in raw, "必须是 LF 换行，否则字节级不可复现"
    assert raw.endswith(b"\n")


def test_write_lots_csv_roundtrips(tmp_path, lots):
    path = tmp_path / "lots_clean.csv"
    write_lots_csv(path, lots)
    back = pd.read_csv(path, dtype={"invoice_no": str, "stock_code": str})
    assert tuple(back.columns) == LOTS_COLUMNS
    assert back["value_pence"].tolist() == [1530, 250]
    assert back["invoice_no"].tolist() == ["536365", "536369"]
    assert back["invoice_date"].tolist() == lots["invoice_date"].tolist()


def test_write_lots_csv_is_byte_identical_on_rerun(tmp_path, lots):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_lots_csv(a, lots)
    write_lots_csv(b, lots)
    assert a.read_bytes() == b.read_bytes()


def test_write_lots_csv_creates_parent_dirs(tmp_path, lots):
    path = tmp_path / "nested" / "deeper" / "lots_clean.csv"
    write_lots_csv(path, lots)
    assert path.is_file()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_lots.py -v`
Expected: FAIL，`ImportError: cannot import name 'build_lots'`

- [ ] **Step 3: 写实现**

在 `experiment/data/collect/clean.py` 的 `apply_cleaning` 之后追加：
```python
def build_lots(cleaned: pd.DataFrame) -> pd.DataFrame:
    """把清洗结果整理成最终输出表。

    只保留 EXPERIMENT_PLAN.md 第 3.1 节点名的这些列：客户标识（CustomerID）
    与描述文本（Description/Country）刻意排除，它们既不是背包的物品属性，
    也会引入与优化目标无关的标识信息。
    """
    return cleaned.loc[:, list(LOTS_COLUMNS)].reset_index(drop=True)


def write_lots_csv(path, lots: pd.DataFrame) -> str:
    """写出批次表，返回内容的 SHA-256。

    显式指定 LF 与 UTF-8：默认的 ``os.linesep`` 在 Windows 上会产生 CRLF，
    使同一份数据在两台机器上哈希不同。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = lots.to_csv(index=False, lineterminator="\n", encoding="utf-8")
    path.write_bytes(payload.encode("utf-8"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

并把顶部 import 补为：
```python
import hashlib
from pathlib import Path

import pandas as pd

from experiment.data.spec import (
    LOTS_COLUMNS,
    REQUIRED_COLUMNS,
    SOURCE_COLUMNS,
    STOCK_CODE_RE,
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_lots.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/collect/clean.py tests/test_lots.py
git commit -m "feat(data): 批次表结构与 CSV 落盘"
```

---

## Task 5: 下载、解压与字节校验

**Files:**
- Create: `experiment/data/collect/download.py`
- Create: `tests/test_collect_io.py`

- [ ] **Step 1: 写失败测试**

`tests/test_collect_io.py`：
```python
"""下载与解压的离线测试。

网络部分不在这里测：CI 与离线环境都必须能跑绿。真实下载由
tests/test_collect_integration.py 负责，那个文件在缺文件时自动跳过。
"""

import zipfile

import pytest

from experiment.data.collect.download import (
    extract_entry,
    sha256_file,
    verify_workbook_size,
)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello knapsack")
    assert sha256_file(path) == hashlib.sha256(b"hello knapsack").hexdigest()


def test_extract_entry_writes_the_named_member(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Online Retail.xlsx", b"xlsx-bytes")
        zf.writestr("readme.txt", b"ignore me")

    dest = tmp_path / "out.xlsx"
    extract_entry(archive, "Online Retail.xlsx", dest)
    assert dest.read_bytes() == b"xlsx-bytes"


def test_extract_entry_missing_member_raises(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("other.xlsx", b"nope")
    with pytest.raises(KeyError):
        extract_entry(archive, "Online Retail.xlsx", tmp_path / "out.xlsx")


def test_verify_workbook_size_accepts_a_matching_file(tmp_path):
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    assert verify_workbook_size(path, expected_bytes=5) == 5


def test_verify_workbook_size_rejects_a_changed_file(tmp_path):
    """UCI 若替换了源文件，必须报错停下，而不是静默换掉整个实验的数据基础。"""
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="字节数"):
        verify_workbook_size(path, expected_bytes=6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_collect_io.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.collect.download'`

- [ ] **Step 3: 写实现**

`experiment/data/collect/download.py`：
```python
"""UCI Online Retail 的下载、解压与完整性校验。

只用标准库 ``urllib.request``：为一个 23 MB 的一次性下载引入 requests 不划算，
而且少一个依赖就少一处版本漂移。
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from experiment.data.spec import (
    UCI_URL,
    UCI_XLSX_BYTES,
    UCI_XLSX_NAME,
    UCI_ZIP_ENTRY,
    UCI_ZIP_NAME,
    USER_AGENT,
)

#: 单次下载的超时（秒）
_TIMEOUT_SECONDS = 180


def sha256_file(path: Path) -> str:
    """文件的 SHA-256 十六进制摘要，分块读取以免整文件进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_zip(url: str = UCI_URL, dest: Path | None = None) -> Path:
    """下载 zip 到 ``dest`` 并返回该路径。

    部分镜像会拒绝没有 User-Agent 的请求，故显式带上。
    """
    dest = Path(dest) if dest is not None else Path(UCI_ZIP_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return dest


def extract_entry(archive: Path, entry: str, dest: Path) -> Path:
    """把 zip 中的 ``entry`` 解到 ``dest``。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        with zf.open(entry) as source, open(dest, "wb") as target:
            shutil.copyfileobj(source, target)
    return dest


def verify_workbook_size(path: Path, expected_bytes: int = UCI_XLSX_BYTES) -> int:
    """校验工作簿字节数，不符即抛错。

    UCI 是外部来源：文件一旦被替换，清洗结果、实例构造、全部指标都会跟着变，
    而实验的可复现性依赖的正是这份固定的输入。宁可在这里硬失败。
    """
    actual = Path(path).stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"工作簿字节数不符：期望 {expected_bytes}，实际 {actual}（{path}）。"
            "上游文件可能已变更，请先核对再继续。"
        )
    return actual


def ensure_workbook(
    dest_dir: Path,
    force: bool = False,
    url: str = UCI_URL,
) -> tuple[Path, Path]:
    """确保 ``dest_dir`` 下有可用的 zip 与 xlsx，返回 ``(zip_path, xlsx_path)``。

    已存在且字节数正确的 xlsx 会被直接复用，避免每次重跑都下载 23 MB。
    """
    dest_dir = Path(dest_dir)
    zip_path = dest_dir / UCI_ZIP_NAME
    xlsx_path = dest_dir / UCI_XLSX_NAME

    if not force and xlsx_path.is_file():
        verify_workbook_size(xlsx_path)
        return zip_path, xlsx_path

    fetch_zip(url=url, dest=zip_path)
    extract_entry(zip_path, UCI_ZIP_ENTRY, xlsx_path)
    verify_workbook_size(xlsx_path)
    return zip_path, xlsx_path
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_collect_io.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/collect/download.py tests/test_collect_io.py
git commit -m "feat(data): UCI 下载、解压与字节数校验"
```

---

## Task 6: 端到端编排与审计报告

**Files:**
- Create: `experiment/data/collect/run.py`
- Create: `tests/test_collect_integration.py`

- [ ] **Step 1: 写失败测试**

`tests/test_collect_integration.py`：
```python
"""真实工作簿的端到端校验。

未下载工作簿时整体跳过，因此不联网也能跑绿。先执行
``python -m experiment.data.collect.cli`` 再跑本文件即可得到真实校验。
"""

import json

import pytest

from experiment.data.collect.run import read_workbook, run_collection
from experiment.data.spec import (
    CLEANING_RULES,
    LOTS_COLUMNS,
    REAL_DIR,
    SOURCE_COLUMNS,
    UCI_XLSX_BYTES,
    UCI_XLSX_NAME,
)

WORKBOOK = REAL_DIR / UCI_XLSX_NAME

pytestmark = pytest.mark.skipif(
    not WORKBOOK.is_file(), reason="真实工作簿尚未下载，先运行 CLI"
)


def test_read_workbook_has_the_source_columns():
    frame = read_workbook(WORKBOOK)
    assert tuple(frame.columns) == SOURCE_COLUMNS
    assert len(frame) > 500_000


def test_run_collection_writes_the_outputs_of_a_supplied_workbook(tmp_path):
    """传了 workbook 就不下载，因此只落盘清洗产物与审计报告。

    四个文件齐全的形态属于下载路径：那时 ensure_workbook 会把 zip 与 xlsx
    一并写进同一个 out_dir（Task 8 用 CLI 验收这一点）。本文件刻意走离线
    路径，所以不该要求那两个下载件——build_report 里 zip_sha256 允许为
    None，正是为这条路径留的。
    """
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    for name in ("lots_clean.csv", "collect_report.json"):
        assert (tmp_path / name).is_file(), name
    assert report["kept_rows"] > 0


def test_report_identity_holds(tmp_path):
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    dropped = sum(report["cleaning"]["dropped_by_reason"].values())
    assert report["raw_rows"] == report["kept_rows"] + dropped


def test_report_rule_keys_are_complete(tmp_path):
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    assert tuple(report["cleaning"]["dropped_by_reason"]) == CLEANING_RULES


def test_report_records_source_provenance(tmp_path):
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    assert report["source"]["url"].startswith("https://archive.ics.uci.edu/")
    assert report["source"]["xlsx_bytes"] == UCI_XLSX_BYTES
    assert len(report["source"]["xlsx_sha256"]) == 64
    assert len(report["source"]["lots_sha256"]) == 64


def test_kept_lots_survive_the_zero_pence_rule(tmp_path):
    """幸存批次的价值必须全部为正——否则规则 9 没生效。"""
    import pandas as pd
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv")
    assert (lots["value_pence"] > 0).all()
    assert (lots["quantity_w"] > 0).all()


def test_kept_lots_have_valid_stock_codes(tmp_path):
    import pandas as pd
    from experiment.data.spec import STOCK_CODE_RE
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv", dtype={"stock_code": str})
    assert lots["stock_code"].str.match(STOCK_CODE_RE).all()


def test_kept_lots_have_no_cancellation_invoices(tmp_path):
    import pandas as pd
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv", dtype={"invoice_no": str})
    assert not lots["invoice_no"].str.upper().str.startswith("C").any()


def test_kept_lots_columns_are_the_frozen_set(tmp_path):
    import pandas as pd
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    header = (tmp_path / "lots_clean.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(LOTS_COLUMNS)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_collect_integration.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'experiment.data.collect.run'`），或 9 skipped（若工作簿尚未下载——此时先把实现写完，Task 7 下载后会真正执行）

- [ ] **Step 3: 写实现**

`experiment/data/collect/run.py`：
```python
"""采集端到端编排：下载 -> 读取 -> 清洗 -> 落盘 -> 写报告。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiment.data.collect.clean import apply_cleaning, build_lots, write_lots_csv
from experiment.data.collect.download import ensure_workbook, sha256_file
from experiment.data.spec import (
    CLEANING_RULES,
    LOTS_COLUMNS,
    REAL_DIR,
    SCHEMA_VERSION,
    SOURCE_COLUMNS,
    UCI_SHEET_NAME,
    UCI_URL,
    UCI_XLSX_BYTES,
    UCI_XLSX_NAME,
    UCI_ZIP_NAME,
)


def read_workbook(path: Path) -> pd.DataFrame:
    """读取工作簿，并校验列集合与规格一致。

    列名或列序变了就说明上游改了结构，此时继续往下算只会产出看似正常的错数据。
    """
    frame = pd.read_excel(path, sheet_name=UCI_SHEET_NAME, engine="openpyxl")
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"工作簿列与规格不符：期望 {SOURCE_COLUMNS}，实际 {tuple(frame.columns)}"
        )
    return frame


def build_report(
    raw_rows: int,
    cleaning: dict,
    zip_path: Path,
    xlsx_path: Path,
    lots_path: Path,
) -> dict:
    """汇总审计报告。每一项都可在事后独立复核。"""
    dropped = cleaning["dropped_by_reason"]
    return {
        "cleaning": {
            "dropped_by_reason": dropped,
            "identity_holds": raw_rows
            == cleaning["kept_rows"] + sum(dropped.values()),
            "rules": list(CLEANING_RULES),
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kept_rows": cleaning["kept_rows"],
        "lots_columns": list(LOTS_COLUMNS),
        "raw_rows": raw_rows,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "url": UCI_URL,
            "xlsx_bytes": UCI_XLSX_BYTES,
            "xlsx_sha256": sha256_file(xlsx_path),
            "zip_sha256": sha256_file(zip_path) if zip_path.is_file() else None,
            "lots_sha256": sha256_file(lots_path),
        },
    }


def run_collection(
    out_dir: Path = REAL_DIR,
    workbook: Path | None = None,
    force_download: bool = False,
) -> dict:
    """执行完整采集流程，返回审计报告。

    Args:
        out_dir: 输出目录，默认 ``datasets/raw/real``
        workbook: 已有的工作簿路径；为 ``None`` 时才走下载
        force_download: 忽略已存在的工作簿，强制重新下载
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workbook is None:
        zip_path, xlsx_path = ensure_workbook(out_dir, force=force_download)
    else:
        xlsx_path = Path(workbook)
        zip_path = out_dir / UCI_ZIP_NAME

    raw = read_workbook(xlsx_path)
    cleaned, cleaning = apply_cleaning(raw)
    lots = build_lots(cleaned)

    lots_path = out_dir / "lots_clean.csv"
    write_lots_csv(lots_path, lots)

    report = build_report(len(raw), cleaning, zip_path, xlsx_path, lots_path)
    (out_dir / "collect_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
```

注意：`run_collection(workbook=...)` 这条路径不下载，测试因此可以在已有工作簿上反复跑而不联网。

- [ ] **Step 4: 运行测试**

Run: `venv\Scripts\python.exe -m pytest tests/test_collect_integration.py -v`
Expected: 9 skipped（工作簿尚未下载；Task 7 下载后重跑本文件应全绿）

- [ ] **Step 5: 全套测试确认无回归**

Run: `venv\Scripts\python.exe -m pytest -v`
Expected: 全部 PASS 或 skipped，无 FAIL

- [ ] **Step 6: Commit**

```bash
git add experiment/data/collect/run.py tests/test_collect_integration.py
git commit -m "feat(data): 采集端到端编排与审计报告"
```

---

## Task 7: 命令行入口与真实数据执行

**Files:**
- Create: `experiment/data/collect/cli.py`

- [ ] **Step 1: 写实现**

`experiment/data/collect/cli.py`：
```python
"""UCI Online Retail 采集入口。

用法::

    python -m experiment.data.collect.cli
    python -m experiment.data.collect.cli --force-download
    python -m experiment.data.collect.cli --out datasets/raw/real
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.data.collect.run import run_collection
from experiment.data.spec import REAL_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="采集并清洗 UCI Online Retail")
    parser.add_argument(
        "--out",
        type=Path,
        default=REAL_DIR,
        help="输出根目录（默认 datasets/raw/real）",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="已有的工作簿路径；给定则不下载",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="忽略已存在的工作簿，强制重新下载",
    )
    args = parser.parse_args()

    report = run_collection(
        out_dir=args.out,
        workbook=args.workbook,
        force_download=args.force_download,
    )

    print(f"输出目录: {args.out}")
    print(f"原始行数: {report['raw_rows']}")
    print(f"幸存行数: {report['kept_rows']}")
    print("逐规则丢弃:")
    for rule, count in report["cleaning"]["dropped_by_reason"].items():
        print(f"  {rule}: {count}")
    print(f"恒等式成立: {report['cleaning']['identity_holds']}")
    print(f"工作簿 sha256: {report['source']['xlsx_sha256']}")
    print(f"批次表 sha256: {report['source']['lots_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 验证 --help 可用**

Run: `venv\Scripts\python.exe -m experiment.data.collect.cli --help`
Expected: 打印用法说明，退出码 0

- [ ] **Step 3: 执行真实采集**

Run: `venv\Scripts\python.exe -m experiment.data.collect.cli`
Expected: 打印原始行数、幸存行数与逐规则丢弃；`恒等式成立: True`。原始行数应为 541909。

- [ ] **Step 4: 核对产物**

Run:
```powershell
Get-ChildItem datasets\raw\real | Select-Object Name, Length
```
Expected: 四个文件——`online_retail.zip`、`online_retail.xlsx`（23715344 字节）、`lots_clean.csv`、`collect_report.json`。

- [ ] **Step 5: 确认 datasets/ 未被 git 跟踪**

Run: `git status --short`
Expected: **不出现** `datasets/`。

- [ ] **Step 6: Commit**

```bash
git add experiment/data/collect/cli.py
git commit -m "feat(data): 采集命令行入口"
```

---

## Task 8: 真实验收

**Files:**
- 无新增文件；本任务是对已交付实现的验收

- [ ] **Step 1: 跑真实集成测试**

Run: `venv\Scripts\python.exe -m pytest tests/test_collect_integration.py -v`
Expected: 9 passed（不再 skip）

- [ ] **Step 2: 逐规则丢弃数是否合理**

Run:
```powershell
$r = Get-Content datasets\raw\real\collect_report.json | ConvertFrom-Json
"原始 $($r.raw_rows) -> 幸存 $($r.kept_rows)"
$r.cleaning.dropped_by_reason.PSObject.Properties |
  ForEach-Object { "  {0}: {1}" -f $_.Name, $_.Value }
"恒等式: $($r.cleaning.identity_holds)"
```
Expected: 恒等式为 True。逐规则数量应满足常识——`cancellation_invoice` 在 8000～10000 量级（原数据约有 9288 条取消发票），`nonpositive_quantity_or_price` 在数千量级，`exact_duplicate_record` 在数千量级，`zero_value_pence` 在数十量级。若某一项为 0 或异常巨大，停下来排查该规则，不要直接接受结果。

- [ ] **Step 3: 幸存数据的形态检查**

Run:
```powershell
$lots = Import-Csv datasets\raw\real\lots_clean.csv
"幸存批次: $($lots.Count)"
"数量范围: $(($lots | Measure-Object quantity_w -Minimum -Maximum).Minimum) ~ $(($lots | Measure-Object quantity_w -Maximum).Maximum)"
"最大单批价值(分): $(($lots | Measure-Object value_pence -Maximum).Maximum)"
"日期范围: $(($lots | Measure-Object invoice_date -Minimum).Minimum) ~ $(($lots | Measure-Object invoice_date -Maximum).Maximum)"
```
Expected: 数量最小值 ≥ 1（正整数，无 0 无负）；最大单批价值应是八位数（全表最大单批为 80995 × 2.08 = 168469.6 英镑，即 16846960 分）——**极端值必须保留，不得缩尾**；日期范围应落在 2010-12-01 至 2011-12-09 之间。

- [ ] **Step 4: 幂等性验收**

Run:
```powershell
$before = (Get-FileHash datasets\raw\real\lots_clean.csv).Hash
venv\Scripts\python.exe -m experiment.data.collect.cli
$after = (Get-FileHash datasets\raw\real\lots_clean.csv).Hash
"before=$before"; "after =$after"; "一致: $($before -eq $after)"
```
Expected: `一致: True`。

- [ ] **Step 5: 全套测试**

Run: `venv\Scripts\python.exe -m pytest -v`
Expected: 全部 PASS，无 skipped、无 FAIL

- [ ] **Step 6: 依赖可从零重建**

Run:
```powershell
python -m venv $env:TEMP\freshvenv
& $env:TEMP\freshvenv\Scripts\python.exe -m pip install -r requirements.txt
& $env:TEMP\freshvenv\Scripts\python.exe -m pytest -q
```
Expected: 依赖安装成功（无版本冲突警告），测试全绿。这条验收的是"换一台机器能否精准复现"。

- [ ] **Step 7: 记录最终产物哈希**

Run:
```powershell
Get-Content datasets\raw\real\collect_report.json | ConvertFrom-Json |
  Select-Object -ExpandProperty source | Format-List
```
把打印出的 `xlsx_sha256` 与 `lots_sha256` 抄进实验记录（`results/` 或论文附录）。`datasets/` 不进版本库，这两个哈希是事后唯一能证明输入未被调换的凭据。

---

## 完成标准

对照设计文档第 11 节中属于采集部分的条目：

6. 清洗恒等式成立（Task 8 Step 2）
7. `requirements.txt` 能在干净 venv 中重建（Task 8 Step 6）

补充验收（本计划新增，源自 EXPERIMENT_PLAN.md 第 3.1 节的可审计性要求）：

- 八条规则逐条可见其丢弃数，且总量恒等（Task 8 Step 2）
- 极端值被保留而非缩尾（Task 8 Step 3）
- 重复执行字节级一致（Task 8 Step 4）

---

## 与设计文档的两处实现细节说明

1. **规则 8 的比较范围**：设计文档写"rule 8 = exact full-record dupes, keep smallest `source_row_id`"，未言明比较范围。本计划按**顺序流水线**语义实现——规则 8 只在规则 2–7 的幸存者之间比较。理由：设计文档第 9 节称这些是"9 ordered rules"，而有序清洗流水线的标准含义就是每条规则作用于上一条的输出。设计文档未言明比较范围因此无害：**两种读法在本规则集下必然等价**——规则 2–7 都是 `SOURCE_COLUMNS` 之内各列的逐行函数，所以逐字段相同的两行必然同生共死，一个幸存行不可能与一个已被丢弃的行互为重复，规则 8 里的 `alive = reason == ""` 掩码因此是恒等操作。已实测：把规则 8 换成对整个原始表做 `duplicated`，全套测试依旧全绿，且在 800 个随机表上两种写法的归因报告逐字节相同。按流水线语义书写仍然更好——它表达了流水线本意，且若某天有规则改用 `SOURCE_COLUMNS` 之外的列，这个选择会立刻变得可观测。另外，`test_duplicate_is_judged_among_survivors_only` 并没有钉死这个选择（它做不到）：该用例构造的两行都是取消发票且数量为负，规则 2 已把它们一并丢弃，所以它的三条断言与规则 8 的行为无关；用例名与 docstring 声称的性质无法被它检验。

2. **`collect/cli.py` 为新增文件**：设计文档第 3 节的模块布局只列了 `collect/download.py` 与 `collect/clean.py`。为对齐 `generate/cli.py` 并满足"两个入口点相互独立"，本计划增加 `collect/cli.py` 与 `collect/run.py`（编排层）。这是布局的补充，不与设计冲突。
