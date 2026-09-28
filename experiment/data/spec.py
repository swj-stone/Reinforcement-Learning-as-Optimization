"""实验规格的唯一真源。

本模块的常量一经确定即视为冻结：改动其中任何一项都会改变已生成语料的
内容与哈希，等同于使既有产物失效。

依据 EXPERIMENT_PLAN.md 第 2 节。
"""

from __future__ import annotations

import math
from pathlib import Path

# --- 冻结的生成维度 ---------------------------------------------------------

FAMILIES: tuple[str, ...] = ("uniform", "normal", "correlated", "skewed")
SIZES: tuple[int, ...] = (20, 50, 100, 200)
ALPHAS: tuple[float, ...] = (0.25, 0.50, 0.75)
SPLITS: tuple[str, ...] = ("train", "validation", "test")

MASTER_SEEDS: dict[str, int] = {"train": 1001, "validation": 2001, "test": 3001}

#: 每个 (family, n, alpha) cell 的实例数
SYNTHETIC_QUOTA: dict[str, int] = {"train": 125, "validation": 25, "test": 25}

# --- 生成语义常量 -----------------------------------------------------------

#: 偏斜族中稀有索引的比例（EXPERIMENT_PLAN.md 第 2 节）
SKEW_FRACTION: float = 0.05
#: 偏斜族稀有项的权重范围，含端点
SKEW_HEAVY_WEIGHT_RANGE: tuple[int, int] = (200, 500)
#: 偏斜族稀有项的密度范围，含端点
SKEW_HEAVY_DENSITY_RANGE: tuple[int, int] = (30, 60)
#: 常规项的权重与密度范围，含端点
SKEW_BASE_WEIGHT_RANGE: tuple[int, int] = (1, 20)
SKEW_BASE_DENSITY_RANGE: tuple[int, int] = (1, 5)

#: 跨 split 哈希冲突时的重抽上限，超过即报错而非死循环
MAX_RESAMPLE_ATTEMPT: int = 1000

#: manifest 的模式版本
SCHEMA_VERSION: int = 1

# --- UCI Online Retail 采集 --------------------------------------------------

UCI_URL: str = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_ZIP_ENTRY: str = "Online Retail.xlsx"
UCI_SHEET_NAME: str = "Online Retail"
#: 冻结的工作簿字节数。UCI 若替换了文件，此校验会立刻失败，而不是静默换数据。
UCI_XLSX_BYTES: int = 23715344
#: 冻结的工作簿 SHA-256。字节数只是粗筛：上游完全可以换一个同样 23715344 字节的
#: 文件，而清洗结果、批次表乃至全部下游指标都取决于这份输入的精确字节。摘要校验
#: 才真正把「换了文件」这件事变得可观测。
UCI_XLSX_SHA256: str = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
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
# --- 输出路径 ---------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS_RAW: Path = REPO_ROOT / "datasets" / "raw"
GENERATION_DIR: Path = DATASETS_RAW / "generation"
REAL_DIR: Path = DATASETS_RAW / "real"


# --- 派生辅助 ---------------------------------------------------------------


def alpha_key(alpha: float) -> str:
    """容量比的稳定字符串键，用于种子派生。

    显式格式化为两位小数，使「本应相同、却因浮点运算而近似」的两个值落到同一个
    键上：``0.1 + 0.2`` 的 ``repr`` 是 ``0.30000000000000004``，直接 ``str()``
    会与 ``0.3`` 派生出两个不同的种子，``f"{...:.2f}"`` 则都得到 ``0.30``。
    """
    return f"{alpha:.2f}"


def alpha_token(alpha: float) -> str:
    """容量比的文件名记号：0.25 -> ``a25``。

    用整数百分比而非 ``a0.25``，避免文件名中的点被工具按扩展名切分。
    """
    return f"a{round(alpha * 100):02d}"


def skew_k(n: int) -> int:
    """偏斜族的稀有索引个数：``max(1, floor(0.05n + 0.5))``。"""
    return max(1, math.floor(SKEW_FRACTION * n + 0.5))


def cell_filename(family: str, n: int, alpha: float) -> str:
    """cell 文件名，如 ``uniform_n20_a25.jsonl``。"""
    return f"{family}_n{n}_{alpha_token(alpha)}.jsonl"


def all_cells() -> list[tuple[str, int, float]]:
    """按冻结顺序枚举全部 48 个 cell。

    顺序即去重时的判定顺序，是冻结语义的一部分：family 在外层、n 居中、alpha
    在内层（即下方列表推导的嵌套序）。改动这个嵌套序会改变哪些实例被重抽。

    注意本函数不含 split 维度：先 train、后 validation、最后 test 的遍历顺序由
    ``generate`` 模块的去重循环负责，不在返回值里。
    """
    return [(f, n, a) for f in FAMILIES for n in SIZES for a in ALPHAS]


# --- 实例装配（第三阶段，依据 EXPERIMENT_PLAN.md 第 4 节与第 8 节）---------

#: 真实实例每个 (n, alpha) cell 的配额（第 4 节）
REAL_QUOTA: dict[str, int] = {"train": 500, "validation": 100, "test": 100}

#: 日期切分的两个边界占比（第 4 节），均下取整
SPLIT_DATE_FRACTIONS: tuple[float, float] = (0.70, 0.85)

#: 真实实例在种子键中占用的 family 槽位记号。刻意不在 FAMILIES 中：
#: 真实实例的种子因此不可能与任何合成族相同。
REAL_SEED_TOKEN: str = "real"

#: 应力集的流记号，只活在种子键内部，不落盘。落盘用的是 E3_NAME / E4_NAME。
#: 第 8 节要求「保留 4001/5001 两条流」；真正承担隔离的是这两个字符串记号，
#: 而不是 STRESS_MASTER_SEEDS 里的数字（见设计文档第 12.1 节）。
STRESS_SIZE_STREAM: str = "stream_size"
STRESS_TAIL_STREAM: str = "stream_tail"

#: E4 变体生成器在种子键中占用的 family 槽位记号
E4_SEED_TOKEN: str = "skewed_e4"

#: 第 8 节声明的应力流主种子。与 MASTER_SEEDS 同样是**声明标识**，不参与
#: derive_seed 的键；记录在此是为了让声明与机制的关系可被复核。
STRESS_MASTER_SEEDS: dict[str, int] = {"size": 4001, "tail": 5001}

#: 应力集的数据集名，同时是目录名与 instance_id 的组成部分
E3_NAME: str = "e3_size"
E4_NAME: str = "e4_tail"

#: E3（未见过的规模）配额，第 8 节
E3_SIZE: int = 500
E3_SYNTHETIC_PER_FAMILY: int = 25
E3_REAL_PER_ALPHA: int = 100

#: E4（更强的不平衡）配额与变体参数，第 8 节
E4_SIZE: int = 200
E4_COUNT_PER_ALPHA: int = 100
E4_SKEW_K: int = 2
E4_SKEW_HEAVY_WEIGHT_RANGE: tuple[int, int] = (500, 2000)
E4_SKEW_HEAVY_DENSITY_RANGE: tuple[int, int] = (100, 200)

#: E4b 阈值分位数（第 8 节）。阈值只用训练记录定；本阶段只算并冻结，
#: 实际标记属于评估阶段。
TAIL_QUANTILE: float = 0.99

# --- 装配阶段的输出路径 -----------------------------------------------------

DATASETS_PROCESSED: Path = REPO_ROOT / "datasets" / "processed"

#: 协议产物放在仓库根目录并纳入版本库。datasets/ 整体被 gitignore，把审计
#: 凭据放在那里等于换一台机器就无从证明数据未被调换。
DATA_MANIFEST_PATH: Path = REPO_ROOT / "data_manifest.json"
EXPERIMENT_CONFIG_PATH: Path = REPO_ROOT / "experiment_config.yaml"


def capacity_cells() -> list[tuple[int, float]]:
    """按冻结顺序枚举 12 个 ``(n, alpha)`` cell：n 在外层、alpha 在内层。

    这是合并文件里实例的排列顺序，也是 50:50 断言的遍历顺序。刻意与
    :func:`all_cells` 的 family-外层嵌套序不同：那个服务生成的去重顺序，
    本函数服务面向读者的产物顺序。
    """
    return [(n, a) for n in SIZES for a in ALPHAS]
