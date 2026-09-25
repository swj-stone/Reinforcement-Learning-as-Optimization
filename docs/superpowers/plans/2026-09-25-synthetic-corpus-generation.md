# 合成语料生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `EXPERIMENT_PLAN.md` 第 2 节的四个族生成 8400 个 0/1 背包实例，按 cell 分片落到 `datasets/raw/generation/`，并产出可校验的 `manifest.json`。

**Architecture:** 内容寻址的种子派生（`blake2b`）保证实例与调用顺序无关；四个族是纯抽样函数，不碰 I/O；`corpus.py` 负责编排、跨 split 哈希去重与确定性重抽、落盘。TDD：每个单元先写测试再实现。

**Tech Stack:** Python 3.12、NumPy、pytest

**设计依据:** [../specs/2026-09-25-raw-data-collection-and-generation-design.md](../specs/2026-09-25-raw-data-collection-and-generation-design.md)

**工作目录:** 所有相对路径均以仓库根目录为基准。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `requirements.txt` | 精确固定的依赖版本 |
| `pyproject.toml` | pytest 配置（`pythonpath = ["."]`） |
| `experiment/__init__.py` | 包标记 |
| `experiment/data/__init__.py` | 包标记 |
| `experiment/data/spec.py` | **唯一真源**：族/规模/容量比/配额/主种子/输出路径/格式常量 |
| `experiment/data/generate/__init__.py` | 包标记 |
| `experiment/data/generate/seeds.py` | `derive_seed()`：内容寻址种子派生 |
| `experiment/data/generate/families.py` | `round_half_up()`、四个族、`capacity_for()`、`compute_stats()` |
| `experiment/data/generate/hashing.py` | `canonical_hash()`：排序后哈希，识别物品重排副本 |
| `experiment/data/generate/corpus.py` | cell 编排、配额、跨 split 去重、确定性重抽、落盘、manifest |
| `experiment/data/generate/cli.py` | 命令行入口 |
| `tests/__init__.py` | 包标记 |
| `tests/test_spec.py` | 冻结常量自洽性 |
| `tests/test_seeds.py` | 种子派生的确定性与顺序无关性 |
| `tests/test_families.py` | 四族抽样公式与边界 |
| `tests/test_hashing.py` | 哈希的重排不变性与区分度 |
| `tests/test_corpus.py` | 去重、重抽、落盘、manifest、幂等性 |
| `tests/test_golden.py` | 金样本：钉死已冻结语义 |

---

## Task 1: 依赖固定与 pytest 配置

**Files:**
- Modify: `requirements.txt`
- Create: `pyproject.toml`

- [ ] **Step 1: 安装依赖到项目 venv**

Run:
```powershell
$py = "venv\Scripts\python.exe"
& $py -m pip install numpy pandas openpyxl pytest
```
Expected: `Successfully installed ...`，无错误。

- [ ] **Step 2: 取出实际安装的版本**

Run:
```powershell
& $py -m pip show numpy pandas openpyxl pytest | Select-String '^(Name|Version):'
```
Expected: 打印四个包名与各自的版本号。**把实际打印出的版本记下来**，下一步要写进 `requirements.txt`。不要凭记忆填写版本号。

- [ ] **Step 3: 写 requirements.txt**

把上一步实际得到的版本填入（下例中的版本号必须替换为 Step 2 的真实输出）：

```
# 运行时依赖
numpy==<Step 2 输出>
pandas==<Step 2 输出>
openpyxl==<Step 2 输出>

# 开发/测试依赖
pytest==<Step 2 输出>
```

- [ ] **Step 4: 写 pyproject.toml**

```toml
[project]
name = "rl-knapsack-experiment"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 5: 验证配置生效**

Run:
```powershell
& $py -m pytest --collect-only -q
```
Expected: 输出 `no tests ran`（此时还没有测试文件），**且不报 `pyproject.toml` 解析错误**。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore: 固定依赖版本并配置 pytest"
```

---

## Task 2: 包骨架与冻结常量

**Files:**
- Create: `experiment/__init__.py`, `experiment/data/__init__.py`, `experiment/data/generate/__init__.py`, `tests/__init__.py`
- Create: `experiment/data/spec.py`
- Test: `tests/test_spec.py`

- [ ] **Step 1: 建包标记文件**

四个 `__init__.py` 均为**空文件**。逐项判断存在性，避免 `-Force` 截断已有内容：
```powershell
foreach ($p in 'experiment\__init__.py','experiment\data\__init__.py','experiment\data\generate\__init__.py','tests\__init__.py') {
  if (-not (Test-Path $p)) { New-Item -ItemType File $p | Out-Null; "created $p" }
}
```
Expected: 打印四行 `created ...`。

- [ ] **Step 2: 写失败测试**

`tests/test_spec.py`：
```python
import math
from pathlib import Path

from experiment.data import spec


def test_families_are_the_four_frozen_ones():
    assert spec.FAMILIES == ("uniform", "normal", "correlated", "skewed")


def test_sizes_and_alphas():
    assert spec.SIZES == (20, 50, 100, 200)
    assert spec.ALPHAS == (0.25, 0.50, 0.75)


def test_master_seeds():
    assert spec.MASTER_SEEDS == {"train": 1001, "validation": 2001, "test": 3001}


def test_cell_count_is_48():
    assert len(spec.FAMILIES) * len(spec.SIZES) * len(spec.ALPHAS) == 48


def test_quota_math_matches_plan():
    assert spec.SYNTHETIC_QUOTA == {"train": 125, "validation": 25, "test": 25}
    assert 48 * spec.SYNTHETIC_QUOTA["train"] == 6000
    assert 48 * spec.SYNTHETIC_QUOTA["validation"] == 1200
    assert 48 * spec.SYNTHETIC_QUOTA["test"] == 1200


def test_alpha_key_is_stable_and_two_decimals():
    assert spec.alpha_key(0.25) == "0.25"
    assert spec.alpha_key(0.50) == "0.50"
    assert spec.alpha_key(0.75) == "0.75"


def test_alpha_token_matches_filename_convention():
    assert spec.alpha_token(0.25) == "a25"
    assert spec.alpha_token(0.50) == "a50"
    assert spec.alpha_token(0.75) == "a75"


def test_output_dirs_are_under_datasets_raw():
    assert spec.GENERATION_DIR.parent == spec.DATASETS_RAW
    assert spec.DATASETS_RAW.name == "raw"
    assert spec.DATASETS_RAW.parent.name == "datasets"


def test_skew_k_per_size():
    assert [spec.skew_k(n) for n in spec.SIZES] == [1, 3, 5, 10]


def test_skew_k_is_floor_of_half_offset():
    for n in (20, 50, 100, 200):
        assert spec.skew_k(n) == max(1, math.floor(0.05 * n + 0.5))
```

- [ ] **Step 3: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_spec.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.spec'`

- [ ] **Step 4: 写实现**

`experiment/data/spec.py`：
```python
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

# --- 输出路径 ---------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS_RAW: Path = REPO_ROOT / "datasets" / "raw"
GENERATION_DIR: Path = DATASETS_RAW / "generation"
REAL_DIR: Path = DATASETS_RAW / "real"


# --- 派生辅助 ---------------------------------------------------------------


def alpha_key(alpha: float) -> str:
    """容量比的稳定字符串键，用于种子派生。

    显式格式化为两位小数，避免 ``str(0.5) == "0.5"`` 与 ``str(0.50) == "0.5"``
    之类由浮点表示差异引起的键不一致。
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
    """按冻结顺序枚举所有 cell。

    顺序即去重时的判定顺序，是冻结语义的一部分：先 train、后 validation、
    最后 test。改动顺序会改变哪些实例被重抽。
    """
    return [(f, n, a) for f in FAMILIES for n in SIZES for a in ALPHAS]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_spec.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add experiment tests
git commit -m "feat(data): 包骨架与冻结的实验规格常量"
```

---

## Task 3: 内容寻址的种子派生

**Files:**
- Create: `experiment/data/generate/seeds.py`
- Test: `tests/test_seeds.py`

- [ ] **Step 1: 写失败测试**

`tests/test_seeds.py`：
```python
import pytest

from experiment.data.generate.seeds import STREAMS, derive_seed

BASE = dict(split="train", family="uniform", n=20, alpha=0.25, index=0)


def test_same_inputs_give_same_seed():
    assert derive_seed(**BASE) == derive_seed(**BASE)


def test_seed_is_uint64():
    s = derive_seed(**BASE)
    assert isinstance(s, int)
    assert 0 <= s < 2**64


@pytest.mark.parametrize(
    "field, value",
    [
        ("split", "test"),
        ("family", "skewed"),
        ("n", 50),
        ("alpha", 0.50),
        ("index", 1),
    ],
)
def test_each_field_changes_the_seed(field, value):
    assert derive_seed(**{**BASE, field: value}) != derive_seed(**BASE)


def test_attempt_changes_the_seed():
    assert derive_seed(**BASE, attempt=1) != derive_seed(**BASE, attempt=0)


def test_stream_changes_the_seed():
    assert derive_seed(**BASE, stream="items") != derive_seed(**BASE, stream="order")


def test_both_streams_are_defined():
    assert STREAMS == ("items", "order")


def test_unknown_stream_raises():
    with pytest.raises(ValueError, match="unknown stream"):
        derive_seed(**BASE, stream="nope")


def test_alpha_key_normalisation_does_not_change_seed():
    """0.5 与 0.50 是同一个 float，必须派生出同一个种子。"""
    assert derive_seed(**{**BASE, "alpha": 0.5}) == derive_seed(**{**BASE, "alpha": 0.50})


def test_seed_is_independent_of_call_order():
    """先算别的组合再算目标组合，结果必须一致。"""
    first = derive_seed(**BASE)
    for other in (
        dict(split="test", family="skewed", n=200, alpha=0.75, index=99),
        dict(split="validation", family="normal", n=100, alpha=0.50, index=7),
    ):
        derive_seed(**other)
    assert derive_seed(**BASE) == first
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_seeds.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.generate.seeds'`

- [ ] **Step 3: 写实现**

`experiment/data/generate/seeds.py`：
```python
"""内容寻址的种子派生。

设计取舍见设计文档第 5 节：用哈希而非位置式的 ``SeedSequence.spawn()``，
使种子与调用顺序无关，语料冻结后不会因为别处新增维度而整体平移。
"""

from __future__ import annotations

import hashlib

from experiment.data.spec import alpha_key

#: 相互隔离的抽样流。``items`` 抽权重与价值/参数，``order`` 抽物品排列。
STREAMS: tuple[str, ...] = ("items", "order")

_MASK64 = (1 << 64) - 1


def derive_seed(
    split: str,
    family: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    stream: str = "items",
) -> int:
    """由实例的完整身份派生一个 64 位种子。

    Args:
        split: ``train`` / ``validation`` / ``test``
        family: 四个族之一
        n: 物品数
        alpha: 容量比
        index: 该 cell 内的实例序号
        attempt: 重抽次数，跨 split 哈希冲突时递增
        stream: ``items`` 或 ``order``

    Returns:
        ``[0, 2**64)`` 内的整数，可直接交给 ``numpy.random.default_rng``。
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream: {stream!r}; expected one of {STREAMS}")

    key = "|".join(
        (
            split,
            family,
            str(n),
            alpha_key(alpha),
            str(index),
            str(attempt),
            stream,
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK64
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_seeds.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/seeds.py tests/test_seeds.py
git commit -m "feat(data): 内容寻址的种子派生"
```

---

## Task 4: half-up 取整、容量公式与 uniform 族

**Files:**
- Create: `experiment/data/generate/families.py`
- Test: `tests/test_families.py`

- [ ] **Step 1: 写失败测试**

`tests/test_families.py`：
```python
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
    assert round_half_up(value) == expected


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


def test_uniform_weights_and_values_are_independent():
    w, v = generate_items("uniform", N_REPEAT, seed=999)
    corr = np.corrcoef(w.astype(float), v.astype(float))[0, 1]
    assert abs(corr) < 0.1


def test_capacity_is_floor_of_alpha_times_total_weight():
    w, _ = generate_items("uniform", 100, seed=7)
    for alpha in (0.25, 0.50, 0.75):
        assert capacity_for(alpha, w) == math.floor(alpha * int(w.sum()))


def test_capacity_of_alpha_one_equals_total_weight():
    w, _ = generate_items("uniform", 50, seed=8)
    assert capacity_for(1.0, w) == int(w.sum())


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        generate_items("nope", 20, seed=1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.generate.families'`

- [ ] **Step 3: 写实现**

`experiment/data/generate/families.py`：
```python
"""四个合成数据族的纯抽样函数。

本模块只做抽样，不碰文件系统、不含 I/O、不持全局状态。

**抽样顺序是冻结语义的一部分**：每个族内部的 rng 调用次序一经确定即不可
更改，否则会改变已生成语料的全部内容。各族顺序见函数注释。
"""

from __future__ import annotations

import math

import numpy as np

from experiment.data.spec import (
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    skew_k,
)

IntArray = np.ndarray


def round_half_up(x):
    """四舍五入（.5 进位），接受标量或数组。

    刻意不用 ``np.rint``：后者是银行家舍入，把 2.5 舍为 2。设计文档第 6 节
    选定 half-up，与 EXPERIMENT_PLAN.md 第 3.1 节对价格的取整要求一致。

    标量输入返回 ``int``，数组输入返回 ``np.int64`` 数组。
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.floor(arr + 0.5).astype(np.int64)
    return int(out) if arr.ndim == 0 else out


def generate_items(family: str, n: int, seed: int) -> tuple[IntArray, IntArray]:
    """生成一个实例的权重与价值数组。

    Args:
        family: ``uniform`` / ``normal`` / ``correlated`` / ``skewed``
        n: 物品数
        seed: 由 :func:`experiment.data.generate.seeds.derive_seed` 派生

    Returns:
        ``(weights, values)``，均为 ``int64`` 且长度为 ``n``。
    """
    rng = np.random.default_rng(seed)
    return _FAMILY_FUNCS[family](rng, n)


def capacity_for(alpha: float, weights: IntArray) -> int:
    """``C = floor(alpha * sum(w))``。

    三个 alpha 取值都是二进制精确值，浮点乘积对本项目的量级无误差；
    仍显式下取整以匹配规格。
    """
    return math.floor(alpha * int(weights.sum()))


# --- 各族实现（顺序即冻结语义）---------------------------------------------


def _uniform(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, v。"""
    w = rng.integers(1, 101, size=n, dtype=np.int64)
    v = rng.integers(1, 101, size=n, dtype=np.int64)
    return w, v


def _normal(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w 的正态抽样、v 的正态抽样。

    先取整、后裁剪，与 EXPERIMENT_PLAN.md 第 2 节的措辞顺序一致。
    """
    w = np.clip(round_half_up(rng.normal(50.0, 15.0, size=n)), 1, 100).astype(np.int64)
    v = np.clip(round_half_up(rng.normal(50.0, 15.0, size=n)), 1, 100).astype(np.int64)
    return w, v


def _correlated(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, e。``v = max(1, w + e)``。"""
    w = rng.integers(1, 101, size=n, dtype=np.int64)
    e = rng.integers(-10, 11, size=n, dtype=np.int64)
    v = np.maximum(1, w + e).astype(np.int64)
    return w, v


def _skewed(rng: np.random.Generator, n: int) -> tuple[IntArray, IntArray]:
    """顺序：w, d, idx_w, 替换权重, idx_d, 替换密度。

    三处易错点（设计文档第 6 节）：
    1. ``idx_w`` 与 ``idx_d`` 独立抽取，二者**可以重叠**；
    2. ``v = w * d`` 用**替换之后**的 w 与 d；
    3. 极端值一律保留，不做裁剪。
    """
    lo_w, hi_w = SKEW_BASE_WEIGHT_RANGE
    lo_d, hi_d = SKEW_BASE_DENSITY_RANGE
    heavy_lo_w, heavy_hi_w = SKEW_HEAVY_WEIGHT_RANGE
    heavy_lo_d, heavy_hi_d = SKEW_HEAVY_DENSITY_RANGE

    w = rng.integers(lo_w, hi_w + 1, size=n, dtype=np.int64)
    d = rng.integers(lo_d, hi_d + 1, size=n, dtype=np.int64)

    k = skew_k(n)

    idx_w = rng.choice(n, size=k, replace=False)
    w[idx_w] = rng.integers(heavy_lo_w, heavy_hi_w + 1, size=k, dtype=np.int64)

    idx_d = rng.choice(n, size=k, replace=False)
    d[idx_d] = rng.integers(heavy_lo_d, heavy_hi_d + 1, size=k, dtype=np.int64)

    v = (w * d).astype(np.int64)
    return w, v


_FAMILY_FUNCS = {
    "uniform": _uniform,
    "normal": _normal,
    "correlated": _correlated,
    "skewed": _skewed,
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/families.py tests/test_families.py
git commit -m "feat(data): half-up 取整、容量公式与 uniform 族"
```

---

## Task 5: normal 族与 correlated 族的公式锁定

**Files:**
- Modify: `tests/test_families.py`（追加测试，实现已在 Task 4 写完）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_families.py` 末尾追加：
```python
def test_normal_is_clipped_to_1_100():
    w, v = generate_items("normal", N_REPEAT, seed=4242)
    assert w.min() >= 1 and w.max() <= 100
    assert v.min() >= 1 and v.max() <= 100


def test_normal_is_centred_near_50():
    w, _ = generate_items("normal", N_REPEAT, seed=4243)
    assert 48.0 <= w.mean() <= 52.0


def test_normal_reaches_both_clip_bounds():
    """正态(50,15)在 20 万次抽样里必然越过两侧边界，否则裁剪没被触发。"""
    w, _ = generate_items("normal", N_REPEAT_BIG, seed=4244)
    assert w.min() == 1
    assert w.max() == 100


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
    """w=1 时 w+e 可低至 -9，v 必须被抬到 1，且永远不低于 1。"""
    w, v = generate_items("correlated", N_REPEAT, seed=557)
    assert (v >= 1).all()
    assert (v == 1).any()
    # w=1 处 e 取遍 [-10,10]，故 v = max(1, 1+e) 只能落在 [1, 11]
    assert v[w == 1].min() >= 1
    assert v[w == 1].max() <= 11
```

- [ ] **Step 2: 运行测试**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: PASS（实现已在 Task 4 写好；若有 FAIL，说明 Task 4 的实现有误，回到 Task 4 修正）

- [ ] **Step 3: Commit**

```bash
git add tests/test_families.py
git commit -m "test(data): 锁定 normal 与 correlated 族的公式与边界"
```

---

## Task 6: skewed 族的三处易错点

**Files:**
- Modify: `tests/test_families.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_families.py` 末尾追加：
```python
@pytest.mark.parametrize("n, k", [(20, 1), (50, 3), (100, 5), (200, 10)])
def test_skewed_has_exactly_k_heavy_weights(n, k):
    w, _ = generate_items("skewed", n, seed=31337)
    assert int((w >= 200).sum()) == k
    assert int((w <= 20).sum()) == n - k


@pytest.mark.parametrize("n, k", [(20, 1), (50, 3), (100, 5), (200, 10)])
def test_skewed_has_exactly_k_heavy_densities(n, k):
    """d = v / w 可精确还原，因为没有取整损失。"""
    w, v = generate_items("skewed", n, seed=31338)
    d = v // w
    assert (v % w == 0).all()
    assert int((d >= 30).sum()) == k
    assert int((d <= 5).sum()) == n - k


def test_skewed_two_index_sets_are_independent_and_can_overlap():
    """两个集合各自大小恰好为 k，但并集大小应有变化——说明它们独立抽取。"""
    sizes = set()
    for seed in range(60):
        w, v = generate_items("skewed", 100, seed=seed)
        heavy_w = set(np.flatnonzero(w >= 200).tolist())
        heavy_d = set(np.flatnonzero((v // w) >= 30).tolist())
        assert len(heavy_w) == 5 and len(heavy_d) == 5
        sizes.add(len(heavy_w | heavy_d))
    assert min(sizes) < 10, "两集合从不重叠，说明它们不是独立抽取的"
    assert 10 in sizes, "两集合从不分离，说明它们其实被绑在了一起"


def test_skewed_heavy_weight_values_are_in_range():
    w, _ = generate_items("skewed", 200, seed=2222)
    heavy = w[w >= 200]
    assert heavy.min() >= 200 and heavy.max() <= 500


def test_skewed_base_densities_are_in_range():
    w, v = generate_items("skewed", 200, seed=2223)
    d = v // w
    base = d[d <= 5]
    assert base.min() >= 1 and base.max() <= 5


def test_skewed_heavy_densities_are_in_range():
    w, v = generate_items("skewed", 200, seed=2224)
    d = v // w
    heavy = d[d >= 30]
    assert heavy.min() >= 30 and heavy.max() <= 60


def test_skewed_keeps_extreme_observations():
    """极端值必须保留：最大值远高于中位数，且没有任何缩尾痕迹。"""
    w, _ = generate_items("skewed", 200, seed=2225)
    assert w.max() / np.median(w) > 10
    assert int((w >= 200).sum()) == 10
```

- [ ] **Step 2: 运行测试确认失败或通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: 全部 PASS。若有 FAIL，最可能是 Task 4 的 `_skewed` 把两个索引集写成了同一组，回到 Task 4 修正——这是设计文档第 6 节点名的最易错处。

- [ ] **Step 3: Commit**

```bash
git add tests/test_families.py
git commit -m "test(data): 锁定 skewed 族的稀有索引、独立性与极端值保留"
```

---

## Task 7: 偏斜描述统计量

**Files:**
- Modify: `experiment/data/generate/families.py`（追加 `compute_stats`）
- Modify: `tests/test_families.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_families.py` 末尾追加：
```python
from experiment.data.generate.families import compute_stats


def test_stats_keys_are_the_frozen_six():
    w, v = generate_items("uniform", 50, seed=1)
    assert set(compute_stats(w, v)) == {
        "top5_weight_share",
        "top5_value_share",
        "max_over_median_weight",
        "max_over_median_value",
        "cv_weight",
        "cv_value",
    }


def test_stats_all_are_finite_floats():
    w, v = generate_items("skewed", 200, seed=2)
    for name, value in compute_stats(w, v).items():
        assert isinstance(value, float), name
        assert math.isfinite(value), name


@pytest.mark.parametrize("n, k5", [(20, 1), (50, 3), (100, 5), (200, 10)])
def test_top5_share_uses_ceil_of_five_percent(n, k5):
    x = np.arange(1, n + 1, dtype=np.int64)
    stats = compute_stats(x, x)
    top = np.sort(x)[::-1][:k5]
    assert stats["top5_weight_share"] == pytest.approx(float(top.sum()) / float(x.sum()))


def test_top5_share_of_uniform_is_modest():
    """均匀分布下前 5% 的占比约 0.10——不是 0.05。

    5% 的物品落在支撑集最右端，U(1,100) 的这 5% 均值约 98，因此份额约为
    98/50.5 * 0.05 ≈ 0.097。这里给一个宽松的双侧区间，只用来确认量级正确。
    """
    w, _ = generate_items("uniform", N_REPEAT, seed=3)
    stats = compute_stats(w, w)
    assert 0.07 < stats["top5_weight_share"] < 0.14


def test_cv_uses_population_std():
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    assert compute_stats(x, x)["cv_weight"] == pytest.approx(
        float(np.std(x, ddof=0)) / float(np.mean(x))
    )


def test_max_over_median_for_constant_array_is_one():
    x = np.full(10, 7, dtype=np.int64)
    assert compute_stats(x, x)["max_over_median_weight"] == pytest.approx(1.0)


def test_skewed_has_larger_top5_share_than_uniform():
    sw, _ = generate_items("skewed", 200, seed=4)
    uw, _ = generate_items("uniform", 200, seed=4)
    assert (
        compute_stats(sw, sw)["top5_weight_share"]
        > compute_stats(uw, uw)["top5_weight_share"]
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: FAIL，`ImportError: cannot import name 'compute_stats'`

- [ ] **Step 3: 写实现**

在 `experiment/data/generate/families.py` 的 `capacity_for` 之后、`# --- 各族实现` 之前插入：
```python
def compute_stats(weights: IntArray, values: IntArray) -> dict[str, float]:
    """描述实例的分布形态，供 EXPERIMENT_PLAN.md 第 8 节 E2 的分解使用。

    六项定义见设计文档第 8.1 节。``k5 = max(1, ceil(0.05n))``；标准差用
    总体标准差（``ddof=0``）；不做任何舍入。
    """
    return {
        "top5_weight_share": _top_share(weights),
        "top5_value_share": _top_share(values),
        "max_over_median_weight": float(weights.max()) / float(np.median(weights)),
        "max_over_median_value": float(values.max()) / float(np.median(values)),
        "cv_weight": float(weights.std(ddof=0)) / float(weights.mean()),
        "cv_value": float(values.std(ddof=0)) / float(values.mean()),
    }


def _top_share(x: IntArray) -> float:
    k5 = max(1, math.ceil(0.05 * len(x)))
    descending = np.sort(x)[::-1]
    return float(descending[:k5].sum()) / float(x.sum())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_families.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/families.py tests/test_families.py
git commit -m "feat(data): 偏斜描述统计量 compute_stats"
```

---

## Task 8: canonical instance hash

**Files:**
- Create: `experiment/data/generate/hashing.py`
- Test: `tests/test_hashing.py`

- [ ] **Step 1: 写失败测试**

`tests/test_hashing.py`：
```python
import numpy as np

from experiment.data.generate.hashing import canonical_hash

W = np.array([3, 1, 4, 1], dtype=np.int64)
V = np.array([10, 20, 30, 40], dtype=np.int64)


def test_hash_is_hex_sha256():
    h = canonical_hash(4, 5, W, V)
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_is_deterministic():
    assert canonical_hash(4, 5, W, V) == canonical_hash(4, 5, W, V)


def test_reordering_items_does_not_change_the_hash():
    """EXPERIMENT_PLAN.md 第 2 节要求识别 copies with reordered items。"""
    perm = np.array([2, 0, 3, 1])
    assert canonical_hash(4, 5, W[perm], V[perm]) == canonical_hash(4, 5, W, V)


def test_capacity_change_changes_the_hash():
    assert canonical_hash(4, 5, W, V) != canonical_hash(4, 6, W, V)


def test_n_change_changes_the_hash():
    assert canonical_hash(4, 5, W, V) != canonical_hash(5, 5, W, V)


def test_changing_one_weight_changes_the_hash():
    other = W.copy()
    other[0] = 99
    assert canonical_hash(4, 5, other, V) != canonical_hash(4, 5, W, V)


def test_changing_one_value_changes_the_hash():
    other = V.copy()
    other[0] = 99
    assert canonical_hash(4, 5, W, other) != canonical_hash(4, 5, W, V)


def test_duplicate_pairs_are_order_insensitive():
    """出现重复 (w, v) 对时，排序后仍应稳定。"""
    w = np.array([2, 2, 5], dtype=np.int64)
    v = np.array([7, 7, 1], dtype=np.int64)
    perm = np.array([1, 2, 0])
    assert canonical_hash(3, 4, w[perm], v[perm]) == canonical_hash(3, 4, w, v)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_hashing.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.generate.hashing'`

- [ ] **Step 3: 写实现**

`experiment/data/generate/hashing.py`：
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_hashing.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/hashing.py tests/test_hashing.py
git commit -m "feat(data): canonical instance hash"
```

---

## Task 9: 单实例构造

**Files:**
- Create: `experiment/data/generate/corpus.py`
- Test: `tests/test_corpus.py`

- [ ] **Step 1: 写失败测试**

`tests/test_corpus.py`：
```python
import numpy as np
import pytest

from experiment.data.generate import corpus


def test_instance_id_format():
    assert (
        corpus.instance_id("train", "uniform", 20, 0.25, 7)
        == "syn-train-uniform-n20-a25-i0007"
    )


def test_instance_id_pads_index_to_four_digits():
    assert corpus.instance_id("test", "skewed", 200, 0.75, 0).endswith("i0000")


def test_build_instance_has_the_frozen_fields():
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    assert set(inst) == {
        "alpha", "canonical_hash", "capacity", "family", "instance_id",
        "item_order", "n", "resample_attempt", "seed", "source", "stats",
        "values", "weights",
    }


def test_build_instance_basic_values():
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    assert inst["source"] == "synthetic"
    assert inst["family"] == "uniform"
    assert inst["n"] == 20
    assert inst["alpha"] == 0.25
    assert len(inst["weights"]) == 20
    assert len(inst["values"]) == 20
    assert inst["resample_attempt"] == 0


def test_item_order_is_a_permutation_of_range_n():
    inst = corpus.build_instance("train", "uniform", 50, 0.50, 3)
    assert sorted(inst["item_order"]) == list(range(50))


def test_build_instance_is_deterministic():
    a = corpus.build_instance("train", "correlated", 100, 0.75, 5)
    b = corpus.build_instance("train", "correlated", 100, 0.75, 5)
    assert a == b


def test_capacity_matches_formula():
    import math
    inst = corpus.build_instance("train", "skewed", 200, 0.25, 1)
    assert inst["capacity"] == math.floor(0.25 * sum(inst["weights"]))


def test_different_splits_give_different_instances():
    a = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    b = corpus.build_instance("test", "uniform", 20, 0.25, 0)
    assert a["canonical_hash"] != b["canonical_hash"]


def test_attempt_changes_the_instance():
    a = corpus.build_instance("train", "uniform", 20, 0.25, 0, attempt=0)
    b = corpus.build_instance("train", "uniform", 20, 0.25, 0, attempt=1)
    assert a["canonical_hash"] != b["canonical_hash"]


def test_seed_is_recorded_and_matches_derivation():
    from experiment.data.generate.seeds import derive_seed
    inst = corpus.build_instance("validation", "normal", 50, 0.50, 9)
    assert inst["seed"] == derive_seed("validation", "normal", 50, 0.50, 9, 0, "items")


def test_item_order_stream_is_isolated_from_items_stream():
    """排列必须来自 order 流，而不是 items 流用剩的 rng 状态。"""
    from experiment.data.generate.seeds import derive_seed
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    rng = np.random.default_rng(
        derive_seed("train", "uniform", 20, 0.25, 0, 0, "order")
    )
    assert inst["item_order"] == rng.permutation(20).tolist()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'experiment.data.generate.corpus'`

- [ ] **Step 3: 写实现**

`experiment/data/generate/corpus.py`：
```python
"""cell 编排、配额、跨 split 去重与落盘。

本模块只做编排，不知道任一族的抽样公式。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment.data.generate.families import (
    capacity_for,
    compute_stats,
    generate_items,
)
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    MAX_RESAMPLE_ATTEMPT,
    REPO_ROOT,
    SCHEMA_VERSION,
    alpha_key,
    alpha_token,
    all_cells,
)


def instance_id(split: str, family: str, n: int, alpha: float, index: int) -> str:
    """如 ``syn-train-uniform-n20-a25-i0007``。"""
    return f"syn-{split}-{family}-n{n}-{alpha_token(alpha)}-i{index:04d}"


def build_instance(
    split: str,
    family: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
) -> dict:
    """构造一个实例的完整记录。

    ``items`` 流产生权重与价值，``order`` 流独立产生物品排列；两条流互不
    影响，因此在 items 流中增减抽样不会改变已冻结的排列。
    """
    seed = derive_seed(split, family, n, alpha, index, attempt, "items")
    weights, values = generate_items(family, n, seed)
    capacity = capacity_for(alpha, weights)

    order_rng = np.random.default_rng(
        derive_seed(split, family, n, alpha, index, attempt, "order")
    )
    item_order = order_rng.permutation(n).tolist()

    return {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": family,
        "instance_id": instance_id(split, family, n, alpha, index),
        "item_order": item_order,
        "n": n,
        "resample_attempt": attempt,
        "seed": seed,
        "source": "synthetic",
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/corpus.py tests/test_corpus.py
git commit -m "feat(data): 单实例构造与双流隔离的物品排列"
```

---

## Task 10: 跨 split 去重与确定性重抽

**Files:**
- Modify: `experiment/data/generate/corpus.py`（追加 `build_cell`）
- Modify: `tests/test_corpus.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_corpus.py` 末尾追加：
```python
def test_build_cell_returns_exactly_quota_instances():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 125, seen)
    assert len(instances) == 125


def test_build_cell_records_every_hash_in_seen():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 10, seen)
    assert {i["canonical_hash"] for i in instances} == seen


def test_build_cell_has_no_internal_duplicates():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 125, seen)
    hashes = [i["canonical_hash"] for i in instances]
    assert len(set(hashes)) == len(hashes)


def test_build_cell_resamples_when_hash_already_seen():
    """把某个已知哈希预先塞进 seen，该实例必须被重抽掉。"""
    known = corpus.build_instance("train", "uniform", 20, 0.25, 0)["canonical_hash"]
    seen = {known}
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 1, seen)
    assert instances[0]["canonical_hash"] != known
    assert instances[0]["resample_attempt"] >= 1


def test_build_cell_is_deterministic_across_calls():
    a = corpus.build_cell("test", "skewed", 50, 0.75, 8, set())
    b = corpus.build_cell("test", "skewed", 50, 0.75, 8, set())
    assert a == b


def test_build_cell_shares_seen_across_cells():
    seen = set()
    first = corpus.build_cell("train", "uniform", 20, 0.25, 20, seen)
    corpus.build_cell("train", "uniform", 20, 0.50, 20, seen)
    assert seen.issuperset(i["canonical_hash"] for i in first)


def test_resample_cap_raises_instead_of_looping_forever():
    """把上限设为 0，则任何一次碰撞都立刻触顶——用于验证死循环防护存在。

    刻意不用 max_attempt=3：那需要人为造出连续 4 次碰撞，代价高且脆弱。
    """
    seen = set()
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    seen.add(inst["canonical_hash"])
    with pytest.raises(RuntimeError, match="resample"):
        corpus.build_cell("train", "uniform", 20, 0.25, 1, seen, max_attempt=0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: FAIL，`AttributeError: module ... has no attribute 'build_cell'`

- [ ] **Step 3: 写实现**

在 `experiment/data/generate/corpus.py` 的 `build_instance` 之后追加：
```python
def build_cell(
    split: str,
    family: str,
    n: int,
    alpha: float,
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
) -> list[dict]:
    """构造一个 cell 的全部实例，并就地更新跨 split 的已用哈希集合。

    哈希冲突时递增 ``attempt`` 重派生，直到哈希未被占用。配额不因重抽改变：
    重抽只换实例内容，不减少条数。

    Raises:
        RuntimeError: ``attempt`` 超过 ``max_attempt`` 仍未取得唯一哈希。
            宁可报错也不死循环。
    """
    instances: list[dict] = []
    for index in range(quota):
        attempt = 0
        while True:
            instance = build_instance(split, family, n, alpha, index, attempt)
            if instance["canonical_hash"] not in seen:
                break
            attempt += 1
            if attempt > max_attempt:
                raise RuntimeError(
                    f"resample exceeded {max_attempt} attempts for "
                    f"{split}/{family}/n={n}/alpha={alpha}/index={index}"
                )
        seen.add(instance["canonical_hash"])
        instances.append(instance)
    return instances
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/corpus.py tests/test_corpus.py
git commit -m "feat(data): 跨 split 去重与确定性重抽"
```

---

## Task 11: 落盘与 manifest

**Files:**
- Modify: `experiment/data/generate/corpus.py`（追加 `write_cell`、`build_manifest`、`run_generation`）
- Modify: `tests/test_corpus.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_corpus.py` 末尾追加：
```python
import hashlib
import json
from pathlib import Path


def _one_instance(split="train"):
    return corpus.build_instance(split, "uniform", 20, 0.25, 0)


def test_write_cell_creates_lf_terminated_jsonl(tmp_path):
    path = tmp_path / "cell.jsonl"
    digest = corpus.write_cell(path, [_one_instance()])
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw, "必须是 LF 换行，否则字节级不可复现"
    assert digest == hashlib.sha256(raw).hexdigest()


def test_write_cell_roundtrips(tmp_path):
    path = tmp_path / "cell.jsonl"
    instance = _one_instance()
    corpus.write_cell(path, [instance])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == instance


def test_write_cell_is_byte_identical_on_rerun(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    corpus.write_cell(a, [_one_instance()])
    corpus.write_cell(b, [_one_instance()])
    assert a.read_bytes() == b.read_bytes()


def test_run_generation_writes_every_cell_and_manifest(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 2, "validation": 1, "test": 1})
    assert (tmp_path / "manifest.json").is_file()
    assert len(manifest["cells"]) == 48 * 3
    for split in ("train", "validation", "test"):
        expected = {"train": 2, "validation": 1, "test": 1}[split]
        for cell in [c for c in manifest["cells"] if c["split"] == split]:
            assert cell["count"] == expected


def test_manifest_records_actual_file_hashes(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 1, "validation": 1, "test": 1})
    for cell in manifest["cells"]:
        raw = (tmp_path / cell["path"]).read_bytes()
        assert cell["sha256"] == hashlib.sha256(raw).hexdigest()


def test_manifest_global_hashes_are_unique(tmp_path):
    corpus.run_generation(out_dir=tmp_path, quota_override={"train": 3, "validation": 2, "test": 2})
    hashes = []
    for path in sorted(tmp_path.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            hashes.append(json.loads(line)["canonical_hash"])
    assert len(set(hashes)) == len(hashes), "跨 split 出现重复实例"


def test_run_generation_is_idempotent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    corpus.run_generation(out_dir=first, quota_override={"train": 2, "validation": 1, "test": 1})
    corpus.run_generation(out_dir=second, quota_override={"train": 2, "validation": 1, "test": 1})
    for path in sorted(first.rglob("*.jsonl")):
        twin = second / path.relative_to(first)
        assert path.read_bytes() == twin.read_bytes(), f"{path.name} 不可复现"


def test_manifest_schema_fields(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 1, "validation": 1, "test": 1})
    assert manifest["schema_version"] == 1
    assert set(manifest) == {
        "cells", "dedup", "generated_at", "generator_fingerprint",
        "schema_version", "spec",
    }
    assert set(manifest["dedup"]) == {
        "collisions_resampled", "max_resample_attempt", "total_instances",
    }
    assert set(manifest["spec"]) == {
        "alphas", "families", "master_seeds", "quota", "sizes",
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: FAIL，`AttributeError: module ... has no attribute 'write_cell'`

- [ ] **Step 3: 写实现**

先在 `experiment/data/generate/corpus.py` 顶部把 import 补齐：
```python
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.data.generate.families import (
    capacity_for,
    compute_stats,
    generate_items,
)
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    ALPHAS,
    FAMILIES,
    GENERATION_DIR,
    MASTER_SEEDS,
    MAX_RESAMPLE_ATTEMPT,
    REPO_ROOT,
    SCHEMA_VERSION,
    SIZES,
    SPLITS,
    SYNTHETIC_QUOTA,
    alpha_token,
    all_cells,
    cell_filename,
)
```

然后在 `build_cell` 之后追加：
```python
def write_cell(path: Path, instances: list[dict]) -> str:
    """把实例写成 JSONL，返回内容的 SHA-256。

    统一的序列化约定：``sort_keys=True`` 保证键序稳定，LF 换行，末尾留一个
    换行符。三者共同保证同一份实例集合永远产出同样的字节。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(instance, sort_keys=True, ensure_ascii=False)
        for instance in instances
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def generator_fingerprint() -> str:
    """所有生成器模块内容的合并哈希，用于事后诊断「输出为何变了」。"""
    modules = [
        Path(__file__),
        Path(__file__).with_name("families.py"),
        Path(__file__).with_name("hashing.py"),
        Path(__file__).with_name("seeds.py"),
        REPO_ROOT / "experiment" / "data" / "spec.py",
    ]
    digest = hashlib.sha256()
    for module in sorted(modules):
        digest.update(module.read_bytes())
    return digest.hexdigest()


def run_generation(
    out_dir: Path = GENERATION_DIR,
    quota_override: dict[str, int] | None = None,
) -> dict:
    """生成全部 cell 与 manifest。

    遍历顺序是冻结语义：split -> family -> n -> alpha -> index。一个实例只
    可能与先于它的实例冲突，因此单遍生成即可判定，无需二次扫描。

    Args:
        out_dir: 输出根目录，默认 ``datasets/raw/generation``
        quota_override: 覆盖每 cell 配额，仅供测试用小样本
    """
    quotas = quota_override or SYNTHETIC_QUOTA
    seen: set[str] = set()
    cells: list[dict] = []
    total_resamples = 0
    max_attempt_seen = 0

    for split in SPLITS:
        quota = quotas[split]
        for family, n, alpha in all_cells():
            instances = build_cell(split, family, n, alpha, quota, seen)
            relative = Path(split) / cell_filename(family, n, alpha)
            digest = write_cell(out_dir / relative, instances)

            attempts = [i["resample_attempt"] for i in instances]
            total_resamples += sum(attempts)
            max_attempt_seen = max(max_attempt_seen, max(attempts))

            cells.append(
                {
                    "alpha": alpha,
                    "count": len(instances),
                    "family": family,
                    "n": n,
                    "path": relative.as_posix(),
                    "resample_attempts": sum(attempts),
                    "sha256": digest,
                    "split": split,
                }
            )

    manifest = {
        "cells": cells,
        "dedup": {
            "collisions_resampled": total_resamples,
            "max_resample_attempt": max_attempt_seen,
            "total_instances": len(seen),
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_fingerprint": generator_fingerprint(),
        "schema_version": SCHEMA_VERSION,
        "spec": {
            "alphas": list(ALPHAS),
            "families": list(FAMILIES),
            "master_seeds": dict(MASTER_SEEDS),
            "quota": dict(quotas),
            "sizes": list(SIZES),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
```

- [ ] **Step 4: 运行测试确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`
Expected: 26 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/data/generate/corpus.py tests/test_corpus.py
git commit -m "feat(data): JSONL 落盘、generator fingerprint 与 manifest"
```

---

## Task 12: 命令行入口与全量生成

**Files:**
- Create: `experiment/data/generate/cli.py`

- [ ] **Step 1: 写实现**

`experiment/data/generate/cli.py`：
```python
"""合成语料生成入口。

用法::

    python -m experiment.data.generate.cli
    python -m experiment.data.generate.cli --out datasets/raw/generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.data.generate.corpus import run_generation
from experiment.data.spec import GENERATION_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成背包语料")
    parser.add_argument(
        "--out",
        type=Path,
        default=GENERATION_DIR,
        help="输出根目录（默认 datasets/raw/generation）",
    )
    args = parser.parse_args()

    manifest = run_generation(out_dir=args.out)

    print(f"输出目录: {args.out}")
    print(f"cell 数: {len(manifest['cells'])}")
    print(f"实例总数: {manifest['dedup']['total_instances']}")
    print(f"重抽次数: {manifest['dedup']['collisions_resampled']}")
    print(f"最大 attempt: {manifest['dedup']['max_resample_attempt']}")
    print(f"generator fingerprint: {manifest['generator_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 验证 --help 可用**

Run: `venv\Scripts\python.exe -m experiment.data.generate.cli --help`
Expected: 打印用法说明，退出码 0

- [ ] **Step 3: 跑全量生成**

Run: `venv\Scripts\python.exe -m experiment.data.generate.cli`
Expected: 打印 `实例总数: 8400`。若 `最大 attempt` 远大于 0，说明哈希冲突异常频繁，停下来排查而不是继续。

- [ ] **Step 4: 核对产物结构**

Run:
```powershell
Get-ChildItem datasets\raw\generation -Recurse -File | Group-Object DirectoryName |
  ForEach-Object { "{0}  {1} 个文件" -f $_.Name, $_.Count }
(Import-Csv datasets\raw\generation\train\uniform_n20_a25.jsonl -Header a | Measure-Object).Count
```
Expected: `train` / `validation` / `test` 三个目录各 48 个 `.jsonl`，加上根目录 1 个 `manifest.json`；最后一个命令输出 `125`。

- [ ] **Step 5: 确认 datasets/ 未被 git 跟踪**

Run: `git status --short`
Expected: **不出现** `datasets/`。若出现，说明 `.gitignore` 失效，先修正再继续。

- [ ] **Step 6: Commit**

```bash
git add experiment/data/generate/cli.py
git commit -m "feat(data): 合成语料生成命令行入口"
```

---

## Task 13: 金样本锁定与最终验收

**Files:**
- Create: `tests/test_golden.py`
- Create: `tests/golden/`（由测试自动生成并入库）

- [ ] **Step 1: 写失败测试**

`tests/test_golden.py`：
```python
"""金样本：把已冻结的语义钉死，任何改动都会被立刻发现。

金样本文件在首次运行时落地并由人工核对后入库。它锁定的是
(seed, weights, values, capacity, item_order) 这组内容，因此任何
会改变语料的重构（改抽样顺序、改取整方向、改种子派生）都会让它失败。
"""

import json
from pathlib import Path

import pytest

from experiment.data.generate import corpus
from experiment.data.spec import alpha_token

GOLDEN_DIR = Path(__file__).parent / "golden"

CASES = [
    ("train", "uniform", 20, 0.25, 0),
    ("train", "normal", 50, 0.50, 3),
    ("train", "correlated", 100, 0.75, 7),
    ("train", "skewed", 200, 0.25, 11),
    ("validation", "skewed", 200, 0.75, 4),
    ("test", "uniform", 50, 0.50, 24),
]


def _golden_path(split, family, n, alpha, index):
    """复用 spec.alpha_token，避免文件名约定在两处各写一遍而漂移。"""
    return GOLDEN_DIR / f"{split}_{family}_n{n}_{alpha_token(alpha)}_i{index}.json"


@pytest.mark.parametrize("split, family, n, alpha, index", CASES)
def test_matches_golden(split, family, n, alpha, index):
    path = _golden_path(split, family, n, alpha, index)
    built = corpus.build_instance(split, family, n, alpha, index)

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(built, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        pytest.skip(f"金样本首次生成，请人工核对后入库: {path.name}")

    assert built == json.loads(path.read_text(encoding="utf-8"))


def test_golden_weights_are_integers():
    for split, family, n, alpha, index in CASES:
        path = _golden_path(split, family, n, alpha, index)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(isinstance(x, int) for x in data["weights"])
        assert all(isinstance(x, int) for x in data["values"])
```

- [ ] **Step 2: 首次运行生成金样本**

Run: `venv\Scripts\python.exe -m pytest tests/test_golden.py -v`
Expected: 6 skipped，并在 `tests/golden/` 下生成 6 个 `.json`

- [ ] **Step 3: 人工核对金样本**

Run:
```powershell
Get-Content tests\golden\train_skewed_n200_a25_i11.json |
  ConvertFrom-Json | Select-Object instance_id, seed, capacity, n, family
```
Expected: 手工确认三项——(a) `capacity` 等于 `floor(0.25 * sum(weights))`；(b) `weights` 中恰好 10 项 ≥ 200；(c) `item_order` 是 0..199 的一个排列。

- [ ] **Step 4: 第二次运行确认通过**

Run: `venv\Scripts\python.exe -m pytest tests/test_golden.py -v`
Expected: 7 passed（6 个金样本比对 + 1 个整数类型检查）

- [ ] **Step 5: 全套测试**

Run: `venv\Scripts\python.exe -m pytest -v`
Expected: 全部 PASS，无 skipped

- [ ] **Step 6: 端到端幂等性验收**

Run:
```powershell
$gen = "datasets\raw\generation"
$snapshot = Join-Path $env:TEMP "gen_first"
Remove-Item -Recurse -Force $gen, $snapshot -ErrorAction SilentlyContinue

venv\Scripts\python.exe -m experiment.data.generate.cli
Copy-Item -Recurse $gen $snapshot

venv\Scripts\python.exe -m experiment.data.generate.cli

# 比对「相对路径 + 内容哈希」，只跳过含时间戳的 manifest.json
function Get-Fingerprint($root) {
  Get-ChildItem $root -Recurse -File |
    Where-Object Name -ne 'manifest.json' |
    ForEach-Object { "{0} {1}" -f (Resolve-Path $_.FullName -Relative), (Get-FileHash $_.FullName).Hash } |
    Sort-Object
}
$diff = Compare-Object (Get-Fingerprint $gen) (Get-Fingerprint $snapshot)
if ($diff) { $diff; "不一致：重跑未能字节级复现" } else { "幂等性通过：全部产物字节级相同" }
```
Expected: `幂等性通过：全部产物字节级相同`，且 `$diff` 为空。

- [ ] **Step 7: 核对 50:50 之外的配额恒等式**

Run:
```powershell
$m = Get-Content datasets\raw\generation\manifest.json | ConvertFrom-Json
$m.cells | Group-Object split | ForEach-Object {
  "{0}: {1} 个 cell, 合计 {2} 个实例" -f $_.Name, $_.Count, ($_.Group | Measure-Object count -Sum).Sum }
"dedup: $($m.dedup.total_instances) 实例, 重抽 $($m.dedup.collisions_resampled) 次, 最大 attempt $($m.dedup.max_resample_attempt)"
```
Expected: `train: 48 个 cell, 合计 6000`、`validation: 48 个 cell, 合计 1200`、`test: 48 个 cell, 合计 1200`、`dedup: 8400 实例`。

- [ ] **Step 8: Commit**

```bash
git add tests/test_golden.py tests/golden
git commit -m "test(data): 金样本锁定已冻结的生成语义"
```

---

## 完成标准

对照设计文档第 11 节：

1. `pytest` 全绿（Task 13 Step 5）
2. 144 个 cell 齐备且配额精确（Task 13 Step 7）
3. 8400 个实例哈希全局唯一（Task 11 的 `test_manifest_global_hashes_are_unique` + Task 13 Step 7）
4. 每个 cell 的 SHA-256 与 manifest 一致（Task 11 的 `test_manifest_records_actual_file_hashes`）
5. 重跑字节级相同（Task 13 Step 6）
6. `datasets/` 未被 git 跟踪（Task 12 Step 5）

设计文档第 11 节第 6、7 条属于 `collect` 子系统的验收，不在本计划范围，见配套计划 `2026-09-25-uci-collection-and-cleaning.md`。
