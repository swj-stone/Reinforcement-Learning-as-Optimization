"""``baseline_results.csv`` 的形状、行序、字节、覆写守卫与计时口径（设计 §5、§8、§11）。"""

import csv
import json
import random
import time
from pathlib import Path

import pytest
import yaml

from experiment.baselines import cli
from experiment.baselines.methods import random_solution
from experiment.baselines.seeds import trial_seed
from experiment.data.spec import DATASETS_PROCESSED, E3_NAME, E4_NAME, REPO_ROOT

#: 两个 test 实例的 id **字典序与它们在文件里的出现顺序相反**，这样"行序按
#: instance_id 升序"才有判别力。
TEST_INSTANCES = [
    ("b-second", {"capacity": 5, "item_order": [0, 1], "values": [7, 1], "weights": [5, 1]}),
    ("a-first", {"capacity": 9, "item_order": [2, 0, 1], "values": [10, 7, 9], "weights": [8, 4, 5]}),
]
E3_INSTANCES = [
    ("c-e3", {"capacity": 6, "item_order": [1, 0], "values": [3, 4], "weights": [2, 6]}),
]

#: 计时测试用的规模：要让"一次构造"明显高于计时器噪声，又不拖慢套件。
TIMING_N = 400

#: 产物的列序**逐字**写在这里，而不是只跟 ``cli.COLUMNS`` 比。
#:
#: 只跟 ``cli.COLUMNS`` 比是**自指**的——``DictWriter(fieldnames=cli.COLUMNS)``
#: 写出的表头必然等于它，所以把 ``COLUMNS`` 里两列对调，那条断言照样通过。
#: 子项目 A 的 ``tests/test_eval_cli.py`` 是同一写法（同一处缺口，尚未修）。
EXPECTED_COLUMNS = (
    "instance_id",
    "split",
    "method",
    "trial",
    "seed",
    "value",
    "weight_used",
    "violations",
    "item_set_hash",
    "latency_s",
)


@pytest.fixture
def corpus(tmp_path):
    """手写的三文件语料：test 两条、E3 一条、E4 空。

    刻意手写而不是跑 `run_assembly`：这里要测的是 **CLI 的读取与落盘**，不是装配；
    手写让"读到了哪些文件"完全可控。

    E4 留成**空文件而不是缺失**：真实语料里它存在（300 个实例），空文件能顺带钉住
    "空子集贡献 0 行"，而缺失文件只会让实现撞上 FileNotFoundError。
    """
    root = tmp_path / "processed"
    layout = {
        root / "test" / "instances.jsonl": TEST_INSTANCES,
        root / "stress" / E3_NAME / "instances.jsonl": E3_INSTANCES,
        root / "stress" / E4_NAME / "instances.jsonl": [],
    }
    for path, items in layout.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(dict(body, instance_id=name), sort_keys=True)
            for name, body in items
        ]
        path.write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return root


def _write_config(path: Path, trials: int, seed: int = 6001) -> Path:
    path.write_bytes(
        yaml.safe_dump(
            {"evaluation": {"evaluation_seed": seed, "random_trials_per_instance": trials}},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).encode("utf-8")
    )
    return path


@pytest.fixture
def config(tmp_path):
    return _write_config(tmp_path / "experiment_config.yaml", trials=2)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _argv(corpus, config, out, *extra):
    return [
        "baseline-cli",
        "--corpus", str(corpus),
        "--config", str(config),
        "--out", str(out),
        *extra,
    ]


# ---- 形状、行序、字节 ----


def test_every_subset_is_read_and_the_row_order_is_fixed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    summary = cli.run(corpus, out, evaluation_seed=6001, trials=2)
    # test 2 + E3 1 + E4 0 = 3 个实例 × (2 次随机 + 1 次贪心)
    assert summary["rows"] == 9
    assert summary["by_method"] == {"greedy": 3, "random": 6}
    assert [(r["instance_id"], r["method"], r["trial"]) for r in _rows(out)] == [
        ("a-first", "greedy", ""),
        ("a-first", "random", "0"),
        ("a-first", "random", "1"),
        ("b-second", "greedy", ""),
        ("b-second", "random", "0"),
        ("b-second", "random", "1"),
        ("c-e3", "greedy", ""),
        ("c-e3", "random", "0"),
        ("c-e3", "random", "1"),
    ]
    assert [r["split"] for r in _rows(out)] == [
        "test", "test", "test", "test", "test", "test",
        E3_NAME, E3_NAME, E3_NAME,
    ]


def test_column_order_and_bytes_are_fixed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    with out.open(encoding="utf-8", newline="") as handle:
        assert list(cli.COLUMNS) == list(EXPECTED_COLUMNS)
        assert next(csv.reader(handle)) == list(EXPECTED_COLUMNS)


def test_greedy_has_one_row_per_instance_with_no_seed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)
    greedy = [r for r in _rows(out) if r["method"] == "greedy"]
    assert len(greedy) == 3
    assert {r["trial"] for r in greedy} == {""}
    assert {r["seed"] for r in greedy} == {""}


def test_the_recorded_seed_is_the_trial_seed_of_that_row(corpus, config, tmp_path):
    """``seed`` 列必须等于该行 ``(instance_id, trial)`` 与评估种子的派生值。

    写进产物就是为了让复核者**不必重跑**就能重算任意一行；若这一列装的是别的
    东西（比如试验号本身），它就没有这个用处。
    """
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)
    for row in _rows(out):
        if row["method"] == "greedy":
            continue
        expected = trial_seed(row["instance_id"], int(row["trial"]), 6001)
        assert row["seed"] == str(expected), row


def test_trials_are_ordered_numerically_not_lexicographically(corpus, tmp_path):
    """``trial`` 列按**数值**升序。

    按字符串排会把 "10" 排在 "2" 前面：0, 1, 10, 11, 2, …。而 10 次试验的试验号是
    0..9，**全是单字符**，字符串序与数值序恰好一样——所以这条测试必须用 **12** 次
    试验才具判别力。
    """
    config = _write_config(tmp_path / "c12.yaml", trials=12)
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=12)
    for instance_id in ("a-first", "b-second", "c-e3"):
        trials = [
            r["trial"]
            for r in _rows(out)
            if r["instance_id"] == instance_id and r["method"] == "random"
        ]
        assert trials == [str(t) for t in range(12)], instance_id


def test_only_the_latency_column_is_unstable(corpus, config, tmp_path):
    """两次运行除 ``latency_s`` 外逐格相同（设计 §5）。

    该列是墙钟测量。**任何把整份 CSV 一起哈希**的校验都会重蹈 Plan 3 的覆辙：当时
    一个存活探针把含临时路径与时钟的 manifest 一起哈希，两次纯净运行都不一致，
    于是每一个"存活"判定都是假的。比对必须排除这一列。
    """
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.run(corpus, first, evaluation_seed=6001, trials=2)
    cli.run(corpus, second, evaluation_seed=6001, trials=2)

    def without_latency(path: Path) -> list[list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        index = rows[0].index("latency_s")
        return [[c for i, c in enumerate(row) if i != index] for row in rows]

    assert without_latency(first) == without_latency(second)


# ---- 参数来源与默认值 ----


def test_read_config_reads_both_parameters(tmp_path):
    config = _write_config(tmp_path / "c.yaml", trials=7, seed=123)
    assert cli.read_config(config) == (123, 7)


def test_default_paths_point_at_the_repo_root():
    """默认产物落在仓库根目录（设计 §8）。

    这条钉的是"产物入库"这个决定：``datasets/`` 被 gitignore，默认写到那里等于
    生成的文件永远不会出现在提交里。
    """
    args = cli.build_parser().parse_args([])
    assert args.corpus == DATASETS_PROCESSED
    assert args.config == REPO_ROOT / "experiment_config.yaml"
    assert args.out == REPO_ROOT / "baseline_results.csv"
    assert args.force is False


# ---- 入口 ----


def test_the_trial_count_comes_from_the_config(corpus, tmp_path, monkeypatch):
    """行数随配置里的 trials 变：10 不是硬编码的。"""
    for trials in (2, 5):
        config = _write_config(tmp_path / f"c{trials}.yaml", trials=trials)
        out = tmp_path / f"t{trials}.csv"
        monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
        assert cli.main() == 0
        assert len(_rows(out)) == 3 * (trials + 1)


def test_main_writes_and_reports_where_it_wrote(corpus, config, tmp_path, monkeypatch, capsys):
    out = tmp_path / "baseline_results.csv"
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 0
    assert len(_rows(out)) == 9
    assert str(out) in capsys.readouterr().out


def test_main_does_not_label_zero_valued_rows_as_trivial(corpus, config, tmp_path, monkeypatch, capsys):
    """B 的汇总**不**打印"平凡解"计数。

    A 的入口有那一行（``value == 0`` 的行），在 A 那里它是对的。搬到这里就错了：
    Random 抽到全反面时返回空集、``value`` 为 0，那是**合法的一次抽样**，不是计划
    §5 说的"没有物品装得下的平凡零最优实例"。一个按 value 数的"平凡解"计数会把
    两者混为一谈。
    """
    out = tmp_path / "baseline_results.csv"
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 0
    printed = capsys.readouterr().out
    assert "平凡" not in printed
    assert str(out) in printed


def test_main_refuses_to_overwrite_without_force(corpus, config, tmp_path, monkeypatch, capsys):
    """目标已存在且没给 ``--force`` ⇒ 报错退出，且**文件逐字节未变**（设计 §8）。

    A 上真的发生过一次：裸跑 CLI 把已入库的 ``reference_results.csv`` 覆写成不同
    字节。只断言"退出码 1"不够——必须同时断言文件内容没变，因为一个"报了错但已经
    写了一半"的实现在退出码上看起来是对的。
    """
    out = tmp_path / "baseline_results.csv"
    out.write_bytes(b"sentinel\n")
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 1
    assert out.read_bytes() == b"sentinel\n"
    assert "force" in capsys.readouterr().err


def test_force_overwrites_and_writes_every_row(corpus, config, tmp_path, monkeypatch):
    out = tmp_path / "baseline_results.csv"
    out.write_bytes(b"sentinel\n")
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out, "--force"))
    assert cli.main() == 0
    assert len(_rows(out)) == 9


# ---- 计时口径 ----


def _timing_instance() -> dict:
    rng = random.Random(20260930)
    weights = [rng.randint(1, 100) for _ in range(TIMING_N)]
    values = [rng.randint(1, 100) for _ in range(TIMING_N)]
    order = list(range(TIMING_N))
    rng.shuffle(order)
    return {
        "capacity": sum(weights) // 3,
        "instance_id": "timing-0001",
        "item_order": order,
        "values": values,
        "weights": weights,
    }


@pytest.fixture
def timing_corpus(tmp_path):
    """只有一个大实例的语料，专供计时测试。"""
    root = tmp_path / "processed"
    path = root / "test" / "instances.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(_timing_instance(), sort_keys=True) + "\n").encode("utf-8"))
    for parts in (("stress", E3_NAME), ("stress", E4_NAME)):
        empty = root.joinpath(*parts, "instances.jsonl")
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_bytes(b"")
    return root


def test_latency_is_one_construction_not_the_sum_of_ten(timing_corpus, tmp_path):
    """单行 ``latency_s`` 是**一次**构造的耗时，不是十次之和（设计 §5）。

    计划 §5 专门禁止了"把十次试验的时长当成一个解的时长上报"。十次之和会落在下面
    ``ten`` 的量级上，而单次约为它的 1/10；取 5 倍余量避开计时噪声。

    断言写成"``latency_s > 0``"是空转的——那样连"记的是十次之和"都挡不住。
    """
    out = tmp_path / "baseline_results.csv"
    cli.run(timing_corpus, out, evaluation_seed=6001, trials=10)

    instance = _timing_instance()
    start = time.perf_counter()
    for trial in range(10):
        random_solution(instance, trial_seed(instance["instance_id"], trial, 6001))
    ten = time.perf_counter() - start

    latencies = [float(r["latency_s"]) for r in _rows(out) if r["method"] == "random"]
    assert len(latencies) == 10
    assert max(latencies) < ten / 2
