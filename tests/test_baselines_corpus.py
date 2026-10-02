"""全语料验收（设计 §10）。

前置事实全部实测于 2026-09-30（探针 probe3.py）：3,300 实例 / 36,300 行 /
violations 合计 0 / 无行的 value 超过 V* / Greedy 平均 V/U 0.975118、
Random 平均 V/U 0.559955。
"""

import csv
from fractions import Fraction

import pytest

from experiment.baselines import cli
from experiment.data.spec import DATASETS_PROCESSED, REPO_ROOT

#: `datasets/` 被 gitignore，临时副本里没有它。本项目的完成标准是**真仓库里
#: 0 skipped**，而这个仓库里语料是在的，所以这条测试在本机真跑；在没有语料的副本
#: 里它跳过而不是失败。
CORPUS_TEST_FILE = DATASETS_PROCESSED / "test" / "instances.jsonl"
REFERENCE_CSV = REPO_ROOT / "reference_results.csv"

pytestmark = pytest.mark.skipif(
    not CORPUS_TEST_FILE.is_file() or not REFERENCE_CSV.is_file(),
    reason="冻结语料或参考解不在位（datasets/ 被 gitignore）；在本仓库里会真跑",
)

SEED = 6001
TRIALS = 10
INSTANCES = 3_300
ROWS = INSTANCES * (TRIALS + 1)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """在**真实语料**上跑一遍入口，产物写到临时目录，再把 CSV 读回来。

    写出去再读回来，顺带把序列化路径也验了。
    """
    out = tmp_path_factory.mktemp("baselines") / "baseline_results.csv"
    summary = cli.run(DATASETS_PROCESSED, out, evaluation_seed=SEED, trials=TRIALS)
    with out.open(encoding="utf-8", newline="") as handle:
        return summary, list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def reference():
    """A 的参考解：``instance_id -> (V*, U)``。"""
    with REFERENCE_CSV.open(encoding="utf-8", newline="") as handle:
        return {
            row["instance_id"]: (int(row["value"]), Fraction(row["upper_bound"]))
            for row in csv.DictReader(handle)
        }


def test_the_corpus_is_the_size_this_plan_was_written_against():
    """语料规模漂移会静默改变下面每一条断言的分母，所以先钉住它。"""
    instances = cli.read_instances(cli.corpus_files(DATASETS_PROCESSED))
    assert len(instances) == INSTANCES
    assert {label for label, _ in instances} == {"test", "e3_size", "e4_tail"}


def test_the_row_count_is_instances_times_eleven(results):
    summary, rows = results
    assert summary["rows"] == ROWS
    assert len(rows) == ROWS
    assert summary["by_method"] == {"greedy": INSTANCES, "random": INSTANCES * TRIALS}


def test_every_instance_is_present_once_per_method_and_trial(results, reference):
    """每个实例恰好 10 条 random + 1 条 greedy，且它的参考解存在。

    **与 `reference` 的关系是包含，不是相等。** `reference_results.csv` 覆盖**全部
    17,700 个实例**（含 `train` / `validation`），而 B 只跑 3,300 个——写成
    ``set(by_instance) == set(reference)`` 会当场失败，右边多出 14,400 个。正向探针
    实测到了这一次失败。所以是：B 的集合**是**参考解集合的子集，且大小正好是 3,300。
    """
    _, rows = results
    by_instance: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], {}).setdefault(row["method"], []).append(
            row["trial"]
        )
    assert set(by_instance) <= set(reference), "有实例找不到参考解"
    assert len(by_instance) == INSTANCES, "B 的实例集大小不对"
    for instance_id, methods in by_instance.items():
        assert sorted(methods["random"], key=int) == [str(t) for t in range(TRIALS)], instance_id
        assert methods["greedy"] == [""], instance_id


def test_no_instance_id_contains_the_key_separator(results):
    """设计 §4 的种子键用 ``|`` 拼接，这就要求各分量不含 ``|``。"""
    _, rows = results
    assert [r["instance_id"] for r in rows if "|" in r["instance_id"]] == []


def test_every_row_is_feasible_and_no_value_exceeds_the_optimum(results, reference):
    """两个**独立计算**的产物之间的一致性。

    这一条**不是**"贪心写对了"的证明：任何**可行**选择的 value 都不可能超过 ``V*``
    ——那是 ``V*`` 的定义。它挡的是**跨子系统的不一致**：实例集错位、参考解列读错、
    ``value`` 从错误的数组求和。

    排序方向写反、``break`` 代替 ``continue`` 这类缺陷**能通过这一条**（它们给出的
    解仍然可行，只是更差），由 `tests/test_baselines_greedy.py` 的合成实例钉住；
    下一条测试补的正是这个缺口。
    """
    _, rows = results
    for row in rows:
        assert int(row["value"]) <= reference[row["instance_id"]][0], row
        assert row["violations"] == "0", row


def test_greedy_average_quality_is_strictly_above_random(results, reference):
    """Greedy 的平均 ``V/U`` **严格高于** Random 的平均 ``V/U``。

    这是上一条补不上的那个缺口：排序反了或提前 ``break`` 的贪心仍然**可行**，
    只是**更差**——在 3,300 个实例的平均上会掉向 Random 的 0.5 掷币。

    只断言严格不等式、**不写死阈值**：阈值会在语料修订后变成一条假失败，而不等式
    不会。实测证据记在这里供人核对：Greedy **0.975118**、Random **0.559955**，
    差额 0.415163。
    """
    _, rows = results
    scored: dict[str, list[Fraction]] = {"greedy": [], "random": []}
    for row in rows:
        upper = reference[row["instance_id"]][1]
        scored[row["method"]].append(Fraction(int(row["value"])) / upper)

    greedy = sum(scored["greedy"]) / len(scored["greedy"])
    random_ = sum(scored["random"]) / len(scored["random"])
    assert greedy > random_
