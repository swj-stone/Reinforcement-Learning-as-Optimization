"""性质测试：生产形状语料上的不变量（设计 §12）。

为什么用**生成器产出的实例**而不是直接读 ``datasets/``：该目录被 gitignore，
新克隆上不存在，依赖它会让测试静默跳过或静默通过——这正是本项目的验收不
允许的。真要验证「冻结语料上 U >= V* 处处成立」，那是 Task 7 的 CLI 验收
步骤：它在真实语料上跑一遍并写进报告，而不是伪装成一条 CI 断言。

用 ``quota_override`` 把每 cell 压到 1 条，但**族、n、alpha 一个不少**，因此
容量量级与真实语料同阶（n=200 时上万格）——这正是「上界算成下界」或「浮点
排序」会露馅的地方。
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from experiment.data.generate.corpus import run_generation
from experiment.data.spec import SPLITS
from experiment.eval.reference import fractional_upper_bound, item_set_hash, solve

#: 4 族 x 4 个 n x 3 个 alpha x 3 个 split
EXPECTED_INSTANCES = 144


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("eval_gen")
    run_generation(out_dir=out, quota_override={split: 1 for split in SPLITS})
    instances: list[dict] = []
    for split in SPLITS:
        for cell in sorted((out / split).glob("*.jsonl")):
            for line in cell.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    instances.append(json.loads(line))
    return instances


def _solve_with(instance: Mapping, weights, values):
    """用**替代的**物品数据求解，容量与 ``instance_id`` 仍取自原实例。"""
    return solve(
        {
            "capacity": instance["capacity"],
            "instance_id": instance["instance_id"],
            "values": values,
            "weights": weights,
        }
    )


def test_upper_bound_dominates_the_optimum(generated_instances):
    assert len(generated_instances) == EXPECTED_INSTANCES
    for instance in generated_instances:
        result = solve(instance)
        upper = fractional_upper_bound(
            instance["weights"], instance["values"], instance["capacity"]
        )
        assert result.upper_bound == upper, instance["instance_id"]
        assert upper >= result.value, instance["instance_id"]


def test_reconstructed_sets_are_feasible_and_optimal(generated_instances):
    for instance in generated_instances:
        weights, values = instance["weights"], instance["values"]
        result = solve(instance)
        chosen = result.selected_indices
        assert len(set(chosen)) == len(chosen), instance["instance_id"]
        assert (
            sum(weights[i] for i in chosen) <= instance["capacity"]
        ), instance["instance_id"]
        assert sum(values[i] for i in chosen) == result.value, instance["instance_id"]


def test_upper_bound_and_optimum_are_permutation_invariant(generated_instances):
    """``U`` 与 ``V*`` 是物品**多重集**的函数，与存储顺序无关。

    刻意**不**断言 ``item_set_hash`` 也不变：打乱物品顺序会把最优解映射成
    另一组索引，哈希因此本就应该变（并列最优时尤甚）。
    """
    rng = np.random.default_rng(20260929)
    for instance in generated_instances:
        weights, values = instance["weights"], instance["values"]
        capacity = instance["capacity"]
        permutation = rng.permutation(len(weights)).tolist()
        shuffled_w = [weights[i] for i in permutation]
        shuffled_v = [values[i] for i in permutation]

        assert (
            fractional_upper_bound(shuffled_w, shuffled_v, capacity)
            == fractional_upper_bound(weights, values, capacity)
        ), instance["instance_id"]
        assert (
            _solve_with(instance, shuffled_w, shuffled_v).value
            == solve(instance).value
        ), instance["instance_id"]


def test_item_set_hash_depends_on_the_set_not_the_list_order():
    """哈希先排序再拼接，因此与传入次序无关（设计 §12）。"""
    assert item_set_hash([2, 0, 1]) == item_set_hash([0, 1, 2])
    assert item_set_hash([5, 3]) == item_set_hash([3, 5])
    assert item_set_hash([0, 1]) != item_set_hash([0, 2])
    assert item_set_hash([1]) != item_set_hash([1, 1])


def test_adversarial_shapes_keep_the_bound_and_the_reconstruction_valid():
    """边界形状：容量 0、一件都整件装不下、同比值并列、零重量的物品、
    重量恰好等于容量的物品、最优值可由多个容量列达到。

    这些形状真实语料里一个都没有（实测平凡解 0/17700），所以必须合成。
    """
    instances = [
        {"capacity": 0, "values": [1, 2], "weights": [1, 2]},
        {"capacity": 1, "values": [5, 6], "weights": [2, 3]},
        {"capacity": 6, "values": [1, 2, 3], "weights": [1, 2, 3]},
        {"capacity": 5, "values": [7, 7], "weights": [5, 5]},
        {"capacity": 10, "values": [0, 0], "weights": [1, 2]},
        {"capacity": 3, "values": [4], "weights": [0]},
    ]
    for instance in instances:
        result = solve(instance)
        upper = fractional_upper_bound(
            instance["weights"], instance["values"], instance["capacity"]
        )
        assert upper >= result.value, instance
        chosen = result.selected_indices
        assert (
            sum(instance["weights"][i] for i in chosen) <= instance["capacity"]
        ), instance
        assert sum(instance["values"][i] for i in chosen) == result.value, instance

    # 下面两条**必须钉住具体期望值**，不能走上面的自洽循环：自洽断言对
    # 「一件都没装进去」同样成立，因此对这两个缺陷一律沉默。两条都已在两版
    # 变异体上探针验证过（`wi <= capacity` → `wi < capacity`；重建起点取最大
    # 列而非最小列）。

    # 重量**恰好**等于容量的物品必须能被选上。变异体把它整件丢掉，V* 由 10
    # 变成 0，而自洽断言照样全绿（空集重量 0 <= 5、价值 0 == 0）。
    assert solve({"capacity": 5, "values": [10], "weights": [5]}).value == 10

    # 最优值能由**多个**容量列达到时，起点取**最小**列，结果才确定。这里 {0}
    # 与 {1} 都达到 5（重量 1 与 2 都 <= 2）；取最大列会选成 [1]，于是
    # selected_mask 与 item_set_hash 一起变——而后者是入库 CSV 的一列。
    tie = solve({"capacity": 2, "values": [5, 5], "weights": [1, 2]})
    assert tie.selected_indices == [0]
