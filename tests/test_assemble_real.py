"""真实实例的抽样、溯源回连、容量公式与重抽。"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from experiment.data.assemble import lots as L
from experiment.data.assemble import real as R
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import MAX_RESAMPLE_ATTEMPT, REAL_SEED_TOKEN

N = 6


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


def _table(path: Path) -> dict[int, dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(r["source_row_id"]): r for r in csv.DictReader(handle)}


def _pool_of(index, inst):
    """按 source_row_ids 从当日池里取回批次对象。"""
    pool = {lot.source_row_id: lot for lot in index.pools["train"][inst["source_date"]]}
    return [pool[rid] for rid in inst["source_row_ids"]]


# --- 标识与基本形状 ---------------------------------------------------------


def test_real_instance_id_format():
    assert R.real_instance_id("train", 20, 0.25, 7) == "real-train-n20-a25-i0007"
    assert R.real_instance_id("e3_size", 500, 0.75, 0) == "real-e3_size-n500-a75-i0000"


def test_real_instance_pads_index_to_four_digits(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 3)
    assert inst["instance_id"] == f"real-train-n{N}-a25-i0003"


def test_real_instance_has_the_frozen_field_set(index):
    """设计文档 §8.2：真实实例恰好在合成实例的 13 个键之上多三个。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert set(inst) == {
        "alpha", "canonical_hash", "capacity", "family", "instance_id",
        "invoice_nos", "item_order", "n", "resample_attempt", "seed",
        "source", "source_date", "source_row_ids", "stats", "values", "weights",
    }


def test_real_instance_declares_itself_real_and_familyless(index):
    """真实数据没有「族」。不变量：family is None <=> source == "real"。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert inst["source"] == "real"
    assert inst["family"] is None


def test_real_instance_lengths_are_consistent(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 1)
    assert len(inst["weights"]) == len(inst["values"]) == N
    assert len(inst["item_order"]) == N
    assert sorted(inst["item_order"]) == list(range(N))
    assert len(inst["source_row_ids"]) == len(inst["invoice_nos"]) == N
    assert all(w > 0 for w in inst["weights"])
    assert all(v > 0 for v in inst["values"])


# --- 抽样语义 ---------------------------------------------------------------


def test_sample_comes_from_a_single_eligible_date(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert inst["source_date"] in index.eligible("train", N)
    pool_ids = {lot.source_row_id for lot in index.pools["train"][inst["source_date"]]}
    assert set(inst["source_row_ids"]) <= pool_ids


def test_sampled_rows_are_distinct(index):
    """第 4 节：无放回抽 n 个不同批次。"""
    for i in range(20):
        inst = R.build_real_instance(index, "train", "train", N, 0.25, i)
        ids = inst["source_row_ids"]
        assert len(set(ids)) == len(ids) == N


def test_sampled_rows_are_not_just_the_pool_prefix(index):
    """防退化：若实现偷懒取池子前 n 条，下标就会恒等于 (0,1,...,n-1)。

    夹具每天 24 条、抽 6 条，C(24,6)=134596 种组合里「正好是前 6 条」只占
    一种；连抽 30 次都命中前缀的概率可以忽略。这条断言让「取前缀」这种
    退化实现必然失败。
    """
    hits = 0
    for i in range(30):
        inst = R.build_real_instance(index, "train", "train", N, 0.25, i)
        pool = index.pools["train"][inst["source_date"]]
        positions = sorted(
            next(j for j, lot in enumerate(pool) if lot.source_row_id == rid)
            for rid in inst["source_row_ids"]
        )
        hits += positions == list(range(N))
    assert hits == 0, "抽样看起来只是取池子前 n 条"


def test_row_order_within_the_instance_follows_the_rng(index):
    """weights/values 必须与 source_row_ids 同序，顺序由 items 流决定。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 2)
    picked = _pool_of(index, inst)
    assert inst["weights"] == [lot.quantity_w for lot in picked]
    assert inst["values"] == [lot.value_pence for lot in picked]
    assert inst["invoice_nos"] == [lot.invoice_no for lot in picked]


# --- 溯源回连（本阶段最重要的一条检查）--------------------------------------


def test_sampled_values_join_back_to_the_lots_table(lots_csv, index):
    """按 source_row_ids 回连批次表，weights/values/invoice 必须逐位相符。

    这条让「抽样被篡改」无处藏身：任何一只手改动都会在回连时暴露。
    """
    table = _table(lots_csv)
    for i in range(5):
        inst = R.build_real_instance(index, "train", "train", N, 0.75, i)
        for rid, w, v, invoice in zip(
            inst["source_row_ids"], inst["weights"], inst["values"], inst["invoice_nos"]
        ):
            row = table[rid]
            assert w == int(row["quantity_w"])
            assert v == int(row["value_pence"])
            assert invoice == row["invoice_no"]
            assert inst["source_date"] == row["invoice_date"][:10]


def test_join_back_would_notice_a_permuted_weights_array(lots_csv, index):
    """回连检查必须是「按位置」比对的，不能是「按集合」。

    把某个实例的 weights 循环右移一位：按集合比对仍然相等，按位置比对必然
    不等——这条断言证明上面那条检查是位置敏感的，不是形同虚设。
    """
    table = _table(lots_csv)
    inst = R.build_real_instance(index, "train", "train", N, 0.75, 1)
    rotated = inst["weights"][1:] + inst["weights"][:1]
    by_position = all(
        w == int(table[rid]["quantity_w"]) for rid, w in zip(inst["source_row_ids"], rotated)
    )
    assert not by_position, "位置比对没有区分度，回连检查形同虚设"


# --- 容量与种子 -------------------------------------------------------------


def test_capacity_matches_the_frozen_formula(index):
    for alpha in (0.25, 0.50, 0.75):
        inst = R.build_real_instance(index, "train", "train", N, alpha, 0)
        assert inst["capacity"] == math.floor(alpha * sum(inst["weights"]))


def test_seed_matches_derivation_and_uses_the_real_token(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 9)
    assert inst["seed"] == derive_seed("train", REAL_SEED_TOKEN, N, 0.50, 9, 0, "items")


def test_item_order_comes_from_the_order_stream(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 0)
    rng = np.random.default_rng(derive_seed("train", REAL_SEED_TOKEN, N, 0.50, 0, 0, "order"))
    assert inst["item_order"] == rng.permutation(N).tolist()


def test_seed_stream_can_be_overridden_for_the_stress_size_set(index):
    """E3 真实用 test 的日期池，但种子键的 split 槽位是保留流记号。"""
    inst = R.build_real_instance(
        index, "e3_size", "test", 4, 0.25, 0, seed_stream="stream_size"
    )
    assert inst["source_date"] in index.eligible("test", 4)
    assert inst["seed"] == derive_seed("stream_size", REAL_SEED_TOKEN, 4, 0.25, 0, 0, "items")
    assert inst["instance_id"].startswith("real-e3_size-n4-")


def test_real_instances_are_deterministic(index):
    a = R.build_real_instance(index, "train", "train", N, 0.25, 4)
    b = R.build_real_instance(index, "train", "train", N, 0.25, 4)
    assert a == b


# --- 空池与重抽 -------------------------------------------------------------


def test_sampling_raises_when_no_date_is_eligible(lots_csv):
    index = L.load_index(lots_csv)
    with pytest.raises(ValueError, match="n=25"):
        R.build_real_instance(index, "train", "train", 25, 0.25, 0)


def test_fill_cell_returns_exactly_quota_and_updates_seen(index):
    seen: set[str] = set()
    insts = R.fill_cell(
        lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
        quota=5,
        seen=seen,
    )
    assert len(insts) == 5
    assert {i["canonical_hash"] for i in insts} == seen
    assert len(seen) == 5


def test_fill_cell_resamples_when_the_hash_is_already_taken(index):
    known = R.build_real_instance(index, "train", "train", N, 0.25, 0)["canonical_hash"]
    insts = R.fill_cell(
        lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
        quota=1,
        seen={known},
    )
    assert insts[0]["canonical_hash"] != known
    assert insts[0]["resample_attempt"] >= 1


def test_fill_cell_raises_instead_of_looping_forever(index):
    """把上限设为 0，任何一次碰撞都立刻触顶——宁可报错也不死循环。"""
    known = R.build_real_instance(index, "train", "train", N, 0.25, 0)["canonical_hash"]
    with pytest.raises(RuntimeError, match="resample"):
        R.fill_cell(
            lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
            quota=1,
            seen={known},
            max_attempt=0,
        )


def test_max_resample_attempt_default_is_the_frozen_constant():
    import inspect

    default = inspect.signature(R.fill_cell).parameters["max_attempt"].default
    assert default == MAX_RESAMPLE_ATTEMPT


def test_build_real_cell_matches_the_quota_and_labels_the_ids(index):
    seen: set[str] = set()
    insts = R.build_real_cell(index, "train", "train", N, 0.25, 4, seen)
    assert len(insts) == 4
    assert [i["instance_id"] for i in insts] == [
        f"real-train-n{N}-a25-i{i:04d}" for i in range(4)
    ]
    assert len({i["canonical_hash"] for i in insts}) == 4
    assert len(seen) == 4
