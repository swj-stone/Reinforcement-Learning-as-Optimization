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


def test_hash_golden_pins_the_payload_format():
    """把哈希字面值写死，钉住设计文档 7.1 节的载荷格式。

    上面 8 条断言全是关系型的（等于/不等于），对**载荷格式**完全不敏感：
    把 f"{n}|{capacity}|{body}" 改成 f"{capacity}|{n}|{body}"、把分隔符换成
    逗号、给 body 加个尾随空格、甚至把排序改成降序，8 条照旧全绿——而 manifest
    每条记录的 canonical_hash 会整体改变（设计文档 201 行），已生成的语料再也
    无法逐字节复现。设计文档 7.1 节把公式写得完全确定，故此处必须钉字面值。
    期望值按 7.1 节公式**独立**算出，不是把实现跑一遍抄下来。
    """
    assert canonical_hash(4, 5, W, V) == (
        "3b5d21a8a635dca0f8d153f7d83993988f2a4d931df48e08b4b93a71fca51da1"
    )


def test_reordering_is_order_insensitive_when_values_tie():
    """两条 v 相同、w 不同的对，必须按 (w, v) 全序排，而不是只按 v 排。

    test_reordering_items_does_not_change_the_hash 的数据里有一对 w 相同
    （w=1 出现两次，v 分别是 20 与 40），所以只按 w 排序会被它抓住；但整组测试
    没有任何 v 相同的对，只按 v 排序（``key=lambda t: t[1]``）会全绿。v 相同时
    的相对次序会退回输入顺序，物品重排的副本于是得到不同哈希——正是 7.1 节要
    消除的情形。字面值同样按 7.1 节公式独立算出，顺带钉住并列时的排序方向。
    """
    w = np.array([3, 1], dtype=np.int64)
    v = np.array([5, 5], dtype=np.int64)
    perm = np.array([1, 0])
    assert canonical_hash(2, 7, w[perm], v[perm]) == canonical_hash(2, 7, w, v)
    assert canonical_hash(2, 7, w, v) == (
        "1ba458e0cf0ed3a0eb3328f9f63022d3596c782b45debdafca663d996067fd98"
    )
