import hashlib

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
    """0.1 + 0.2 的 repr 是 0.30000000000000004，必须与 0.3 归一到同一种子。

    不能用 0.5 与 0.50 来测：二者是同一个 float 字面量，断言恒真、什么都不验证。
    这一对输入还顺带守住「derive_seed 的键确实经过 alpha_key」这层接线——若键
    退化成 str(alpha)，两个键会不同，本测试即红。
    """
    noisy = derive_seed(**{**BASE, "alpha": 0.1 + 0.2})
    exact = derive_seed(**{**BASE, "alpha": 0.3})
    assert noisy == exact


def test_seed_is_independent_of_call_order():
    """先算别的组合再算目标组合，结果必须一致。"""
    first = derive_seed(**BASE)
    for other in (
        dict(split="test", family="skewed", n=200, alpha=0.75, index=99),
        dict(split="validation", family="normal", n=100, alpha=0.50, index=7),
    ):
        derive_seed(**other)
    assert derive_seed(**BASE) == first


def test_golden_seed_pins_key_format():
    """硬编码期望值，钉住键的分隔符与字段顺序。

    期望值由规格公式**内联**算出，而不是调用 derive_seed——否则就是拿实现给实现
    自己当标准。把分隔符从 "|" 换成 "," 或调换字段顺序时，上面那些关系型断言会
    全绿，而 8400 个实例的种子会全部改变。用 0.50 而非 0.25，可顺带钉住 alpha 是
    按 alpha_key 渲染的：若退化成 str(alpha)，键里会变成 0.5，本测试即红。
    """
    expected = int.from_bytes(
        hashlib.blake2b(
            "train|uniform|20|0.50|0|0|items".encode("utf-8"), digest_size=8
        ).digest(),
        "big",
    )
    assert derive_seed("train", "uniform", 20, 0.50, 0) == expected
    # 字面值让「重算一遍贴回去」变成一次显式的重新冻结，而不是顺手修绿。
    assert expected == 11113391727366433376
