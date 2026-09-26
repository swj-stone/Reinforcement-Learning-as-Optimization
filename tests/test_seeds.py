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
