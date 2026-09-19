import polars as pl
import pytest

from perda.timeline.sentinels import is_sentinel, sentinel_bits


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.802596928649634e-45, True),
        (4.5914945482066956e-41, True),
        (0.0, False),
        (1.0, False),
        (-3.5, False),
        (1e-30, False),
        (1e-39, True),
        (1e-37, False),
    ],
)
def test_is_sentinel(value, expected):
    frame = pl.DataFrame({"value": [value]})
    assert frame.select(is_sentinel(pl.col("value")))["value"][0] is expected


@pytest.mark.parametrize(
    "value, bits",
    [(2.802596928649634e-45, 2), (1.401298464324817e-45, 1), (4.5914945482066956e-41, 32766)],
)
def test_sentinel_bits_recovers_integer(value, bits):
    frame = pl.DataFrame({"value": [value]})
    assert frame.select(sentinel_bits(pl.col("value")))["value"][0] == bits
