from __future__ import annotations

import polars as pl

FLOAT32_MIN_NORMAL = 1.17549435e-38
SMALLEST_FLOAT32_DENORMAL = 2.0**-149


def is_sentinel(value: pl.Expr) -> pl.Expr:
    """
    Flag samples that carry an invalid-reading sentinel rather than a measurement.

    Parameters
    ----------
    value : pl.Expr
        Expression selecting the value column.

    Returns
    -------
    pl.Expr
        Boolean expression, true where the sample is a sentinel.

    Notes
    -----
    Every logged value occupies a float32 field. When firmware writes a raw
    integer instead of a measurement the bit pattern decodes as a denormal, so
    ``pcm.pedals.accel`` emits ``2.8026e-45`` (bit pattern ``2``) to mean the
    pedal reading is invalid. Such values look like zero to a naive aggregate.
    """
    magnitude = value.abs()
    return (magnitude > 0.0) & (magnitude < FLOAT32_MIN_NORMAL)


def sentinel_bits(value: pl.Expr) -> pl.Expr:
    """
    Recover the integer bit pattern behind a denormal sentinel.

    Parameters
    ----------
    value : pl.Expr
        Expression selecting sentinel values only.

    Returns
    -------
    pl.Expr
        The underlying bit pattern as an integer expression.

    Notes
    -----
    The input must already be filtered to sentinels; dividing a normal float by
    the smallest denormal overflows int64.
    """
    return (value.abs() / SMALLEST_FLOAT32_DENORMAL).round(0).cast(pl.Int64, strict=False)
