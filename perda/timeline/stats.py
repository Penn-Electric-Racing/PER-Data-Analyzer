from __future__ import annotations

import polars as pl

from ..units import US_PER_SECOND
from .sentinels import is_sentinel, sentinel_bits

# Column order must match the COPY that writes session_var_stats.
STATS_COLUMNS = [
    "var_id",
    "n",
    "n_invalid",
    "sentinel_bits",
    "t_first_us",
    "t_last_us",
    "v_min",
    "v_max",
    "v_mean",
    "v_first",
    "v_last",
    "v_p01",
    "v_p50",
    "v_p95",
    "v_p99",
    "n_changes",
    "n_rising",
    "hz",
    "max_gap_us",
]


def compute_session_stats(frame: pl.DataFrame) -> pl.DataFrame:
    """Summarise one session's samples into a per-variable row.

    Distributional statistics cover *valid* samples only, with sentinels
    counted separately -- a variable that is entirely sentinel would otherwise
    report a plausible-looking minimum near zero.

    Parameters
    ----------
    frame : pl.DataFrame
        Columns ``var_id``, ``t_us`` (absolute) and ``value``, sorted by
        ``(var_id, t_us)``.

    Returns
    -------
    pl.DataFrame
        ``STATS_COLUMNS`` plus ``dtype`` (``bool``/``int``/``float``/``unknown``).
    """
    valid = ~is_sentinel(pl.col("value"))
    good = pl.col("value").filter(valid)

    aggregated = frame.group_by("var_id").agg(
        pl.len().alias("n"),
        valid.not_().sum().alias("n_invalid"),
        sentinel_bits(pl.col("value").filter(valid.not_())).first().alias("sentinel_bits"),
        pl.col("t_us").min().alias("t_first_us"),
        pl.col("t_us").max().alias("t_last_us"),
        good.min().alias("v_min"),
        good.max().alias("v_max"),
        good.mean().alias("v_mean"),
        good.first().alias("v_first"),
        good.last().alias("v_last"),
        good.quantile(0.01).alias("v_p01"),
        good.quantile(0.50).alias("v_p50"),
        good.quantile(0.95).alias("v_p95"),
        good.quantile(0.99).alias("v_p99"),
        (pl.col("value").diff() != 0).sum().alias("n_changes"),
        ((pl.col("value").shift() == 0) & (pl.col("value") != 0)).sum().alias("n_rising"),
        pl.col("t_us").diff().max().alias("max_gap_us"),
        (good == good.round(0)).all().alias("_integral"),
        good.len().alias("_n_valid"),
    )

    span_us = pl.col("t_last_us") - pl.col("t_first_us")
    return aggregated.with_columns(
        pl.when(span_us > 0)
        .then((pl.col("n") - 1) * US_PER_SECOND / span_us.cast(pl.Float64))
        .alias("hz"),
        pl.when(pl.col("_n_valid") == 0)
        .then(pl.lit("unknown"))
        .when(pl.col("_integral") & (pl.col("v_min") >= 0) & (pl.col("v_max") <= 1))
        .then(pl.lit("bool"))
        .when(pl.col("_integral"))
        .then(pl.lit("int"))
        .otherwise(pl.lit("float"))
        .alias("dtype"),
    ).select([*STATS_COLUMNS, "dtype"])
