# helper.py
from typing import Callable, Tuple

import numpy as np


def align_union_interp(
    a_ts: np.ndarray,
    a_val: np.ndarray,
    b_ts: np.ndarray,
    b_val: np.ndarray,
    *,
    drop_nan=True,
    fill=-1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; linearly interpolate each to that grid.
    If drop_nan=True, rows with NaN in either series are dropped.
    Otherwise NaNs are filled with `fill`.
    """
    timestamp_match = np.union1d(a_ts, b_ts)
    a_fill = np.interp(timestamp_match, a_ts, a_val)
    b_fill = np.interp(timestamp_match, b_ts, b_val)
    if drop_nan:
        keep_idx = ~np.isnan(a_fill) & ~np.isnan(b_fill)
        timestamp_match, a_fill, b_fill = (
            timestamp_match[keep_idx],
            a_fill[keep_idx],
            b_fill[keep_idx],
        )
    else:
        a_fill = np.nan_to_num(a_fill, nan=float(fill))
        b_fill = np.nan_to_num(b_fill, nan=float(fill))
    return timestamp_match, a_fill, b_fill


def _step_fill(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, *, how: str
) -> np.ndarray:
    """
    Step "interpolation" at points x using samples (xp, fp).
    - how="prev": take last fp where xp <= x (NaN if none)
    - how="next": take first fp where xp >= x (NaN if none)
    Assumes xp is 1D; sorts (xp, fp) just in case.
    """
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    x = np.asarray(x)

    # sort by xp to satisfy searchsorted precondition
    order = np.argsort(xp, kind="mergesort")
    xp = xp[order]
    fp = fp[order]

    y = np.full(x.shape, np.nan, dtype=float)

    if how == "prev":
        idx = np.searchsorted(xp, x, side="right") - 1  # last xp <= x
        valid = idx >= 0
        if np.any(valid):
            y[valid] = fp[idx[valid]]
    elif how == "next":
        idx = np.searchsorted(xp, x, side="left")  # first xp >= x
        valid = idx < len(xp)
        if np.any(valid):
            y[valid] = fp[idx[valid]]
    else:
        raise ValueError("how must be 'prev' or 'next'")

    return y


def align_union_prev(
    a_ts: np.ndarray,
    a_val: np.ndarray,
    b_ts: np.ndarray,
    b_val: np.ndarray,
    *,
    drop_nan=True,
    fill=-1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the **previous** sample
    (last value where timestamp <= t). Outside each series' range -> NaN
    unless drop_nan=False, in which case NaNs are replaced by `fill`.
    """
    timestamp_match = np.union1d(a_ts, b_ts)
    a_fill = _step_fill(timestamp_match, a_ts, a_val, how="prev")
    b_fill = _step_fill(timestamp_match, b_ts, b_val, how="prev")

    if drop_nan:
        keep_idx = ~np.isnan(a_fill) & ~np.isnan(b_fill)
        timestamp_match, a_fill, b_fill = (
            timestamp_match[keep_idx],
            a_fill[keep_idx],
            b_fill[keep_idx],
        )
    else:
        a_fill = np.nan_to_num(a_fill, nan=float(fill))
        b_fill = np.nan_to_num(b_fill, nan=float(fill))

    return timestamp_match, a_fill, b_fill


def align_union_next(
    a_ts: np.ndarray,
    a_val: np.ndarray,
    b_ts: np.ndarray,
    b_val: np.ndarray,
    *,
    drop_nan=True,
    fill=-1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the **next** sample
    (first value where timestamp >= t). Outside each series' range -> NaN
    unless drop_nan=False, in which case NaNs are replaced by `fill`.
    """
    timestamp_match = np.union1d(a_ts, b_ts)
    a_fill = _step_fill(timestamp_match, a_ts, a_val, how="next")
    b_fill = _step_fill(timestamp_match, b_ts, b_val, how="next")

    if drop_nan:
        keep_idx = ~np.isnan(a_fill) & ~np.isnan(b_fill)
        timestamp_match, a_fill, b_fill = (
            timestamp_match[keep_idx],
            a_fill[keep_idx],
            b_fill[keep_idx],
        )
    else:
        a_fill = np.nan_to_num(a_fill, nan=float(fill))
        b_fill = np.nan_to_num(b_fill, nan=float(fill))

    return timestamp_match, a_fill, b_fill


# --- generic combiner for binary ops ---
def combine(
    a_ts: np.ndarray,
    a_val: np.ndarray,
    b_ts: np.ndarray,
    b_val: np.ndarray,
    ufunc: Callable,
    align=align_union_interp,
) -> Tuple[np.ndarray, np.ndarray]:
    T, av, bv = align(a_ts, a_val, b_ts, b_val)
    return T, ufunc(av, bv)
