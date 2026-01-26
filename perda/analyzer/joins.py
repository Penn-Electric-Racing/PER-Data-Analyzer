# helper.py
from typing import Callable, Tuple

import numpy as np


def align_union_interp(
    a_ts: np.ndarray,
    a_val: np.ndarray,
    b_ts: np.ndarray,
    b_val: np.ndarray,
    *,
    drop_nan: bool = True,
    fill: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; linearly interpolate each to that grid.

    Parameters
    ----------
    a_ts : np.ndarray
        Timestamps for first series
    a_val : np.ndarray
        Values for first series
    b_ts : np.ndarray
        Timestamps for second series
    b_val : np.ndarray
        Values for second series
    drop_nan : bool, optional
        If True, rows with NaN in either series are dropped. Default is True
    fill : int, optional
        Fill value for NaNs when drop_nan is False. Default is -1

    Returns
    -------
    timestamp_match : np.ndarray
        Unified timestamp array
    a_fill : np.ndarray
        Interpolated values from first series
    b_fill : np.ndarray
        Interpolated values from second series
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

    Parameters
    ----------
    x : np.ndarray
        Query points where interpolation is needed
    xp : np.ndarray
        Known sample timestamps (will be sorted internally)
    fp : np.ndarray
        Known sample values corresponding to xp
    how : str
        Interpolation method: "prev" (forward fill) or "next" (backward fill)

    Returns
    -------
    y : np.ndarray
        Interpolated values at query points x

    Notes
    -----
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
    drop_nan: bool = True,
    fill: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the previous sample (forward fill).

    Parameters
    ----------
    a_ts : np.ndarray
        Timestamps for first series
    a_val : np.ndarray
        Values for first series
    b_ts : np.ndarray
        Timestamps for second series
    b_val : np.ndarray
        Values for second series
    drop_nan : bool, optional
        If True, rows with NaN in either series are dropped. Default is True
    fill : int, optional
        Fill value for NaNs when drop_nan is False. Default is -1

    Returns
    -------
    timestamp_match : np.ndarray
        Unified timestamp array
    a_fill : np.ndarray
        Forward-filled values from first series
    b_fill : np.ndarray
        Forward-filled values from second series

    Notes
    -----
    Uses last value where timestamp <= t. Outside each series' range produces NaN
    unless drop_nan=False, in which case NaNs are replaced by fill.
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
    drop_nan: bool = True,
    fill: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the next sample (backward fill).

    Parameters
    ----------
    a_ts : np.ndarray
        Timestamps for first series
    a_val : np.ndarray
        Values for first series
    b_ts : np.ndarray
        Timestamps for second series
    b_val : np.ndarray
        Values for second series
    drop_nan : bool, optional
        If True, rows with NaN in either series are dropped. Default is True
    fill : int, optional
        Fill value for NaNs when drop_nan is False. Default is -1

    Returns
    -------
    timestamp_match : np.ndarray
        Unified timestamp array
    a_fill : np.ndarray
        Backward-filled values from first series
    b_fill : np.ndarray
        Backward-filled values from second series

    Notes
    -----
    Uses first value where timestamp >= t. Outside each series' range produces NaN
    unless drop_nan=False, in which case NaNs are replaced by fill.
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
    align: Callable = align_union_interp,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic combiner for binary operations on two time series.

    Parameters
    ----------
    a_ts : np.ndarray
        Timestamps for first series
    a_val : np.ndarray
        Values for first series
    b_ts : np.ndarray
        Timestamps for second series
    b_val : np.ndarray
        Values for second series
    ufunc : Callable
        NumPy universal function to apply (e.g., np.add, np.subtract)
    align : Callable, optional
        Alignment function to use. Default is align_union_interp

    Returns
    -------
    T : np.ndarray
        Unified timestamp array
    result : np.ndarray
        Result of applying ufunc to aligned values
    """
    T, av, bv = align(a_ts, a_val, b_ts, b_val)
    return T, ufunc(av, bv)


def name_matches(short_name: str, full_name: str) -> bool:
    """
    Check if a short name matches a full CAN variable name.

    Parameters
    ----------
    short_name : str
        Short variable name to search for
    full_name : str
        Full CAN variable name (may include description and parentheses)

    Returns
    -------
    bool
        True if short_name is found in full_name (with or without parentheses)
    """
    return f"({short_name})" in full_name or f"{short_name}" in full_name
