# helper.py
from typing import Callable, Tuple

import numpy as np
from datainstance import DataInstance
from numpy.typing import NDArray


def _new(T: NDArray[np.integer], V: NDArray[np.number]) -> DataInstance:
    return DataInstance(T, np.asarray(V))


# --- alignment policies ---


def align_union_interp(a: DataInstance, b: DataInstance, *, drop_nan=True, fill=-1):
    """
    Union of timestamps; linearly interpolate each to that grid.
    If drop_nan=True, rows with NaN in either series are dropped.
    Otherwise NaNs are filled with `fill`.
    """
    timestamp_match = np.union1d(a.timestamp_np, b.timestamp_np)
    a_fill = np.interp(timestamp_match, a.timestamp_np, a.val_np)
    b_fill = np.interp(timestamp_match, b.timestamp_np, b.val_np)
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
    a, b, *, drop_nan: bool = True, fill: float = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the **previous** sample
    (last value where timestamp <= t). Outside each series' range -> NaN
    unless drop_nan=False, in which case NaNs are replaced by `fill`.
    """
    timestamp_match = np.union1d(a.timestamp_np, b.timestamp_np)
    a_fill = _step_fill(timestamp_match, a.timestamp_np, a.val_np, how="prev")
    b_fill = _step_fill(timestamp_match, b.timestamp_np, b.val_np, how="prev")

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
    a, b, *, drop_nan: bool = True, fill: float = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Union of timestamps; fill each series with the **next** sample
    (first value where timestamp >= t). Outside each series' range -> NaN
    unless drop_nan=False, in which case NaNs are replaced by `fill`.
    """
    timestamp_match = np.union1d(a.timestamp_np, b.timestamp_np)
    a_fill = _step_fill(timestamp_match, a.timestamp_np, a.val_np, how="next")
    b_fill = _step_fill(timestamp_match, b.timestamp_np, b.val_np, how="next")

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


def _combine(
    a: DataInstance, b: DataInstance, ufunc: Callable, align=align_union_interp
) -> DataInstance:
    T, av, bv = align(a, b)
    return _new(T, ufunc(av, bv))


# --- public API ---
def add(x, y, align=align_union_interp):
    if isinstance(x, DataInstance) and isinstance(y, DataInstance):
        return _combine(x, y, np.add, align)
    if isinstance(x, DataInstance) and np.isscalar(y):
        return _new(x.timestamp_np, np.add(x.val_np, y))
    if np.isscalar(x) and isinstance(y, DataInstance):
        return _new(y.timestamp_np, np.add(x, y.val_np))
    raise TypeError("add expects (DataInstance, DataInstance|scalar) in any order")


def sub(x, y, align=align_union_interp):
    if isinstance(x, DataInstance) and isinstance(y, DataInstance):
        return _combine(x, y, np.subtract, align)
    if isinstance(x, DataInstance) and np.isscalar(y):
        return _new(x.timestamp_np, np.subtract(x.val_np, y))
    if np.isscalar(x) and isinstance(y, DataInstance):
        return _new(y.timestamp_np, np.subtract(x, y.val_np))
    raise TypeError("sub expects (DataInstance, DataInstance|scalar)")


def mul(x, y, align=align_union_interp):
    if isinstance(x, DataInstance) and isinstance(y, DataInstance):
        return _combine(x, y, np.multiply, align)
    if isinstance(x, DataInstance) and np.isscalar(y):
        return _new(x.timestamp_np, np.multiply(x.val_np, y))
    if np.isscalar(x) and isinstance(y, DataInstance):
        return _new(y.timestamp_np, np.multiply(x, y.val_np))
    raise TypeError("mul expects (DataInstance, DataInstance|scalar)")


def div(x, y, align=align_union_interp):
    if isinstance(x, DataInstance) and isinstance(y, DataInstance):
        return _combine(x, y, np.true_divide, align)
    if isinstance(x, DataInstance) and np.isscalar(y):
        return _new(x.timestamp_np, np.true_divide(x.val_np, y))
    if np.isscalar(x) and isinstance(y, DataInstance):
        return _new(y.timestamp_np, np.true_divide(x, y.val_np))
    raise TypeError("div expects (DataInstance, DataInstance|scalar)")


def pow_(x, y, align=align_union_interp):
    if isinstance(x, DataInstance) and isinstance(y, DataInstance):
        return _combine(x, y, np.power, align)
    if isinstance(x, DataInstance) and np.isscalar(y):
        return _new(x.timestamp_np, np.power(x.val_np, y))
    if np.isscalar(x) and isinstance(y, DataInstance):
        return _new(y.timestamp_np, np.power(x, y.val_np))
    raise TypeError("pow_ expects (DataInstance, DataInstance|scalar)")
