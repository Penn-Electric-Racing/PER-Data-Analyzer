from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .resampling import ResampleMethod, _interpolate


def left_join(
    left_ts: NDArray,
    left_val: NDArray,
    right_ts: NDArray,
    right_val: NDArray,
    *,
    method: ResampleMethod = ResampleMethod.LINEAR,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Left join: interpolate right values onto the left timestamp grid.

    Parameters
    ----------
    left_ts : NDArray
        Timestamps for left series (used as the target grid)
    left_val : NDArray
        Values for left series
    right_ts : NDArray
        Timestamps for right series
    right_val : NDArray
        Values for right series
    method : ResampleMethod, optional
        Interpolation method. Default is LINEAR.

    Returns
    -------
    timestamps : NDArray
        Left timestamps (unchanged)
    left_values : NDArray
        Left values (unchanged)
    right_values : NDArray
        Right values interpolated onto left timestamps
    """
    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    target_f = left_ts.astype(np.float64)
    right_values = _interpolate(
        target_f, right_ts.astype(np.float64), right_val, method
    )

    return left_ts.copy(), left_val.copy(), right_values


def outer_join(
    left_ts: NDArray,
    left_val: NDArray,
    right_ts: NDArray,
    right_val: NDArray,
    *,
    method: ResampleMethod = ResampleMethod.LINEAR,
    drop_nan: bool = True,
    fill: float = 0.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Outer join: union of timestamps with interpolation.

    Parameters
    ----------
    left_ts : NDArray
        Timestamps for left series
    left_val : NDArray
        Values for left series
    right_ts : NDArray
        Timestamps for right series
    right_val : NDArray
        Values for right series
    method : ResampleMethod, optional
        Interpolation method. Default is LINEAR.
    drop_nan : bool, optional
        If True, drop rows where either series has NaN after interpolation.
        Default is True.
    fill : float, optional
        Fill value for NaNs when drop_nan is False. Default is 0.0.

    Returns
    -------
    timestamps : NDArray
        Union of all timestamps
    left_values : NDArray
        Left values interpolated to union timestamps
    right_values : NDArray
        Right values interpolated to union timestamps
    """
    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    timestamps = np.union1d(left_ts, right_ts)
    target_f = timestamps.astype(np.float64)
    left_values = _interpolate(target_f, left_ts.astype(np.float64), left_val, method)
    right_values = _interpolate(
        target_f, right_ts.astype(np.float64), right_val, method
    )

    if drop_nan:
        keep_mask = ~np.isnan(left_values) & ~np.isnan(right_values)
        timestamps = timestamps[keep_mask]
        left_values = left_values[keep_mask]
        right_values = right_values[keep_mask]
    else:
        left_values = np.nan_to_num(left_values, nan=fill)
        right_values = np.nan_to_num(right_values, nan=fill)

    return timestamps, left_values, right_values


def inner_join(
    left_ts: NDArray,
    left_val: NDArray,
    right_ts: NDArray,
    right_val: NDArray,
    *,
    tolerance: float,
    method: ResampleMethod = ResampleMethod.LINEAR,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Inner join: keep only left timestamps that have a right timestamp within tolerance,
    then interpolate right values onto those timestamps.

    Parameters
    ----------
    left_ts : NDArray
        Timestamps for left series
    left_val : NDArray
        Values for left series
    right_ts : NDArray
        Timestamps for right series
    right_val : NDArray
        Values for right series
    tolerance : float
        Maximum allowed distance to the nearest right timestamp for a left
        timestamp to be kept.
    method : ResampleMethod, optional
        Interpolation method for right values. Default is LINEAR.

    Returns
    -------
    timestamps : NDArray
        Subset of left timestamps within tolerance of a right timestamp
    left_values : NDArray
        Left values at the kept timestamps
    right_values : NDArray
        Right values interpolated onto the kept timestamps
    """
    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    # For each left timestamp, find the distance to the nearest right timestamp
    idx = np.searchsorted(right_ts, left_ts)
    idx_hi = np.clip(idx, 0, len(right_ts) - 1)
    idx_lo = np.clip(idx - 1, 0, len(right_ts) - 1)
    min_dist = np.minimum(
        np.abs(left_ts - right_ts[idx_lo]), np.abs(left_ts - right_ts[idx_hi])
    )

    keep = min_dist <= tolerance
    timestamps = left_ts[keep]
    left_values = left_val[keep]
    right_values = _interpolate(
        timestamps.astype(np.float64), right_ts.astype(np.float64), right_val, method
    )

    return timestamps, left_values, right_values
