from collections import defaultdict
from typing import Tuple

import numpy as np


def left_join(
    left_ts: np.ndarray,
    left_val: np.ndarray,
    right_ts: np.ndarray,
    right_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Left join: keep all left timestamps, match and interpolate right values.

    Parameters
    ----------
    left_ts : np.ndarray
        Timestamps for left series (these are kept exactly)
    left_val : np.ndarray
        Values for left series
    right_ts : np.ndarray
        Timestamps for right series (these will be matched to left)
    right_val : np.ndarray
        Values for right series
    interpolate : bool, optional
        If True, interpolate right values to fill all left timestamps.
        If False, only use matched values (NaN for unmatched). Default is True.

    Returns
    -------
    timestamps : np.ndarray
        The left timestamps (unchanged)
    left_values : np.ndarray
        The left values (unchanged)
    right_values : np.ndarray
        Right values matched/interpolated to left timestamps

    Notes
    -----
    Process:
    1. For each right timestamp, find the closest left timestamp
    2. If multiple right timestamps map to the same left timestamp, average them
    3. Interpolate right values to fill remaining left timestamps (if interpolate=True)
    """
    timestamps = left_ts.copy()
    left_values = left_val.copy()

    right_values = np.full(left_ts.shape, np.nan, dtype=float)

    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    # Find insertion indices
    idx = np.searchsorted(left_ts, right_ts)

    # Clamp indices to valid range
    idx_right = np.clip(idx, 0, len(left_ts) - 1)
    idx_left = np.clip(idx - 1, 0, len(left_ts) - 1)

    # Choose closer neighbor
    dist_left = np.abs(right_ts - left_ts[idx_left])
    dist_right = np.abs(right_ts - left_ts[idx_right])
    closest_idx = np.where(dist_left <= dist_right, idx_left, idx_right)

    # Average right values mapped to same left index
    matches = defaultdict(list)
    for li, rv in zip(closest_idx, right_val):
        matches[li].append(rv)

    for li, vals in matches.items():
        right_values[li] = np.mean(vals)

    # Interpolate missing values
    valid = ~np.isnan(right_values)
    right_values = np.interp(left_ts, left_ts[valid], right_values[valid])

    return timestamps, left_values, right_values


def outer_join(
    left_ts: np.ndarray,
    left_val: np.ndarray,
    right_ts: np.ndarray,
    right_val: np.ndarray,
    *,
    drop_nan: bool = True,
    fill: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Outer join: union of timestamps with linear interpolation.

    Parameters
    ----------
    left_ts : np.ndarray
        Timestamps for left series
    left_val : np.ndarray
        Values for left series
    right_ts : np.ndarray
        Timestamps for right series
    right_val : np.ndarray
        Values for right series
    drop_nan : bool, optional
        If True, drop rows where either series has NaN after interpolation.
        Default is True.
    fill : float, optional
        Fill value for NaNs when drop_nan is False. Default is 0.0.

    Returns
    -------
    timestamps : np.ndarray
        Union of all timestamps
    left_values : np.ndarray
        Left values interpolated to union timestamps
    right_values : np.ndarray
        Right values interpolated to union timestamps
    """
    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    timestamps = np.union1d(left_ts, right_ts)
    left_values = np.interp(timestamps, left_ts, left_val)
    right_values = np.interp(timestamps, right_ts, right_val)

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
    left_ts: np.ndarray,
    left_val: np.ndarray,
    right_ts: np.ndarray,
    right_val: np.ndarray,
    *,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inner join: keep only left timestamps that have a matching right timestamp within tolerance.

    Process:
    1. For each left timestamp, find the closest right timestamp
    2. Keep the left timestamp only if the distance is within tolerance
    3. Match right values to the kept left timestamps

    Parameters
    ----------
    left_ts : np.ndarray
        Timestamps for left series
    left_val : np.ndarray
        Values for left series
    right_ts : np.ndarray
        Timestamps for right series
    right_val : np.ndarray
        Values for right series
    tolerance : float
        Maximum allowed distance between left and right timestamps for a match.
        Timestamps with distance > tolerance are dropped.

    Returns
    -------
    timestamps : np.ndarray
        Subset of left timestamps that have matches within tolerance
    left_values : np.ndarray
        Left values at the matched timestamps
    right_values : np.ndarray
        Right values interpolated to the matched timestamps
    """
    if right_ts.size == 0 or left_ts.size == 0:
        raise ValueError("Both time series must be non-empty")

    # Output arrays
    timestamps = left_ts.copy()
    left_values = left_val.copy()
    right_values = np.full(left_ts.shape, np.nan, dtype=float)

    # Find insertion points of left_ts into right_ts
    idx = np.searchsorted(right_ts, left_ts)

    # Candidate neighbors
    idx_right = np.clip(idx, 0, len(right_ts) - 1)
    idx_left = np.clip(idx - 1, 0, len(right_ts) - 1)

    # Distances
    dist_left = np.abs(left_ts - right_ts[idx_left])
    dist_right = np.abs(left_ts - right_ts[idx_right])

    # Choose closest right index
    closest_right_idx = np.where(dist_left <= dist_right, idx_left, idx_right)

    # Distance to chosen right timestamp
    min_dist = np.minimum(dist_left, dist_right)

    # Apply tolerance filter
    valid_mask = min_dist <= tolerance
    valid_left_idx = np.where(valid_mask)[0]
    valid_right_idx = closest_right_idx[valid_mask]

    # Deduplicate: multiple right values may map to same left index
    matches = defaultdict(list)
    for li, ri in zip(valid_left_idx, valid_right_idx):
        matches[li].append(right_val[ri])

    for li, vals in matches.items():
        right_values[li] = np.mean(vals)

    # Filter to only include matched timestamps
    matched_indices = np.array(sorted(matches.keys()))
    timestamps = timestamps[matched_indices]
    left_values = left_values[matched_indices]
    right_values = right_values[matched_indices]

    return timestamps, left_values, right_values


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
