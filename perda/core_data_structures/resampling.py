from enum import Enum
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class ResampleMethod(str, Enum):
    """Interpolation strategy used when aligning a series to a target timestamp grid."""

    LINEAR = "linear"
    ZOH = "zoh"
    NEAREST = "nearest"
    CUBIC = "cubic"


def _interpolate(
    target: NDArray,
    src_t: NDArray,
    src_v: NDArray,
    method: ResampleMethod,
) -> NDArray:
    """
    Interpolate src_v (sampled at src_t) onto target timestamps.

    Parameters
    ----------
    target : NDArray
        Target timestamps (float64)
    src_t : NDArray
        Source timestamps (float64)
    src_v : NDArray
        Source values
    method : ResampleMethod
        Interpolation strategy

    Returns
    -------
    NDArray
        Interpolated values at target timestamps
    """
    if method == ResampleMethod.LINEAR:
        return np.interp(target, src_t, src_v)
    elif method == ResampleMethod.ZOH:
        f = interp1d(
            src_t,
            src_v,
            kind="previous",
            bounds_error=False,
            fill_value=(src_v[0], src_v[-1]),
        )
        return f(target)
    elif method == ResampleMethod.NEAREST:
        f = interp1d(
            src_t,
            src_v,
            kind="nearest",
            bounds_error=False,
            fill_value=(src_v[0], src_v[-1]),
        )
        return f(target)
    elif method == ResampleMethod.CUBIC:
        if len(src_t) < 4:
            return np.interp(target, src_t, src_v)
        f = interp1d(
            src_t,
            src_v,
            kind="cubic",
            bounds_error=False,
            fill_value=(src_v[0], src_v[-1]),
        )
        return f(target)
    else:
        raise ValueError(f"Unknown resample method '{method}'")


def resample_to_freq(
    ts: NDArray,
    val: NDArray,
    freq_hz: float,
    timestamp_divisor: float,
    method: ResampleMethod = ResampleMethod.LINEAR,
) -> Tuple[NDArray, NDArray]:
    """
    Resample a time series onto a uniform frequency grid.

    Parameters
    ----------
    ts : NDArray
        Source timestamps (int64)
    val : NDArray
        Source values
    freq_hz : float
        Target sampling frequency in Hz
    timestamp_divisor : float
        Raw timestamp units per second (e.g. 1e6 for microseconds)
    method : ResampleMethod, optional
        Interpolation method. Default is LINEAR.

    Returns
    -------
    target_ts : NDArray
        Uniform timestamp grid (int64)
    resampled_val : NDArray
        Values interpolated onto the uniform grid
    """
    dt = timestamp_divisor / freq_hz
    target_ts = np.arange(ts[0], ts[-1], dt, dtype=np.float64).astype(np.int64)
    resampled_val = _interpolate(
        target_ts.astype(np.float64), ts.astype(np.float64), val, method
    )
    return target_ts, resampled_val
