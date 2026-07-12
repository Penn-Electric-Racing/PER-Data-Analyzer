from enum import Enum

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
