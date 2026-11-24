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


INTERP1D_KIND = {
    ResampleMethod.ZOH: "previous",
    ResampleMethod.NEAREST: "nearest",
    ResampleMethod.CUBIC: "cubic",
}
CUBIC_MIN_POINTS = 4


def interpolate_onto_timestamps(
    target_timestamps: NDArray[np.float64],
    source_timestamps: NDArray[np.float64],
    source_values: NDArray[np.float64],
    method: ResampleMethod,
) -> NDArray[np.float64]:
    """
    Interpolate values sampled at ``source_timestamps`` onto ``target_timestamps``.

    Parameters
    ----------
    target_timestamps : NDArray[np.float64]
        Timestamps to produce values at.
    source_timestamps : NDArray[np.float64]
        Timestamps the source values were sampled at.
    source_values : NDArray[np.float64]
        Source values, same length as ``source_timestamps``.
    method : ResampleMethod
        Interpolation strategy.

    Returns
    -------
    NDArray[np.float64]
        Values at ``target_timestamps``. Targets outside the source range take the nearest end value.

    Notes
    -----
    Cubic interpolation falls back to linear when there are fewer than 4 source points.
    """
    if method == ResampleMethod.LINEAR or (
        method == ResampleMethod.CUBIC and len(source_timestamps) < CUBIC_MIN_POINTS
    ):
        return np.interp(target_timestamps, source_timestamps, source_values)
    if method not in INTERP1D_KIND:
        raise ValueError(f"Unknown resample method '{method}'")
    interpolator = interp1d(
        source_timestamps,
        source_values,
        kind=INTERP1D_KIND[method],
        bounds_error=False,
        fill_value=(source_values[0], source_values[-1]),
    )
    return interpolator(target_timestamps)
