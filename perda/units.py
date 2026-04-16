from enum import Enum

import numpy.typing as npt


class Timescale(Enum):
    """Supported timestamp units for time-series data."""

    US = "us"
    MS = "ms"
    S = "s"


def _to_seconds(
    timestamp: float | npt.NDArray, source_time_unit: Timescale
) -> float | npt.NDArray:
    """Convert a timestamp from the given unit to seconds.

    Parameters
    ----------
    timestamp : float | NDArray
        Timestamp(s) to convert.
    source_time_unit : Timescale
        Unit of the input timestamp.

    Returns
    -------
    float | NDArray
        Timestamp(s) in seconds.
    """
    if source_time_unit == Timescale.US:
        return timestamp / 1e6
    if source_time_unit == Timescale.MS:
        return timestamp / 1e3
    return timestamp


def _from_seconds(
    timestamp_s: float | npt.NDArray, target_time_unit: Timescale
) -> float | npt.NDArray:
    """Convert a timestamp from seconds to the given unit.

    Parameters
    ----------
    timestamp_s : float | NDArray
        Timestamp(s) in seconds.
    target_time_unit : Timescale
        Desired output unit.

    Returns
    -------
    float | NDArray
        Timestamp(s) in the target unit.
    """
    if target_time_unit == Timescale.US:
        return timestamp_s * 1e6
    if target_time_unit == Timescale.MS:
        return timestamp_s * 1e3
    return timestamp_s


def convert_time(
    timestamp: float | npt.NDArray,
    source_time_unit: Timescale,
    target_time_unit: Timescale,
) -> float | npt.NDArray:
    """Convert a timestamp between two timescale units.

    Parameters
    ----------
    timestamp : float | NDArray
        Timestamp(s) to convert.
    source_time_unit : Timescale
        Unit of the input timestamp.
    target_time_unit : Timescale
        Desired output unit.

    Returns
    -------
    float | NDArray
        Timestamp(s) in the target unit.

    Examples
    --------
    >>> convert_time(5000.0, Timescale.MS, Timescale.S)
    5.0
    """
    return _from_seconds(_to_seconds(timestamp, source_time_unit), target_time_unit)


def mph_to_m_per_s(value: float | npt.NDArray) -> float | npt.NDArray:
    """Convert a speed value from miles per hour to meters per second.

    Parameters
    ----------
    value : float | NDArray
        Speed in mph.

    Returns
    -------
    float | NDArray
        Speed in m/s.

    Examples
    --------
    >>> mph_to_m_per_s(1.0)
    0.44704
    """
    return value / 3600 * 1609.34


def in_to_m(value: float | npt.NDArray) -> float | npt.NDArray:
    """Convert a length value from inches to meters.

    Parameters
    ----------
    value : float | NDArray
        Length in inches.

    Returns
    -------
    float | NDArray
        Length in meters.

    Examples
    --------
    >>> in_to_m(1.0)
    0.0254
    """
    return value * 0.0254


# Consistency factor to convert MAD to an equivalent standard deviation estimate
# under the assumption of normally distributed data: sigma ≈ MAD / 0.6745
MAD_TO_STD = 1.4826
