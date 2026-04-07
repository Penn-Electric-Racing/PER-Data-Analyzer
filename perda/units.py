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


def mph_seconds_to_meters(value: float | npt.NDArray) -> float | npt.NDArray:
    """Convert a value in mph·s (e.g. integrated speed) to meters.

    Parameters
    ----------
    value : float | NDArray
        Value in mph·s, such as the result of integrating a speed signal
        in mph over a time axis in seconds.

    Returns
    -------
    float | NDArray
        Equivalent distance in meters.

    Examples
    --------
    >>> mph_seconds_to_meters(3600.0)
    1609.34
    """
    return value / 3600 * 1609.34


# Consistency factor to convert MAD to an equivalent standard deviation estimate
# under the assumption of normally distributed data: sigma ≈ MAD / 0.6745
MAD_TO_STD = 1.4826
