from enum import Enum
from typing import TypeVar, overload

from numpy import float64
from numpy.typing import NDArray

Numeric = TypeVar("Numeric", float, NDArray[float64])


class Timescale(Enum):
    """Supported timestamp units for time-series data."""

    US = "us"
    MS = "ms"
    S = "s"


def to_seconds(timestamp: Numeric, source_time_unit: Timescale) -> Numeric:
    """Convert a timestamp from the given unit to seconds.

    Parameters
    ----------
    timestamp : float | NDArray[Float64]
        Timestamp(s) to convert.
    source_time_unit : Timescale
        Unit of the input timestamp.

    Returns
    -------
    float | NDArray[Float64]
        Timestamp(s) in seconds.
    """
    if source_time_unit == Timescale.US:
        return timestamp / 1e6
    if source_time_unit == Timescale.MS:
        return timestamp / 1e3
    return timestamp


def from_seconds(timestamp_s: Numeric, target_time_unit: Timescale) -> Numeric:
    """Convert a timestamp from seconds to the given unit.

    Parameters
    ----------
    timestamp_s : float | NDArray[Float64]
        Timestamp(s) in seconds.
    target_time_unit : Timescale
        Desired output unit.

    Returns
    -------
    float | NDArray[Float64]
        Timestamp(s) in the target unit.
    """
    if target_time_unit == Timescale.US:
        return timestamp_s * 1e6
    if target_time_unit == Timescale.MS:
        return timestamp_s * 1e3
    return timestamp_s


@overload
def convert_time(
    timestamp: float, source_time_unit: Timescale, target_time_unit: Timescale
) -> float: ...
@overload
def convert_time(
    timestamp: NDArray[float64],
    source_time_unit: Timescale,
    target_time_unit: Timescale,
) -> NDArray[float64]: ...
def convert_time(
    timestamp: Numeric,
    source_time_unit: Timescale,
    target_time_unit: Timescale,
) -> Numeric:
    """Convert a timestamp between two timescale units.

    Parameters
    ----------
    timestamp : float | NDArray[Float64]
        Timestamp(s) to convert.
    source_time_unit : Timescale
        Unit of the input timestamp.
    target_time_unit : Timescale
        Desired output unit.

    Returns
    -------
    float | NDArray[Float64]
        Timestamp(s) in the target unit.

    Examples
    --------
    >>> convert_time(5000.0, Timescale.MS, Timescale.S)
    5.0
    """
    return from_seconds(to_seconds(timestamp, source_time_unit), target_time_unit)


@overload
def mph_to_m_per_s(value: float) -> float: ...
@overload
def mph_to_m_per_s(value: NDArray[float64]) -> NDArray[float64]: ...
def mph_to_m_per_s(value: Numeric) -> Numeric:
    """Convert a speed value from miles per hour to meters per second.

    Parameters
    ----------
    value : float | NDArray[Float64]
        Speed in mph.

    Returns
    -------
    float | NDArray[Float64]
        Speed in m/s.

    Examples
    --------
    >>> mph_to_m_per_s(1.0)
    0.44704
    """
    return value / 3600 * 1609.34


@overload
def in_to_m(value: float) -> float: ...
@overload
def in_to_m(value: NDArray[float64]) -> NDArray[float64]: ...
def in_to_m(value: Numeric) -> Numeric:
    """Convert a length value from inches to meters.

    Parameters
    ----------
    value : float | NDArray[Float64]
        Length in inches.

    Returns
    -------
    float | NDArray[Float64]
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
