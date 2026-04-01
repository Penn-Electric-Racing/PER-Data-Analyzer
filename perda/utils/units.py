from enum import Enum

import numpy.typing as npt


class Timescale(Enum):
    US = "us"
    MS = "ms"
    S = "s"


def _to_seconds(
    timestamp: float | npt.NDArray, source_time_unit: Timescale
) -> float | npt.NDArray:
    if source_time_unit == Timescale.US:
        return timestamp / 1e6
    if source_time_unit == Timescale.MS:
        return timestamp / 1e3
    return timestamp


def _from_seconds(
    timestamp_s: float | npt.NDArray, target_time_unit: Timescale
) -> float | npt.NDArray:
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
    return _from_seconds(_to_seconds(timestamp, source_time_unit), target_time_unit)


# Consistency factor to convert MAD to an equivalent standard deviation estimate
# under the assumption of normally distributed data: sigma ≈ MAD / 0.6745
MAD_TO_STD = 1.4826
