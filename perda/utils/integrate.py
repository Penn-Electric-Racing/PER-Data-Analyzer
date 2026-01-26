import numpy as np
from numpy.typing import NDArray

from ..analyzer.data_instance import DataInstance
from .types import Timescale


def integrate_over_time_range(
    data_instance: DataInstance,
    start_time: int = 0,
    end_time: int = -1,
    time_unit: Timescale = Timescale.S,
) -> float:
    """
    Get integral of the value over time using the trapezoidal rule.

    Parameters
    ----------
    data_instance : DataInstance
        DataInstance to integrate
    start_time : int, optional
        Start time for integration. Default is 0
    end_time : int, optional
        End time for integration. -1 means end of data. Default is -1
    time_unit : Timescale, optional
        Time unit for integration: milliseconds or seconds. Default is seconds

    Returns
    -------
    float
        Integral of the value over the time range

    Notes
    -----
    Uses numpy.trapz (trapezoidal rule) for numerical integration of discrete data.
    """
    if len(data_instance.timestamp_np) < 2:
        return 0.0

    ts = data_instance.timestamp_np.astype(np.float64)
    values = data_instance.value_np.astype(np.float64)

    # Convert time to desired unit
    if time_unit == Timescale.S:
        ts = ts / 1e3

    # Set actual bounds
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

    if actual_start_time >= actual_end_time:
        return 0.0

    # Filter data to the time range
    mask = (ts >= actual_start_time) & (ts <= actual_end_time)
    ts_filtered = ts[mask]
    values_filtered = values[mask]

    if len(ts_filtered) < 2:
        return 0.0

    # Integrate using trapezoidal rule
    integral = np.trapezoid(values_filtered, ts_filtered)
    return float(integral)


def average_over_time_range(
    data_instance: DataInstance,
    start_time: int = 0,
    end_time: int = -1,
    time_unit: Timescale = Timescale.MS,
) -> float:
    """
    Get average value over time using integral divided by time range.

    Parameters
    ----------
    data_instance : DataInstance
        DataInstance to average
    start_time : int, optional
        Start time for averaging. Default is 0
    end_time : int, optional
        End time for averaging. -1 means end of data. Default is -1
    time_unit : Timescale, optional
        Time unit for averaging: "ms" or "s". Default is Timescale.MS

    Returns
    -------
    float
        Time-weighted average value over the time range

    Notes
    -----
    Uses numpy.trapz (trapezoidal rule) for numerical integration of discrete data.
    """
    integral = integrate_over_time_range(data_instance, start_time, end_time, time_unit)

    if integral == 0.0:
        return 0.0

    ts = data_instance.timestamp_np.astype(np.float64)
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

    if time_unit == Timescale.S:
        actual_start_time = actual_start_time / 1e3
        actual_end_time = actual_end_time / 1e3

    time_range = actual_end_time - actual_start_time

    return float(integral / time_range) if time_range > 0 else 0.0


def get_data_slice_by_timestamp(
    original_instance: DataInstance, start_time: int = 0, end_time: int = -1
) -> DataInstance:
    """
    Get a new DataInstance with data in [start_time, end_time).

    Parameters
    ----------
    original_instance : DataInstance
        Original DataInstance to slice
    start_time : int, optional
        Start time (inclusive). Default is 0
    end_time : int, optional
        End time (exclusive). -1 means till end. Default is -1

    Returns
    -------
    DataInstance
        New DataInstance containing only data within the specified time range
    """
    if end_time < 0:
        mask = original_instance.timestamp_np >= start_time
    else:
        mask = (original_instance.timestamp_np >= start_time) & (
            original_instance.timestamp_np < end_time
        )
    return DataInstance(
        timestamp_np=original_instance.timestamp_np[mask],
        value_np=original_instance.value_np[mask],
        label=original_instance.label,
        var_id=original_instance.var_id,
    )
