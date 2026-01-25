import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..analyzer.models import DataInstance


def integrate_over_time_range(
    data_instance: DataInstance,
    start_time: int = 0,
    end_time: int = -1,
    time_unit: str = "ms",
) -> float:
    """
    Get integral of the value over time using interpolation and exact bounds integration.

    Parameters
    ----------
    data_instance : DataInstance
        DataInstance to integrate
    start_time : int, optional
        Start time for integration. Default is 0
    end_time : int, optional
        End time for integration. -1 means end of data. Default is -1
    time_unit : str, optional
        Time unit for integration: "ms" or "s". Default is "ms"

    Returns
    -------
    float
        Integral of the value over the time range

    Notes
    -----
    Uses scipy.interpolate.interp1d to create a continuous function from discrete data,
    then scipy.integrate.quad for precise integration over exact time bounds.
    """
    if len(data_instance.timestamp_np) < 2:
        return 0.0

    ts = data_instance.timestamp_np.astype(np.float64)
    values = data_instance.value_np.astype(np.float64)

    # Set actual bounds
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

    if actual_start_time >= actual_end_time:
        return 0.0

    # Create interpolation function
    # Use linear interpolation with bounds handling
    interp_func = interp1d(
        ts,
        values,
        kind="linear",
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )

    # Convert time bounds for integration
    if time_unit == "s":
        integration_start = actual_start_time / 1e3
        integration_end = actual_end_time / 1e3

        # Create wrapper function that handles time unit conversion
        def integrand(t_seconds):
            t_ms = t_seconds * 1e3
            return interp_func(t_ms)

    else:
        integration_start = actual_start_time
        integration_end = actual_end_time
        integrand = interp_func

    # Integrate using scipy.integrate.quad
    integral, _ = quad(integrand, integration_start, integration_end)
    return float(integral)


def average_over_time_range(
    data_instance: DataInstance,
    start_time: int = 0,
    end_time: int = -1,
    time_unit: str = "ms",
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
    time_unit : str, optional
        Time unit for averaging: "ms" or "s". Default is "ms"

    Returns
    -------
    float
        Time-weighted average value over the time range

    Notes
    -----
    Uses scipy.interpolate.interp1d to create a continuous function from discrete data,
    then scipy.integrate.quad for precise integration over exact time bounds.
    """
    if len(data_instance.timestamp_np) < 2:
        return 0.0

    ts = data_instance.timestamp_np.astype(np.float64)

    # Set actual bounds
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

    if actual_start_time >= actual_end_time:
        return 0.0

    # Calculate time range
    time_range = actual_end_time - actual_start_time
    if time_unit == "s":
        time_range /= 1e3

    if time_range == 0:
        return 0.0

    # Get integral over the time range
    integral = integrate_over_time_range(data_instance, start_time, end_time, time_unit)
    average = integral / time_range
    return float(average)


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
        canid=original_instance.canid,
    )
