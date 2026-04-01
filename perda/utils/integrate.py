import numpy as np
import polars as pl
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from ..analyzer.data_instance import DataInstance
from .units import MAD_TO_STD, Timescale, convert_time


def integrate_over_time_range(
    data_instance: DataInstance,
    start_time: int = 0,
    end_time: int = -1,
    source_time_unit: Timescale = Timescale.MS,
    target_time_unit: Timescale = Timescale.S,
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
    source_time_unit : Timescale, optional
        Time unit of inputs. Default is Timescale.MS
    target_time_unit : Timescale, optional
        Target time unit for averaging. Default is Timescale.S

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

    ts: npt.NDArray = convert_time(
        data_instance.timestamp_np.astype(np.float64),
        source_time_unit,
        target_time_unit,
    )
    values = data_instance.value_np.astype(np.float64)

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
    source_time_unit: Timescale = Timescale.MS,
    target_time_unit: Timescale = Timescale.S,
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
    source_time_unit : Timescale, optional
        Time unit of inputs. Default is Timescale.MS
    target_time_unit : Timescale, optional
        Target time unit for averaging. Default is Timescale.S

    Returns
    -------
    float
        Time-weighted average value over the time range

    Notes
    -----
    Uses numpy.trapz (trapezoidal rule) for numerical integration of discrete data.
    """
    if len(data_instance.timestamp_np) == 1:
        return float(data_instance.value_np[0])

    integral = integrate_over_time_range(
        data_instance, start_time, end_time, source_time_unit, target_time_unit
    )

    if integral == 0.0:
        return 0.0

    ts = data_instance.timestamp_np.astype(np.float64)
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

    actual_start_time = convert_time(
        actual_start_time, source_time_unit, target_time_unit
    )
    actual_end_time = convert_time(actual_end_time, source_time_unit, target_time_unit)

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


def smoothed_filtered_integration(
    data,
    source_time_unit: Timescale = Timescale.US,
    target_time_unit: Timescale = Timescale.S,
    filter_window_size=10,
    n_sigmas=3,
    smoothing_window_len=11,
    smoothing_poly_order=2,
):
    """Integrate a time-series signal after outlier removal and smoothing.

    Cleans spikes via rolling MAD-based outlier detection, applies Savitzky-Golay
    smoothing, then computes the cumulative trapezoidal integral over time.

    Parameters
    ----------
    data : DataInstance
        Time-series signal to integrate.
    source_time_unit : Timescale
        Time unit of inputs. Default is Timescale.US.
    target_time_unit : Timescale
        Target time unit for integration result. Default is Timescale.S.
    filter_window_size : int
        Rolling window size used for median and MAD outlier detection.
    n_sigmas : float
        Number of scaled MAD units beyond which a point is considered an outlier.
    smoothing_window_len : int
        Window length for Savitzky-Golay smoothing filter.
    smoothing_poly_order : int
        Polynomial order for Savitzky-Golay smoothing filter.

    Returns
    -------
    tuple[NDArray[Float64], NDArray[Float64], NDArray[Float64]]
        A tuple of (timestamps, smoothed values, cumulative integral), all of the
        same length as the input signal.
    """

    v = pl.Series(data.value_np)
    t = np.array(data.timestamp_np)

    # Calculate rolling median and the MAD (Median Absolute Deviation)
    rolling_median = v.rolling(window=filter_window_size, center=True).median()
    rolling_std = (
        v.rolling(window=filter_window_size, center=True).apply(
            lambda x: np.abs(x - x.median()).median()
        )
        * MAD_TO_STD
    )

    # Identify spikes (points more than X "standard deviations" from the local median)
    is_outlier = np.abs(v - rolling_median) > (n_sigmas * rolling_std)

    # interpolate the outliers
    v_cleaned = v.copy()
    v_cleaned[is_outlier] = np.nan
    v_cleaned = v_cleaned.interpolate(method="linear").bfill().ffill()

    # Smoothing
    if len(v_cleaned) > smoothing_window_len:
        v_smooth = savgol_filter(v_cleaned, smoothing_window_len, smoothing_poly_order)
    else:
        v_smooth = v_cleaned

    v_integrated = cumulative_trapezoid(
        v_smooth, x=convert_time(t, source_time_unit, target_time_unit), initial=0
    )

    return t, v_smooth, v_integrated
