import numpy as np
import polars as pl
from numpy import float64
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from ..core_data_structures.data_instance import DataInstance
from ..units import MAD_TO_STD, Timescale, convert_time


def integrate_over_time_range(
    data_instance: DataInstance,
    source_time_unit: Timescale,
    start_time: int = 0,
    end_time: int = -1,
    target_time_unit: Timescale = Timescale.S,
) -> float:
    """
    Get integral of the value over time using the trapezoidal rule.

    Parameters
    ----------
    data_instance : DataInstance
        DataInstance to integrate
    source_time_unit : Timescale
        Time unit of ``data_instance.timestamp_np``. Pass ``aly.data.timestamp_unit``.
    start_time : int, optional
        Start of the window, as an absolute timestamp in ``target_time_unit``. Default is 0
    end_time : int, optional
        End of the window, as an absolute timestamp in ``target_time_unit``. -1 means end
        of data. Default is -1
    target_time_unit : Timescale, optional
        Time unit to integrate over, e.g. seconds gives value * seconds. Default is Timescale.S

    Returns
    -------
    float
        Integral of the value over the time range

    Notes
    -----
    Uses numpy.trapezoid (trapezoidal rule) for numerical integration of discrete data.

    Examples
    --------
    >>> from perda.utils import integrate_over_time_range
    >>> current = aly.data["bms.pack.current"]
    >>> charge_amp_seconds = integrate_over_time_range(
    ...     current, source_time_unit=aly.data.timestamp_unit
    ... )
    """
    if len(data_instance.timestamp_np) < 2:
        return 0.0

    ts: NDArray[float64] = convert_time(
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
    source_time_unit: Timescale,
    start_time: int = 0,
    end_time: int = -1,
    target_time_unit: Timescale = Timescale.S,
) -> float:
    """
    Get average value over time using integral divided by time range.

    Parameters
    ----------
    data_instance : DataInstance
        DataInstance to average
    source_time_unit : Timescale
        Time unit of ``data_instance.timestamp_np``. Pass ``aly.data.timestamp_unit``.
    start_time : int, optional
        Start of the window, as an absolute timestamp in ``target_time_unit``. Default is 0
    end_time : int, optional
        End of the window, as an absolute timestamp in ``target_time_unit``. -1 means end
        of data. Default is -1
    target_time_unit : Timescale, optional
        Time unit of ``start_time`` and ``end_time``. Default is Timescale.S

    Returns
    -------
    float
        Time-weighted average value over the time range

    Notes
    -----
    Uses numpy.trapezoid (trapezoidal rule) for numerical integration of discrete data.

    Examples
    --------
    >>> from perda.utils import average_over_time_range
    >>> voltage = aly.data["bms.pack.voltage"]
    >>> mean_voltage = average_over_time_range(
    ...     voltage, source_time_unit=aly.data.timestamp_unit
    ... )
    """
    if len(data_instance.timestamp_np) == 1:
        return float(data_instance.value_np[0])

    integral = integrate_over_time_range(
        data_instance,
        source_time_unit,
        start_time=start_time,
        end_time=end_time,
        target_time_unit=target_time_unit,
    )

    ts: NDArray[float64] = convert_time(
        data_instance.timestamp_np.astype(np.float64),
        source_time_unit,
        target_time_unit,
    )
    actual_start_time = max(start_time, ts[0])
    actual_end_time = ts[-1] if end_time == -1 else min(end_time, ts[-1])

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

    Notes
    -----
    ``start_time`` and ``end_time`` are raw timestamps, in ``aly.data.timestamp_unit``.

    Examples
    --------
    >>> from perda.utils import get_data_slice_by_timestamp
    >>> di = aly.data["pcm.wheelSpeeds.frontRight"]
    >>> first_half = get_data_slice_by_timestamp(di, end_time=int(di.timestamp_np[len(di) // 2]))
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
    data: DataInstance,
    source_time_unit: Timescale,
    target_time_unit: Timescale = Timescale.S,
    filter_window_size: int = 10,
    n_sigmas: float = 3,
    smoothing_window_len: int = 11,
    smoothing_poly_order: int = 2,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Integrate a time-series signal after outlier removal and smoothing.

    Cleans spikes via rolling MAD-based outlier detection, applies Savitzky-Golay
    smoothing, then computes the cumulative trapezoidal integral over time.

    Parameters
    ----------
    data : DataInstance
        Time-series signal to integrate.
    source_time_unit : Timescale
        Time unit of ``data.timestamp_np``. Pass ``aly.data.timestamp_unit``.
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

    Examples
    --------
    >>> from perda.utils import smoothed_filtered_integration
    >>> power = aly.data["bms.pack.voltage"] * aly.data["bms.pack.current"]
    >>> ts, smoothed_power, energy_joules = smoothed_filtered_integration(
    ...     power, source_time_unit=aly.data.timestamp_unit
    ... )
    """
    v_np = np.array(data.value_np, dtype=np.float64)
    t = np.array(data.timestamp_np, dtype=np.float64)
    v_series = pl.Series(v_np)

    # Rolling median and MAD (Median Absolute Deviation)
    rolling_median = v_series.rolling_median(
        window_size=filter_window_size, min_samples=1, center=True
    ).to_numpy()
    rolling_std = v_series.rolling_map(
        lambda x: np.median(np.abs(x.to_numpy() - np.median(x.to_numpy())))
        * MAD_TO_STD,
        window_size=filter_window_size,
        min_samples=1,
        center=True,
    ).to_numpy()

    # Identify and replace spikes with NaN
    is_outlier = np.abs(v_np - rolling_median) > (n_sigmas * rolling_std)
    v_cleaned = v_np.copy()
    v_cleaned[is_outlier] = np.nan

    # Linear interpolation over outlier gaps
    nans = np.isnan(v_cleaned)
    if nans.any():
        idx = np.arange(len(v_cleaned))
        not_nan_idx = idx[~nans]
        if len(not_nan_idx) > 1:
            v_cleaned = np.interp(idx, not_nan_idx, v_cleaned[~nans])
        elif len(not_nan_idx) == 1:
            v_cleaned[:] = v_cleaned[not_nan_idx[0]]

    # Smoothing
    if len(v_cleaned) > smoothing_window_len:
        v_smooth = savgol_filter(v_cleaned, smoothing_window_len, smoothing_poly_order)
    else:
        v_smooth = v_cleaned

    v_integrated = cumulative_trapezoid(
        v_smooth, x=convert_time(t, source_time_unit, target_time_unit), initial=0
    )

    return t, v_smooth, v_integrated
