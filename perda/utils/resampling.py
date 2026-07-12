import numpy as np
from numpy.typing import NDArray

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.resampling_helpers import ResampleMethod, _interpolate
from ..units import Timescale, _to_seconds


def apply_time_offset(
    di: DataInstance | list[DataInstance],
    offset_s: float,
    source_time_unit: Timescale = Timescale.MS,
) -> DataInstance | list[DataInstance]:
    """Shift signals in time by a fixed offset, but keeping the same timestamp list.

    Shifts the values, not the timestamps. Create a new time series with the same values offset_s away,
    then re-interpolate the original timestamps onto this series.

    Parameters
    ----------
    di : DataInstance | list[DataInstance]
        Input signal(s). Lists are processed independently.
    offset_s : float
        Time shift in seconds (positive shifts the signal later).
    source_time_unit : Timescale, optional
        Timestamp unit of ``di.timestamp_np``. Default is ``Timescale.MS``.

    Returns
    -------
    DataInstance | list[DataInstance]
        New DataInstance(s) with time-shifted ``value_np`` on the original timestamp grid.

    Examples
    --------
    >>> di_aligned = apply_time_offset(gps_speed_di, offset_s=-0.08)
    >>> a, b = apply_time_offset([di_a, di_b], offset_s=0.05)
    """
    di_list = [di] if isinstance(di, DataInstance) else di

    results: list[DataInstance] = []
    for instance in di_list:
        if offset_s == 0.0:
            results.append(instance)
            continue

        t_s: NDArray = _to_seconds(
            instance.timestamp_np.astype(np.float64), source_time_unit
        )
        signal = instance.value_np.astype(np.float64)
        valid = ~np.isnan(signal)
        if valid.sum() < 2:
            print("Too few valid points to interpolate, skipping time offset")
            results.append(instance)
            continue

        src_t: NDArray = t_s[valid] + offset_s
        src_v: NDArray = signal[valid]
        shifted = np.interp(t_s, src_t, src_v, left=src_v[0], right=src_v[-1])

        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=shifted,
                label=instance.label or f"var_id={instance.var_id}",
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if len(results) == 1 else results


def resample_to_freq(
    di: DataInstance,
    freq_hz: float,
    timestamp_divisor: float,
    method: ResampleMethod = ResampleMethod.LINEAR,
) -> DataInstance:
    """
    Resample a DataInstance onto a uniform frequency grid.

    Parameters
    ----------
    di : DataInstance
        Source DataInstance
    freq_hz : float
        Target sampling frequency in Hz
    timestamp_divisor : float
        Raw timestamp units per second (e.g. 1e6 for microseconds)
    method : ResampleMethod, optional
        Interpolation method. Default is LINEAR.

    Returns
    -------
    DataInstance
        New DataInstance with values resampled onto a uniform timestamp grid
    """
    dt = timestamp_divisor / freq_hz
    target_ts = np.arange(
        di.timestamp_np[0], di.timestamp_np[-1], dt, dtype=np.float64
    ).astype(np.int64)
    resampled_val = _interpolate(
        target_ts.astype(np.float64),
        di.timestamp_np.astype(np.float64),
        di.value_np,
        method,
    )

    return DataInstance(
        timestamp_np=target_ts,
        value_np=resampled_val,
        label=di.label,
        var_id=di.var_id,
        cpp_name=di.cpp_name,
    )
