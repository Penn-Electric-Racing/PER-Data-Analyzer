from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..units import Timescale, to_seconds


def apply_sos_filter(
    signal: NDArray[float64],
    sos: NDArray[float64],
    order: int,
) -> NDArray[float64] | None:
    """Apply a second-order-section filter with NaN masking.

    Parameters
    ----------
    signal : NDArray[float64]
        Input signal, may contain NaN.
    sos : NDArray[float64]
        Second-order sections from ``scipy.signal.butter``.
    order : int
        Filter order (used to compute minimum valid sample threshold).

    Returns
    -------
    NDArray[float64] | None
        Filtered signal with NaN preserved at original NaN positions, or
        ``None`` if there are too few valid samples to filter.
    """
    valid = ~np.isnan(signal)
    min_samples = 3 * (2 * order + 1)

    count = int(valid.sum())
    if count < min_samples:
        print(f"Too few valid points " f"({count} < {min_samples}), skipping")
        return None

    filtered = np.full_like(signal, np.nan)
    filtered[valid] = sosfiltfilt(sos, signal[valid])
    return filtered


def lowpass_filter(
    di: DataInstance | list[DataInstance],
    cutoff_hz: float,
    source_time_unit: Timescale = Timescale.MS,
    order: int = 4,
) -> DataInstance | list[DataInstance]:
    """Apply a Butterworth lowpass filter in the time domain.

    Returns a new DataInstance (or list thereof) with filtered values.
    Original timestamps and metadata are preserved. NaN positions in the
    input remain NaN in the output.

    To re-filter at a different cutoff without stacking, pass the original
    (pre-filter) DataInstance again::

        di_original = aly.data["pcm.wheelSpeeds.frontRight"]
        di_10hz = lowpass_filter(di_original, cutoff_hz=10.0)
        di_5hz  = lowpass_filter(di_original, cutoff_hz=5.0)   # not stacked

    Parameters
    ----------
    di : DataInstance | list[DataInstance]
        Input signal(s). Lists are processed independently.
    cutoff_hz : float
        Cutoff frequency in Hz.
    source_time_unit : Timescale, optional
        Timestamp unit of ``di.timestamp_np``. Default is ``Timescale.MS``.
    order : int, optional
        Butterworth filter order. The effective order is 2× due to
        forward-backward (``sosfiltfilt``) application. Default is 4.

    Returns
    -------
    DataInstance | list[DataInstance]
        New DataInstance(s) with filtered ``value_np``.

    Examples
    --------
    >>> di_filtered = lowpass_filter(di, cutoff_hz=10.0)
    >>> di_filtered = lowpass_filter([di_a, di_b], cutoff_hz=5.0)
    """
    di_list = [di] if isinstance(di, DataInstance) else di

    results: list[DataInstance] = []

    for instance in di_list:
        ts_s = to_seconds(instance.timestamp_np.astype(np.float64), source_time_unit)
        dt = float(np.median(np.diff(ts_s)))
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(
                "Non-positive median time step "
                f"({dt:.6f} s). Cannot determine sample rate."
            )

        fs = 1.0 / dt
        nyq = fs / 2.0

        if cutoff_hz >= nyq:
            print(
                f"Cutoff frequency ({cutoff_hz} Hz) greater than Nyquist frequency"
                f"({nyq:.1f} Hz), skipping filter"
            )
            results.append(instance)
            continue

        sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
        signal = instance.value_np.astype(np.float64)
        filtered = apply_sos_filter(signal, sos, order)

        if filtered is None:
            results.append(instance)
            continue

        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label or f"var_id={instance.var_id}",
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if len(results) == 1 else results


def lowpass_filter_by_distance(
    di: DataInstance | list[DataInstance],
    distance_di: DataInstance,
    cutoff_freq_per_meter: float,
    order: int = 4,
) -> DataInstance | list[DataInstance]:
    """Apply a Butterworth lowpass filter in the spatial domain.

    The sample rate is derived from a cumulative-distance DataInstance
    rather than timestamps.  ``distance_di`` is interpolated onto each
    signal's timestamp grid before use, so the two need not share the
    same grid.

    Parameters
    ----------
    di : DataInstance | list[DataInstance]
        Input signal(s) to filter.
    distance_di : DataInstance
        Cumulative distance in meters on any timestamp grid.
    cutoff_freq_per_meter : float
        Cutoff frequency in cycles per meter (1/m).
    order : int, optional
        Butterworth filter order. Default is 4.

    Returns
    -------
    DataInstance | list[DataInstance]
        New DataInstance(s) with spatially filtered ``value_np``, on the
        original signal timestamp grid.

    Examples
    --------
    >>> di_filtered = lowpass_filter_by_distance(di, distance_di, cutoff_freq_per_meter=0.05)
    """
    di_list = [di] if isinstance(di, DataInstance) else di
    results: list[DataInstance] = []

    for instance in di_list:
        # Align distance onto the signal's timestamp grid
        _, distance_aligned = left_join_data_instances(instance, distance_di)
        dist_values = distance_aligned.value_np.astype(np.float64)
        signal = instance.value_np.astype(np.float64)

        # Remove duplicate-distance samples (car stationary) before spatial filtering.
        diffs = np.diff(dist_values)
        moving = np.concatenate(([True], diffs > 0))
        dist_dedup = dist_values[moving]
        signal_dedup = signal[moving]

        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size == 0:
            raise ValueError(
                "No positive distance steps found. Cannot determine spatial sample rate."
            )

        dx_median = float(np.median(positive_diffs))
        fs = 1.0 / dx_median
        nyq = fs / 2.0

        if cutoff_freq_per_meter >= nyq:
            print(
                f"Cutoff frequency ({cutoff_freq_per_meter} 1/m) greater than Nyquist frequency"
                f"({nyq:.4f} 1/m), skipping filter"
            )
            results.append(instance)
            continue

        sos = butter(order, cutoff_freq_per_meter / nyq, btype="low", output="sos")
        filtered_dedup = apply_sos_filter(signal_dedup, sos, order)

        if filtered_dedup is None:
            results.append(instance)
            continue

        # Interpolate filtered values back onto the original distance grid
        filtered = np.interp(dist_values, dist_dedup, filtered_dedup)

        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label or f"var_id={instance.var_id}",
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if len(results) == 1 else results


def zscore_filter(
    di: DataInstance | list[DataInstance],
    window_s: float,
    threshold: float,
    source_time_unit: Timescale = Timescale.MS,
) -> DataInstance | list[DataInstance]:
    """Remove outliers using a rolling-window z-score and interpolate gaps.

    For each sample, a z-score is computed relative to the local rolling
    mean and standard deviation.  Samples with ``|z| > threshold`` are
    replaced with ``NaN`` and then linearly interpolated.

    Parameters
    ----------
    di : DataInstance | list[DataInstance]
        Input signal(s).
    window_s : float
        Rolling window size in seconds
    threshold : float
        Z-score threshold; samples exceeding this are masked.
    source_time_unit : Timescale, optional
        Timestamp unit. Default is ``Timescale.MS``.

    Returns
    -------
    DataInstance | list[DataInstance]
        New DataInstance(s) with outliers replaced by linear interpolation.

    Examples
    --------
    >>> di_clean = zscore_filter(di, window_s=1.0, threshold=3.0)
    """
    di_list = [di] if isinstance(di, DataInstance) else di

    results: list[DataInstance] = []

    for instance in di_list:
        ts_s = to_seconds(instance.timestamp_np.astype(np.float64), source_time_unit)
        dt = float(np.median(np.diff(ts_s)))
        if dt <= 0:
            raise ValueError(
                "Non-positive median time step "
                f"({dt:.6f} s). Cannot determine sample rate."
            )

        fs = 1.0 / dt
        win_samples = max(3, int(round(window_s * fs)))

        signal = instance.value_np.astype(np.float64)

        roll_mean = uniform_filter1d(signal, size=win_samples, mode="nearest")
        roll_sq_mean = uniform_filter1d(signal**2, size=win_samples, mode="nearest")
        # Guard against floating-point negatives inside sqrt
        roll_std = np.sqrt(np.maximum(roll_sq_mean - roll_mean**2, 0.0))

        z = np.zeros_like(signal, dtype=np.float64)
        mask = roll_std > 0
        z[mask] = np.abs((signal[mask] - roll_mean[mask]) / roll_std[mask])

        outlier_mask = z > threshold
        n_masked = int(outlier_mask.sum())

        filtered = signal.copy()
        filtered[outlier_mask] = np.nan

        if n_masked > 0:
            indices = np.arange(len(filtered), dtype=np.float64)
            valid_mask = ~np.isnan(filtered)
            if valid_mask.sum() >= 2:
                filtered = np.interp(
                    indices,
                    indices[valid_mask],
                    filtered[valid_mask],
                )

        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label or f"var_id={instance.var_id}",
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if len(results) == 1 else results


def compute_fft(
    di: DataInstance,
    source_time_unit: Timescale = Timescale.MS,
    distance_di: DataInstance | None = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute the real FFT magnitude spectrum of a DataInstance.

    NaN values are dropped before the transform.  The signal is
    mean-subtracted to suppress the DC component.

    Parameters
    ----------
    di : DataInstance
        Signal to transform.
    source_time_unit : Timescale, optional
        Timestamp unit. Used only in time-domain mode. Default is
        ``Timescale.MS``.
    distance_di : DataInstance | None, optional
        If provided, the FFT is computed in the spatial domain and the
        sample spacing is derived from cumulative distance in meters
        (interpolated onto ``di``'s timestamp grid). Default is None
        (time domain, result in Hz).

    Returns
    -------
    frequencies : NDArray[float64]
        Frequency axis values.
    magnitudes : NDArray[float64]
        FFT magnitude (``abs(rfft(signal - mean))``).

    Examples
    --------
    >>> freqs, mags = compute_fft(di)
    >>> fig = plot_fft_spectrum([freqs], [mags], [di.label])
    """
    signal = di.value_np.astype(np.float64)
    valid = ~np.isnan(signal)
    signal_clean = signal[valid]

    if len(signal_clean) < 2:
        raise ValueError("Not enough valid samples to compute FFT.")

    if distance_di is not None:
        _, distance_aligned = left_join_data_instances(di, distance_di)
        dist_values = distance_aligned.value_np.astype(np.float64)[valid]
        positive_diffs = np.diff(dist_values)
        positive_diffs = positive_diffs[positive_diffs > 0]
        if positive_diffs.size == 0 or not np.all(np.isfinite(positive_diffs)):
            raise ValueError(
                "No positive distance steps found; cannot compute spatial FFT."
            )
        dx = float(np.median(positive_diffs))
    else:
        ts_s = to_seconds(di.timestamp_np.astype(np.float64), source_time_unit)
        dt = float(np.median(np.diff(ts_s)))
        if dt <= 0:
            raise ValueError(
                "Non-positive median time step "
                f"({dt:.6f} s). Cannot determine sample rate."
            )
        dx = dt

    frequencies: NDArray[float64] = rfftfreq(len(signal_clean), d=dx)
    magnitudes: NDArray[float64] = np.abs(rfft(signal_clean - np.mean(signal_clean)))
    return frequencies, magnitudes
