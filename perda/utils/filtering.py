"""Digital signal filtering and FFT computation utilities for DataInstances."""

from __future__ import annotations

import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..units import Timescale, _to_seconds

# ---------------------------------------------------------------------------
# Variables that should not be lowpass-filtered on raw values because they
# contain wrapping discontinuities (e.g. yaw angle wraps at ±180°).
# Filter after unwrapping if needed.
# ---------------------------------------------------------------------------
_NEVER_FILTER: frozenset[str] = frozenset(
    {
        "pcm.vnav.yawPitchRoll.yaw",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_list_di(
    di: DataInstance | list[DataInstance],
) -> tuple[list[DataInstance], bool]:
    """Normalize a single DataInstance or list to a list, reporting original type.

    Parameters
    ----------
    di : DataInstance | list[DataInstance]
        Input to normalize.

    Returns
    -------
    tuple[list[DataInstance], bool]
        The normalized list and a flag that is ``True`` when the input was a
        single ``DataInstance`` (so the caller can unwrap the result).
    """
    if isinstance(di, DataInstance):
        return [di], True
    return list(di), False


def _median_sample_spacing_s(
    timestamp_np: NDArray[int64],
    source_time_unit: Timescale,
) -> float:
    """Compute the median sample spacing in seconds.

    Parameters
    ----------
    timestamp_np : NDArray[int64]
        Raw integer timestamps in the given time unit.
    source_time_unit : Timescale
        Unit of the timestamps.

    Returns
    -------
    float
        Median inter-sample interval in seconds.

    Notes
    -----
    Raises ``ValueError`` when the median spacing is non-positive, which
    indicates a degenerate or non-monotonic timestamp array.
    """
    ts_s = _to_seconds(timestamp_np.astype(np.float64), source_time_unit)
    dt = np.median(np.diff(ts_s))
    if dt <= 0:
        raise ValueError(
            "Non-positive median sample spacing; cannot determine sample rate. "
            "Ensure timestamps are strictly increasing."
        )
    return float(dt)


def _apply_sos_filter(
    signal: NDArray[float64],
    sos: NDArray[float64],
    order: int,
    label: str,
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
    label : str
        Variable label for diagnostic messages.

    Returns
    -------
    NDArray[float64] | None
        Filtered signal with NaN preserved at original NaN positions, or
        ``None`` if there are too few valid samples to filter.
    """
    valid = ~np.isnan(signal)
    min_samples = 3 * (2 * order + 1)
    if valid.sum() < min_samples:
        print(
            f"  [filter] {label}: too few valid points "
            f"({int(valid.sum())} < {min_samples}), skipping"
        )
        return None

    filtered = np.full_like(signal, np.nan)
    filtered[valid] = sosfiltfilt(sos, signal[valid])
    return filtered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    di_list, was_single = _ensure_list_di(di)
    results: list[DataInstance] = []

    for instance in di_list:
        label = instance.label or f"var_id={instance.var_id}"
        dt_s = _median_sample_spacing_s(instance.timestamp_np, source_time_unit)
        fs = 1.0 / dt_s
        nyq = fs / 2.0

        if cutoff_hz >= nyq:
            print(
                f"  [filter] {label}: cutoff ({cutoff_hz} Hz) >= Nyquist "
                f"({nyq:.1f} Hz), skipping filter"
            )
            results.append(instance)
            continue

        sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
        signal = instance.value_np.astype(np.float64)
        filtered = _apply_sos_filter(signal, sos, order, label)

        if filtered is None:
            results.append(instance)
            continue

        print(
            f"  [filter] {label}: lowpass @ {cutoff_hz} Hz  "
            f"({len(instance)} pts, fs={fs:.1f} Hz)"
        )
        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label,
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if was_single else results


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
    di_list, was_single = _ensure_list_di(di)
    results: list[DataInstance] = []

    for instance in di_list:
        label = instance.label or f"var_id={instance.var_id}"

        # Align distance onto the signal's timestamp grid
        _, distance_aligned = left_join_data_instances(instance, distance_di)
        dist_values = distance_aligned.value_np.astype(np.float64)

        dx_median = float(np.median(np.diff(dist_values)))
        if dx_median <= 0:
            raise ValueError(
                f"  [filter] {label}: non-positive median distance step "
                f"({dx_median:.6f} m); cannot determine spatial sample rate."
            )

        fs = 1.0 / dx_median
        nyq = fs / 2.0

        if cutoff_freq_per_meter >= nyq:
            print(
                f"  [filter] {label}: cutoff ({cutoff_freq_per_meter} 1/m) >= "
                f"Nyquist ({nyq:.4f} 1/m), skipping filter"
            )
            results.append(instance)
            continue

        sos = butter(order, cutoff_freq_per_meter / nyq, btype="low", output="sos")
        signal = instance.value_np.astype(np.float64)
        filtered = _apply_sos_filter(signal, sos, order, label)

        if filtered is None:
            results.append(instance)
            continue

        print(
            f"  [filter] {label}: lowpass @ {cutoff_freq_per_meter} 1/m  "
            f"({len(instance)} pts, fs={fs:.4f} 1/m)"
        )
        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label,
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if was_single else results


def zscore_filter(
    di: DataInstance | list[DataInstance],
    window_s: float = 2.0,
    threshold: float = 4.8,
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
    window_s : float, optional
        Rolling window size in seconds. Default is 2.0.
    threshold : float, optional
        Z-score threshold; samples exceeding this are masked. Default is 4.8.
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
    di_list, was_single = _ensure_list_di(di)
    results: list[DataInstance] = []

    for instance in di_list:
        label = instance.label or f"var_id={instance.var_id}"
        dt_s = _median_sample_spacing_s(instance.timestamp_np, source_time_unit)
        fs = 1.0 / dt_s
        win_samples = max(3, int(round(window_s * fs)))

        signal = instance.value_np.astype(np.float64)

        roll_mean = uniform_filter1d(signal, size=win_samples, mode="nearest")
        roll_sq_mean = uniform_filter1d(signal**2, size=win_samples, mode="nearest")
        # Guard against floating-point negatives inside sqrt
        roll_std = np.sqrt(np.maximum(roll_sq_mean - roll_mean**2, 0.0))

        with np.errstate(invalid="ignore"):
            z = np.where(roll_std > 0, np.abs((signal - roll_mean) / roll_std), 0.0)

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

        print(
            f"  [zscore] {label}: window={window_s} s ({win_samples} pts), "
            f"threshold={threshold}, masked {n_masked}/{len(instance)} pts (interpolated)"
        )
        results.append(
            DataInstance(
                timestamp_np=instance.timestamp_np,
                value_np=filtered,
                label=instance.label,
                var_id=instance.var_id,
                cpp_name=instance.cpp_name,
            )
        )

    return results[0] if was_single else results


def compute_fft(
    di: DataInstance,
    source_time_unit: Timescale = Timescale.MS,
    distance_di: DataInstance | None = None,
) -> tuple[NDArray[float64], NDArray[float64], str]:
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
    freq_unit : str
        ``"Hz"`` for time-domain, ``"1/m"`` for spatial-domain.

    Examples
    --------
    >>> freqs, mags, unit = compute_fft(di)
    >>> fig = plot_fft_spectrum([freqs], [mags], [di.label], freq_unit=unit)
    """
    signal = di.value_np.astype(np.float64)
    valid = ~np.isnan(signal)
    signal_clean = signal[valid]

    if distance_di is not None:
        _, distance_aligned = left_join_data_instances(di, distance_di)
        dist_values = distance_aligned.value_np.astype(np.float64)[valid]
        dx = float(np.median(np.diff(dist_values)))
        if dx <= 0:
            raise ValueError(
                "Non-positive median distance step; cannot compute spatial FFT."
            )
        freq_unit = "1/m"
    else:
        dt_s = _median_sample_spacing_s(di.timestamp_np, source_time_unit)
        dx = dt_s
        freq_unit = "Hz"

    frequencies: NDArray[float64] = rfftfreq(len(signal_clean), d=dx)
    magnitudes: NDArray[float64] = np.abs(rfft(signal_clean - np.mean(signal_clean)))
    return frequencies, magnitudes, freq_unit
