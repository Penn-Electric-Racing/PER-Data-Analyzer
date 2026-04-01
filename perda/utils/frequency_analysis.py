import numpy as np
from numpy import float64
from numpy.typing import NDArray
from plotly import graph_objects as go

from ..constants import DELIMITER, title_block
from ..models.data_instance import DataInstance
from ..plotting.plotting_constants import *
from ..plotting.scatter_histogram_plotter import plot_scatter_and_histogram
from .units import Timescale


def _s_conversion_factor(time_unit: Timescale) -> float:
    """Return the factor to convert from the given time unit to seconds."""
    if time_unit == Timescale.US:
        return 1e-6
    if time_unit == Timescale.MS:
        return 1e-3
    return 1.0


def analyze_frequency(
    data_instance: DataInstance,
    expected_frequency_hz: float | None = None,
    source_time_unit: Timescale = Timescale.MS,
    gap_threshold_multiplier: float = 2.0,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    plot_config: ScatterHistogramPlotConfig = DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG,
) -> go.Figure:
    """
    Analyse the sampling frequency of a DataInstance and return a diagnostic figure.

    Prints a summary of frequency statistics and gap detection, then returns a
    figure with two subplots: instantaneous frequency over time and a frequency
    histogram.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance whose logging cadence to analyse.
    expected_frequency_hz : float | None, optional
        Nominal expected sampling frequency in Hz. When provided, additional
        diagnostics (frequency error, missed-message estimate, reference lines)
        are included. Default is None.
    source_time_unit : Timescale, optional
        Timestamp unit used in ``data_instance.timestamp_np``. Default is ms.
    gap_threshold_multiplier : float, optional
        An interval is flagged as a gap when it exceeds this multiple of the
        expected interval (if ``expected_frequency_hz`` is given) or the median
        interval. Default is 2.0.
    font_config : FontConfig, optional
        Font sizes for plot elements. Default is DEFAULT_FONT_CONFIG.
    layout_config : LayoutConfig, optional
        Plot dimensions and margins. Default is DEFAULT_LAYOUT_CONFIG.
    plot_config : ScatterHistogramPlotConfig | None, optional
        Colors and histogram bin count. Default is DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG.

    Returns
    -------
    go.Figure
        Plotly figure with frequency time-series and frequency histogram subplots.

    Examples
    --------
    >>> fig = analyze_frequency(di, expected_frequency_hz=100)
    >>> fig.show()
    """
    label = data_instance.label or f"var_id={data_instance.var_id}"

    if len(data_instance) < 2:
        print(f"{label}: insufficient data points (need >= 2).")
        return go.Figure()

    ts = data_instance.timestamp_np.astype(np.float64)
    ts_s = (ts[:-1] - ts[0]) * _s_conversion_factor(source_time_unit)
    dt_s = np.diff(ts) * _s_conversion_factor(source_time_unit)

    if expected_frequency_hz is not None and expected_frequency_hz > 0:
        baseline_interval = 1.0 / expected_frequency_hz
    else:
        baseline_interval = float(np.median(dt_s))
    gap_threshold = gap_threshold_multiplier * baseline_interval
    gap_mask = dt_s > gap_threshold
    n_gaps = int(np.sum(gap_mask))

    freq: NDArray[float64] = 1.0 / dt_s
    total_duration_s = float(ts_s[-1] - ts_s[0])

    mean_freq = float(np.mean(freq))
    median_freq = float(np.median(freq))
    std_freq = float(np.std(freq))
    min_freq = float(np.min(freq))
    max_freq = float(np.max(freq))
    p5 = float(np.percentile(freq, 5))
    p95 = float(np.percentile(freq, 95))

    W = 10
    W_Label = 16
    print(str(data_instance))
    print(DELIMITER)
    print(f"{'Duration:':<{W_Label}} {total_duration_s:{W}.3f} s")
    print(
        f"{'Total samples:':<{W_Label}} {len(data_instance):{W}d}"
        + (
            f"    (Expected: {int(round(expected_frequency_hz * total_duration_s))} Hz)"
            if expected_frequency_hz
            else ""
        )
    )
    print(
        "Frequency (Hz)"
        + (
            f"    (Expected: {expected_frequency_hz:{W}.3f} Hz)"
            if expected_frequency_hz
            else ""
        )
    )
    print(
        f"    {'Mean:':<{W_Label - 4}} {mean_freq:{W}.3f}    {'Median:':<{W_Label - 4}} {median_freq:.3f}"
    )
    print(
        f"    {'Std dev:':<{W_Label - 4}} {std_freq:{W}.3f}    {'Min:':<{W_Label - 4}} {min_freq:.3f}    {'Max:':<{W_Label - 4}} {max_freq:.3f}"
    )
    print(
        f"    {'P5:':<{W_Label - 4}} {p5:{W}.3f}    {'P95:':<{W_Label - 4}} {p95:.3f}"
    )

    print("Gaps")
    print(f"    {'Threshold:':<{W_Label - 4}} {gap_threshold} s")
    print(f"    {'Detected:':<{W_Label - 4}} {n_gaps}")

    line_label = (
        f"Expected {expected_frequency_hz} Hz"
        if expected_frequency_hz is not None
        else None
    )
    return plot_scatter_and_histogram(
        x=ts_s,
        y=freq,
        title=f"Frequency Analysis — {label}",
        scatter_title="Instantaneous Frequency over Time",
        histogram_title="Instantaneous Frequency Distribution",
        x_label="Time (s)",
        y_label="Frequency (Hz)",
        scatter_name="Freq (Hz)",
        histogram_name="Freq (Hz)",
        hline=expected_frequency_hz,
        hline_label=line_label,
        vline=expected_frequency_hz,
        vline_label=line_label,
        font_config=font_config,
        layout_config=layout_config,
        plot_config=plot_config,
    )
