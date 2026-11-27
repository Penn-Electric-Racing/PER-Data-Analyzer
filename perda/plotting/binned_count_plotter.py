import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def count_per_bin(
    values: npt.NDArray[np.float64],
    bin_start: float,
    bin_width: float,
    n_bins: int,
) -> npt.NDArray[np.int64]:
    """Count how many values fall into each of ``n_bins`` fixed-width bins starting at ``bin_start``.

    Values outside the bin range are clamped into the first or last bin.

    Parameters
    ----------
    values : npt.NDArray[np.float64]
        Values to count, e.g. event timestamps in seconds.
    bin_start : float
        Left edge of the first bin.
    bin_width : float
        Width of every bin, in the same unit as ``values``.
    n_bins : int
        Total number of bins.

    Returns
    -------
    npt.NDArray[np.int64]
        Array of length ``n_bins`` with the count in each bin.

    Examples
    --------
    >>> count_per_bin(np.array([0.1, 0.2, 1.5]), bin_start=0.0, bin_width=1.0, n_bins=2)
    array([2, 1])
    """
    if values.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    indices = ((values - bin_start) / bin_width).clip(0, n_bins - 1).astype(np.intp)
    return np.bincount(indices, minlength=n_bins).astype(np.int64)


def plot_binned_counts(
    values_by_series: dict[str, npt.NDArray[np.float64]],
    bin_width: float,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = "Count",
    show_legend: bool = True,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Count how many values of each series fall into fixed-width bins, and draw the counts as overlaid bar charts.

    Every series is binned on the same grid, which spans the smallest to largest value across all series, so bars line
    up for comparison. Typical use: how many events of each kind happened in each time window. Bars are drawn in the
    order given, so put the largest (background) series first.

    Parameters
    ----------
    values_by_series : dict[str, npt.NDArray[np.float64]]
        Maps each series' legend name to the values to count, e.g. ``{"Errors": error_times_s, "Warnings":
        warning_times_s}``. Empty arrays are allowed and give all-zero bars.
    bin_width : float
        Width of each bin (bar), in the same unit as the values, e.g. seconds. Smaller bins show finer detail but
        noisier counts.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    x_label : str | None, optional
        X-axis label, e.g. ``"Time (s)"``. Default is None (no label).
    y_label : str | None, optional
        Y-axis label. Default is ``"Count"``.
    show_legend : bool, optional
        Whether to show the legend listing each series. Default is True.
    log_y : bool, optional
        Use a logarithmic count (y) axis, useful when some series are much rarer than others. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure with one bar trace per series, named by its key. Change anything not exposed here with
        ``fig.update_layout`` or ``fig.update_traces`` before calling ``fig.show()``. Style one series with
        ``fig.update_traces(marker_color="gray", opacity=0.4, selector=dict(name="Errors"))``. Returns an empty figure
        if every series is empty.

    Examples
    --------
    >>> from perda.plotting import plot_binned_counts
    >>> fig = plot_binned_counts(
    ...     {"All messages": all_times_s, "Dropped": dropped_times_s},
    ...     bin_width=1.0,
    ...     x_label="Time (s)",
    ...     log_y=True,
    ... )
    """
    non_empty = [values for values in values_by_series.values() if values.size > 0]
    if not non_empty:
        print("No values to plot.")
        return go.Figure()

    bin_start = float(min(values.min() for values in non_empty))
    bin_end = float(max(values.max() for values in non_empty))
    n_bins = max(int((bin_end - bin_start) / bin_width) + 1, 1)
    bin_midpoints = np.arange(n_bins, dtype=np.float64) * bin_width + bin_start + bin_width * 0.5

    fig = go.Figure()
    for name, values in values_by_series.items():
        fig.add_trace(
            go.Bar(
                x=bin_midpoints,
                y=count_per_bin(values, bin_start, bin_width, n_bins),
                name=name,
            )
        )

    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        xaxis_title=dict(text=x_label, font=dict(size=FONT_SIZE_MEDIUM)),
        yaxis_title=dict(text=y_label, font=dict(size=FONT_SIZE_MEDIUM)),
        showlegend=show_legend,
        hovermode="x unified",
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        xaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_y else "linear"),
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )

    return fig
