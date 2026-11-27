from numpy import float64
from numpy.typing import NDArray
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    HISTOGRAM_BINS,
    HISTOGRAM_COLOR,
    MARGIN,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    REFERENCE_LINE_COLOR,
    SCATTER_COLOR,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def plot_scatter_and_histogram(
    x: NDArray[float64],
    y: NDArray[float64],
    title: str,
    scatter_title: str,
    histogram_title: str,
    x_label: str,
    y_label: str,
    scatter_name: str,
    histogram_name: str,
    hline: float | None = None,
    hline_label: str | None = None,
    vline: float | None = None,
    vline_label: str | None = None,
    histogram_bins: int = HISTOGRAM_BINS,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Build a two-panel figure: a line plot of ``y`` against ``x`` on top, and a histogram of the same ``y`` values below.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values for the top line plot, e.g. time in seconds.
    y : NDArray[float64]
        Values shown in both panels: as the y-axis of the top plot and as the binned quantity of the histogram.
    title : str
        Title shown above the whole figure.
    scatter_title : str
        Title of the top (line plot) panel.
    histogram_title : str
        Title of the bottom (histogram) panel.
    x_label : str
        X-axis label of the top panel.
    y_label : str
        Y-axis label of the top panel, also used as the x-axis label of the histogram.
    scatter_name : str
        Legend name for the top panel's line.
    histogram_name : str
        Legend name for the histogram bars.
    hline : float | None, optional
        Y value at which to draw a dashed horizontal reference line on the top panel. Default is None (no line).
    hline_label : str | None, optional
        Text shown next to the horizontal reference line. Default is None (no text).
    vline : float | None, optional
        X value at which to draw a dashed vertical reference line on the histogram. Default is None (no line).
    vline_label : str | None, optional
        Text shown next to the vertical reference line. Default is None (no text).
    histogram_bins : int, optional
        Maximum number of histogram bins. Default is HISTOGRAM_BINS (80).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, log axes, ...) with
        ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling
        ``fig.show()``. Target one panel with ``row=1`` (line plot) or ``row=2`` (histogram) and ``col=1``.

    Examples
    --------
    >>> import numpy as np
    >>> from perda.plotting import plot_scatter_and_histogram
    >>> from perda.units import to_seconds
    >>> di = aly.data["bms.pack.voltage"]
    >>> fig = plot_scatter_and_histogram(
    ...     x=to_seconds(di.timestamp_np, aly.data.timestamp_unit),
    ...     y=di.value_np,
    ...     title="Pack Voltage",
    ...     scatter_title="Over time",
    ...     histogram_title="Distribution",
    ...     x_label="Time (s)",
    ...     y_label="Voltage (V)",
    ...     scatter_name="Voltage",
    ...     histogram_name="Count",
    ...     hline=float(np.mean(di.value_np)),
    ...     hline_label="Mean",
    ... )
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(scatter_title, histogram_title),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            name=scatter_name,
            line=dict(color=SCATTER_COLOR, width=1),
        ),
        row=1,
        col=1,
    )

    if hline is not None:
        fig.add_hline(
            y=hline,
            line=dict(color=REFERENCE_LINE_COLOR, width=1.5, dash="dash"),
            annotation_text=hline_label,
            annotation_font_size=FONT_SIZE_SMALL,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=y,
            nbinsx=histogram_bins,
            name=histogram_name,
            marker_color=HISTOGRAM_COLOR,
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    if vline is not None:
        fig.add_vline(
            x=vline,
            line=dict(color=REFERENCE_LINE_COLOR, width=1.5, dash="dash"),
            annotation_text=vline_label,
            annotation_font_size=FONT_SIZE_SMALL,
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text=x_label, row=1, col=1, title_font_size=FONT_SIZE_MEDIUM)
    fig.update_yaxes(title_text=y_label, row=1, col=1, title_font_size=FONT_SIZE_MEDIUM)
    fig.update_xaxes(title_text=y_label, row=2, col=1, title_font_size=FONT_SIZE_MEDIUM)
    fig.update_yaxes(title_text="Count", row=2, col=1, title_font_size=FONT_SIZE_MEDIUM)

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        showlegend=True,
    )

    return fig
