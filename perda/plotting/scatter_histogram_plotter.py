from numpy import float64
from numpy.typing import NDArray
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .plotting_constants import *


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
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    plot_config: ScatterHistogramPlotConfig = DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG,
) -> go.Figure:
    """
    Build a figure with a scatter plot over time and a histogram of the same values.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values for the scatter plot (e.g. time).
    y : NDArray[float64]
        Y-axis values shared between the scatter plot and histogram.
    title : str
        Overall figure title.
    scatter_title : str
        Subplot title for the scatter plot.
    histogram_title : str
        Subplot title for the histogram.
    x_label : str
        X-axis label for the scatter plot.
    y_label : str
        Y-axis label for the scatter plot and X-axis label for the histogram.
    scatter_name : str
        Legend name for the scatter trace.
    histogram_name : str
        Legend name for the histogram trace.
    hline : float | None, optional
        Y value at which to draw a horizontal reference line on the scatter plot.
        Default is None.
    hline_label : str | None, optional
        Annotation text for the horizontal reference line. Default is None.
    vline : float | None, optional
        X value at which to draw a vertical reference line on the histogram.
        Default is None.
    vline_label : str | None, optional
        Annotation text for the vertical reference line. Default is None.
    font_config : FontConfig, optional
        Font sizes for plot elements.
    layout_config : LayoutConfig, optional
        Plot dimensions and margins.
    plot_config : ScatterHistogramPlotConfig, optional
        Colors and histogram bin count.

    Returns
    -------
    go.Figure
        Plotly figure with scatter and histogram subplots.
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
            line=dict(color=plot_config.color_scatter, width=1),
        ),
        row=1,
        col=1,
    )

    if hline is not None:
        fig.add_hline(
            y=hline,
            line=dict(color=plot_config.color_line, width=1.5, dash="dash"),
            annotation_text=hline_label,
            annotation_font_size=font_config.small,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=y,
            nbinsx=plot_config.histogram_bins,
            name=histogram_name,
            marker_color=plot_config.color_histogram,
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    if vline is not None:
        fig.add_vline(
            x=vline,
            line=dict(color=plot_config.color_line, width=1.5, dash="dash"),
            annotation_text=vline_label,
            annotation_font_size=font_config.small,
            row=2,
            col=1,
        )

    fig.update_xaxes(
        title_text=x_label, row=1, col=1, title_font_size=font_config.medium
    )
    fig.update_yaxes(
        title_text=y_label, row=1, col=1, title_font_size=font_config.medium
    )
    fig.update_xaxes(
        title_text=y_label, row=2, col=1, title_font_size=font_config.medium
    )
    fig.update_yaxes(
        title_text="Count", row=2, col=1, title_font_size=font_config.medium
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        width=layout_config.width,
        height=layout_config.height,
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        showlegend=True,
    )

    return fig
