import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from ..utils.types import Timescale
from .plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    LayoutConfig,
)


def plot_parametric_curve(
    x: NDArray[float64],
    y: NDArray[float64],
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """
    Plot a 2D parametric curve as a static Plotly figure.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values.
    y : NDArray[float64]
        Y-axis values. Must have the same length as `x`.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    title : str | None, optional
        Plot title.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(size=2),
            line=dict(width=1),
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        xaxis_title=dict(text=x_label, font=dict(size=font_config.medium)),
        yaxis_title=dict(text=y_label, font=dict(size=font_config.medium)),
        showlegend=False,
        width=int(layout_config.width),
        height=int(layout_config.height),
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
    )
    return fig


def plot_parametric_curve_square(
    x: NDArray[float64],
    y: NDArray[float64],
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """
    Plot a 2D parametric curve with equal axes and a square aspect ratio.

    Automatically computes axis ranges so both axes span the same interval,
    centered on the data midpoint. Useful for curves where the spatial
    relationship between x and y must be preserved (e.g. GPS tracks).

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values.
    y : NDArray[float64]
        Y-axis values. Must have the same length as `x`.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    title : str | None, optional
        Plot title.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    go.Figure
    """
    xmid = (x.min() + x.max()) / 2
    ymid = (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) * 1.1 / 2

    fig = plot_parametric_curve(
        x, y, x_label, y_label, title, layout_config, font_config
    )
    fig.update_layout(
        width=int(layout_config.height),  # square: use height for both
        height=int(layout_config.height),
    )
    fig.update_xaxes(range=[xmid - half, xmid + half])
    fig.update_yaxes(range=[ymid - half, ymid + half])
    return fig


def plot_parametric_trimmer(
    x: NDArray[float64],
    y: NDArray[float64],
    timestamps: NDArray[float64] | None = None,
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    timestamp_unit: Timescale = Timescale.MS,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> widgets.VBox:
    """
    Interactive parametric curve trimmer with a dual-handle range slider.

    Displays a 2D parametric scatter plot alongside an ipywidgets IntRangeSlider
    that trims which points are shown. Designed for use in Jupyter notebooks.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values.
    y : NDArray[float64]
        Y-axis values. Must have the same length as `x`.
    timestamps : NDArray[float64] | None, optional
        Timestamps corresponding to each point, shown in the slider labels.
        If None, slider labels show index only.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    title : str | None, optional
        Plot title.
    timestamp_unit : Timescale, optional
        Unit for timestamp labels in the slider. Ignored if `timestamps` is None.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    ipywidgets.VBox
    """
    n = len(x)

    fig = go.FigureWidget()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(size=2),
            line=dict(width=1),
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        xaxis_title=dict(text=x_label, font=dict(size=font_config.medium)),
        yaxis_title=dict(text=y_label, font=dict(size=font_config.medium)),
        showlegend=False,
        width=int(layout_config.width),
        height=int(layout_config.height),
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
    )

    range_slider = widgets.IntRangeSlider(
        value=[0, n - 1],
        min=0,
        max=n - 1,
        step=1,
        description="Trim:",
        continuous_update=True,
        layout=widgets.Layout(width="100%"),
    )

    def _label(idx: int) -> str:
        if timestamps is not None:
            return f"index: {idx} | timestamp: {timestamps[idx]} {timestamp_unit.value}"
        return f"index: {idx}"

    start_label = widgets.Label(value=f"Start {_label(0)}")
    end_label = widgets.Label(value=f"End {_label(n - 1)}")
    label_box = widgets.HBox(
        [start_label, end_label],
        layout=widgets.Layout(justify_content="space-between", width="90%"),
    )

    def on_range_change(change: dict) -> None:
        lo, hi = change["new"]
        with fig.batch_update():
            fig.data[0].x = x[lo : hi + 1]
            fig.data[0].y = y[lo : hi + 1]
        start_label.value = f"Start {_label(lo)}"
        end_label.value = f"End {_label(hi)}"

    range_slider.observe(on_range_change, names="value")

    return widgets.VBox([fig, range_slider, label_box])
