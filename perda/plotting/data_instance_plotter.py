from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core_data_structures.data_instance import DataInstance
from ..units import Timescale
from .plotting_constants import *


def _timestamps_to_seconds(
    timestamp_np,
    timestamp_unit: Timescale,
):
    timestamps = timestamp_np.astype(float)
    if timestamp_unit == Timescale.US:
        return timestamps / 1e6
    if timestamp_unit == Timescale.MS:
        return timestamps / 1e3
    return timestamps


def _add_vlines(
    fig: go.Figure,
    vlines: List[float],
    vline_config: "VLineConfig",
) -> None:
    """
    Add vertical lines to an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Target figure
    vlines : List[float]
        X-axis positions (in seconds) where vertical lines are drawn
    vline_config : VLineConfig
        Visual configuration for the vertical lines
    """
    for x in vlines:
        fig.add_vline(
            x=x,
            line_dash=vline_config.dash,
            line_color=vline_config.color,
            line_width=vline_config.width,
            opacity=vline_config.opacity,
        )


def plot_single_axis(
    data_instances: List[DataInstance],
    title: str | None = None,
    y_axis_title: str | None = None,
    show_legend: bool = True,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    timestamp_unit: Timescale = Timescale.MS,
    vlines: List[float] | None = None,
    vline_config: "VLineConfig" = DEFAULT_VLINE_CONFIG,
) -> go.Figure:
    """
    Plot one or more DataInstances on a single y-axis using Plotly.

    Parameters
    ----------
    data_instances : List[DataInstance]
        List of DataInstance objects to plot
    title : str | None, optional
        Title for the entire plot.
    y_axis_title : str | None, optional
        Label for the y-axis.
    show_legend : bool, optional
        Whether to show plot legends. Default is True
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional
    timestamp_unit : Timescale, optional
        Timestamp unit in the underlying data. Converted to seconds for x-axis display.
    vlines : List[float] | None, optional
        X-axis positions (in seconds) where vertical lines are drawn. Default is None.
    vline_config : VLineConfig, optional
        Visual configuration for the vertical lines. Default is DEFAULT_VLINE_CONFIG.

    Returns
    -------
    go.Figure
    """
    if not data_instances:
        print("Warning: No data instances provided for plotting")
        return

    fig = go.Figure()

    for di in data_instances:
        if len(di) == 0:
            print(f"Warning: No data points in DataInstance for {di.label}")
            continue

        # Convert timestamps from the log unit to seconds for plotting.
        timestamps_s = _timestamps_to_seconds(di.timestamp_np, timestamp_unit)

        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=di.value_np,
                mode="lines",
                name=di.label,
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
        xaxis_title=dict(text="Time (s)", font=dict(size=font_config.medium)),
        yaxis_title=dict(text=y_axis_title, font=dict(size=font_config.medium)),
        showlegend=show_legend,
        hovermode="x unified",
        width=layout_config.width,
        height=layout_config.height,
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
        legend=dict(font=dict(size=font_config.small)),
    )

    if vlines:
        _add_vlines(fig, vlines, vline_config)

    return fig


def plot_dual_axis(
    left_data_instances: List[DataInstance],
    right_data_instances: List[DataInstance],
    title: str | None = None,
    left_y_axis_title: str | None = None,
    right_y_axis_title: str | None = None,
    show_legend: bool = True,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    timestamp_unit: Timescale = Timescale.MS,
    vlines: List[float] | None = None,
    vline_config: VLineConfig = DEFAULT_VLINE_CONFIG,
) -> go.Figure:
    """
    Plot DataInstances on dual y-axes using Plotly.

    Parameters
    ----------
    left_data_instances : List[DataInstance]
        List of DataInstance objects to plot on the left y-axis
    right_data_instances : List[DataInstance]
        List of DataInstance objects to plot on the right y-axis
    title : str | None, optional
        Title for the entire plot.
    left_y_axis_title : str | None, optional
        Label for the left y-axis.
    right_y_axis_title : str | None, optional
        Label for the right y-axis.
    show_legend : bool, optional
        Whether to show plot legends. Default is True
    font_config : FontConfig, optional
        Font configuration for plot elements. Default is DEFAULT_FONT_CONFIG
    layout_config : LayoutConfig, optional
        Layout configuration for plot dimensions. Default is DEFAULT_LAYOUT_CONFIG
    timestamp_unit : Timescale, optional
        Timestamp unit in the underlying data. Converted to seconds for x-axis display.
    vlines : List[float] | None, optional
        X-axis positions (in seconds) where vertical lines are drawn. Default is None.
    vline_config : VLineConfig, optional
        Visual configuration for the vertical lines. Default is DEFAULT_VLINE_CONFIG.

    Returns
    -------
    go.Figure
    """
    if not left_data_instances and not right_data_instances:
        print("Warning: No data instances provided for plotting")
        return

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot left axis data
    for di in left_data_instances:
        if len(di) == 0:
            print(f"Warning: No data points in DataInstance for {di.label}")
            continue

        # Convert timestamps from the log unit to seconds for plotting.
        timestamps_s = _timestamps_to_seconds(di.timestamp_np, timestamp_unit)

        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=di.value_np,
                mode="lines",
                name=di.label,
            ),
            secondary_y=False,
        )

    # Plot right axis data
    for di in right_data_instances:
        if len(di) == 0:
            print(f"Warning: No data points in DataInstance for {di.label}")
            continue

        # Convert timestamps from the log unit to seconds for plotting.
        timestamps_s = _timestamps_to_seconds(di.timestamp_np, timestamp_unit)

        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=di.value_np,
                mode="lines",
                name=di.label,
                line=dict(dash="dash"),
            ),
            secondary_y=True,
        )

    # Set axis titles
    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=font_config.medium),
        tickfont=dict(size=font_config.small),
    )
    fig.update_yaxes(
        title_text=left_y_axis_title,
        title_font=dict(size=font_config.medium),
        tickfont=dict(size=font_config.small),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=right_y_axis_title,
        title_font=dict(size=font_config.medium),
        tickfont=dict(size=font_config.small),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        showlegend=show_legend,
        hovermode="x unified",
        width=layout_config.width,
        height=layout_config.height,
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        legend=dict(font=dict(size=font_config.small)),
    )

    if vlines:
        _add_vlines(fig, vlines, vline_config)

    return fig
