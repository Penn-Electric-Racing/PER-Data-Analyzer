from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analyzer.data_instance import DataInstance
from .plotting_constants import *


def plot_single_axis(
    data_instances: List[DataInstance],
    title: str | None = None,
    y_axis_title: str | None = None,
    show_legend: bool = True,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
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

    Returns
    -------
    None
    """
    if not data_instances:
        print("Warning: No data instances provided for plotting")
        return

    fig = go.Figure()

    for di in data_instances:
        if len(di) == 0:
            print(f"Warning: No data points in DataInstance for {di.label}")
            continue

        # Convert timestamps from milliseconds to seconds
        timestamps_s = di.timestamp_np.astype(float) / 1000.0

        fig.add_trace(
            go.Scatter(
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

        # Convert timestamps from milliseconds to seconds
        timestamps_s = di.timestamp_np.astype(float) / 1000.0

        fig.add_trace(
            go.Scatter(
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

        # Convert timestamps from milliseconds to seconds
        timestamps_s = di.timestamp_np.astype(float) / 1000.0

        fig.add_trace(
            go.Scatter(
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

    return fig
