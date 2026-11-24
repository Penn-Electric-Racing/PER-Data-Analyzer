import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from ..core_data_structures.data_instance import DataInstance
from ..units import Timescale, to_seconds
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
    VLINE_COLOR,
    VLINE_DASH,
    VLINE_OPACITY,
    VLINE_WIDTH,
)


def _add_vlines(fig: go.Figure, vlines: list[float]) -> None:
    for x in vlines:
        fig.add_vline(
            x=x,
            line_dash=VLINE_DASH,
            line_color=VLINE_COLOR,
            line_width=VLINE_WIDTH,
            opacity=VLINE_OPACITY,
        )


def _time_series_xy(
    di: DataInstance, timestamp_unit: Timescale, max_points: int | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_points = len(di)
    if max_points is not None and n_points > max_points and max_points > 0:
        step = (n_points + max_points - 1) // max_points
        return to_seconds(di.timestamp_np[::step], timestamp_unit), di.value_np[::step]
    return to_seconds(di.timestamp_np, timestamp_unit), di.value_np


def plot_single_axis(
    data_instances: list[DataInstance],
    timestamp_unit: Timescale,
    title: str | None = None,
    y_axis_title: str | None = None,
    show_legend: bool = True,
    vlines: list[float] | None = None,
    max_points: int | None = None,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Plot one or more DataInstances against time on a single shared y-axis.

    Parameters
    ----------
    data_instances : list[DataInstance]
        Signals to plot. Each becomes one line, named by its ``label`` in the legend.
    timestamp_unit : Timescale
        Unit of the raw timestamps in the DataInstances. They are converted to seconds for the x-axis. Pass ``aly.data.timestamp_unit``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    y_axis_title : str | None, optional
        Y-axis label, e.g. ``"Speed (mph)"``. Default is None (no label).
    show_legend : bool, optional
        Whether to show the legend listing each signal. Default is True.
    vlines : list[float] | None, optional
        Times in seconds at which to draw dashed vertical marker lines. Default is None (no lines).
    max_points : int | None, optional
        Maximum number of points drawn per signal. Longer signals are evenly downsampled (every Nth point), which speeds up rendering of large logs. Default is None (draw every point).
    log_x : bool, optional
        Use a logarithmic time (x) axis. Default is False (linear).
    log_y : bool, optional
        Use a logarithmic y-axis. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``.

    Examples
    --------
    >>> from perda.plotting import plot_single_axis
    >>> fig = plot_single_axis(
    ...     [speed_di, torque_di],
    ...     title="Speed & Torque",
    ...     y_axis_title="Value",
    ...     timestamp_unit=aly.data.timestamp_unit,
    ...     log_y=True,
    ... )
    >>> fig.update_layout(font_size=16)  # further customization
    """
    if not data_instances:
        print("Warning: No data instances provided for plotting")
        return

    fig = go.Figure()

    for di in data_instances:
        if len(di) == 0:
            print(f"Warning: No data points in DataInstance for {di.label}")
            continue

        timestamps_s, values_np = _time_series_xy(di, timestamp_unit, max_points)
        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=values_np,
                mode="lines",
                name=di.label,
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        xaxis_title=dict(text="Time (s)", font=dict(size=FONT_SIZE_MEDIUM)),
        yaxis_title=dict(text=y_axis_title, font=dict(size=FONT_SIZE_MEDIUM)),
        showlegend=show_legend,
        hovermode="x unified",
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        xaxis=dict(
            tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_x else "linear"
        ),
        yaxis=dict(
            tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_y else "linear"
        ),
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )

    if vlines:
        _add_vlines(fig, vlines)

    return fig


def plot_dual_axis(
    left_data_instances: list[DataInstance],
    right_data_instances: list[DataInstance],
    timestamp_unit: Timescale,
    title: str | None = None,
    left_y_axis_title: str | None = None,
    right_y_axis_title: str | None = None,
    show_legend: bool = True,
    vlines: list[float] | None = None,
    max_points: int | None = None,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Plot DataInstances against time with two independent y-axes (left and right).

    Useful for signals with different units or scales. Right-axis signals are drawn dashed.

    Parameters
    ----------
    left_data_instances : list[DataInstance]
        Signals plotted against the left y-axis (solid lines).
    right_data_instances : list[DataInstance]
        Signals plotted against the right y-axis (dashed lines).
    timestamp_unit : Timescale
        Unit of the raw timestamps in the DataInstances. They are converted to seconds for the x-axis. Pass ``aly.data.timestamp_unit``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    left_y_axis_title : str | None, optional
        Left y-axis label, e.g. ``"Speed (mph)"``. Default is None (no label).
    right_y_axis_title : str | None, optional
        Right y-axis label, e.g. ``"Torque (Nm)"``. Default is None (no label).
    show_legend : bool, optional
        Whether to show the legend listing each signal. Default is True.
    vlines : list[float] | None, optional
        Times in seconds at which to draw dashed vertical marker lines. Default is None (no lines).
    max_points : int | None, optional
        Maximum number of points drawn per signal. Longer signals are evenly downsampled (every Nth point), which speeds up rendering of large logs. Default is None (draw every point).
    log_x : bool, optional
        Use a logarithmic time (x) axis. Default is False (linear).
    log_y : bool, optional
        Use logarithmic scaling on both the left and right y-axes. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``. To log-scale only one side, use ``fig.update_yaxes(type="log", secondary_y=True)``.

    Examples
    --------
    >>> from perda.plotting import plot_dual_axis
    >>> fig = plot_dual_axis(
    ...     [speed_di],
    ...     [torque_di],
    ...     left_y_axis_title="Speed (mph)",
    ...     right_y_axis_title="Torque (Nm)",
    ...     timestamp_unit=aly.data.timestamp_unit,
    ... )
    """
    if not left_data_instances and not right_data_instances:
        print("Warning: No data instances provided for plotting")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data_instances, on_right in (
        (left_data_instances, False),
        (right_data_instances, True),
    ):
        for di in data_instances:
            if len(di) == 0:
                print(f"Warning: No data points in DataInstance for {di.label}")
                continue

            timestamps_s, values_np = _time_series_xy(di, timestamp_unit, max_points)
            fig.add_trace(
                go.Scattergl(
                    x=timestamps_s,
                    y=values_np,
                    mode="lines",
                    name=di.label,
                    line=dict(dash="dash") if on_right else None,
                ),
                secondary_y=on_right,
            )

    y_axis_type = "log" if log_y else "linear"
    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        type="log" if log_x else "linear",
    )
    fig.update_yaxes(
        title_text=left_y_axis_title,
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        type=y_axis_type,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=right_y_axis_title,
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        type=y_axis_type,
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        showlegend=show_legend,
        hovermode="x unified",
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )

    if vlines:
        _add_vlines(fig, vlines)

    return fig
