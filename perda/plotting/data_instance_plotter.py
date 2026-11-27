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
    RIGHT_AXIS_LINE_DASH,
    SUBPLOT_HEIGHT,
    SUBPLOT_VERTICAL_SPACING,
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


def _add_row_traces(
    fig: go.Figure,
    data_instances: list[DataInstance],
    row: int,
    on_right: bool,
    timestamp_unit: Timescale,
    max_points: int | None,
) -> None:
    for di in data_instances:
        if len(di) == 0:
            raise ValueError(f"No data points in DataInstance for {di.label}")
        timestamps_s, values_np = _time_series_xy(di, timestamp_unit, max_points)
        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=values_np,
                mode="lines",
                name=di.label,
                legendgroup=str(row),
                line=dict(dash=RIGHT_AXIS_LINE_DASH) if on_right else None,
            ),
            row=row,
            col=1,
            secondary_y=on_right,
        )


def _axis_titles_per_subplot(titles: list[str | None] | None, n_rows: int) -> list[str | None]:
    if titles is None:
        return [None] * n_rows
    if len(titles) != n_rows:
        raise ValueError(f"Expected {n_rows} axis titles (one per subplot), got {len(titles)}")
    return titles


def plot_time_series(
    rows: list[list[DataInstance]],
    timestamp_unit: Timescale,
    right_rows: list[list[DataInstance]] | None = None,
    title: str | None = None,
    y_axis_titles: list[str | None] | None = None,
    right_y_axis_titles: list[str | None] | None = None,
    show_legend: bool = True,
    vlines: list[float] | None = None,
    max_points: int | None = None,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int | None = None,
    vertical_spacing: float = SUBPLOT_VERTICAL_SPACING,
) -> go.Figure:
    """
    Plot DataInstances against time as vertically stacked subplots sharing one time axis.

    Each subplot has a left y-axis and, optionally, an independent right y-axis for signals with different units or
    scales. A single row gives an ordinary one-panel plot. Zooming the time axis zooms every subplot together.

    Parameters
    ----------
    rows : list[list[DataInstance]]
        Outer list is the subplots, top to bottom. Inner list is the signals drawn against that subplot's left y-axis
        (solid lines). An inner list may be empty if that subplot only uses its right axis.
    timestamp_unit : Timescale
        Unit of the raw timestamps in the DataInstances. They are converted to seconds for the x-axis. Pass
        ``aly.data.timestamp_unit``.
    right_rows : list[list[DataInstance]] | None, optional
        Signals drawn against each subplot's right y-axis (dashed lines), one inner list per subplot in the same order
        as ``rows``. An empty inner list means that subplot has no right axis. Default is None (no right axes anywhere).
    title : str | None, optional
        Title shown above the whole figure. Default is None (no title).
    y_axis_titles : list[str | None] | None, optional
        Left y-axis label for each subplot, e.g. ``["Speed (mph)", None]``. ``None`` entries leave that axis unlabeled.
        Default is None (no labels).
    right_y_axis_titles : list[str | None] | None, optional
        Right y-axis label for each subplot. Ignored for subplots without right-axis signals. Default is None (no
        labels).
    show_legend : bool, optional
        Whether to show the legend listing each signal. Default is True.
    vlines : list[float] | None, optional
        Times in seconds at which to draw dashed vertical marker lines on every subplot. Default is None (no lines).
    max_points : int | None, optional
        Maximum number of points drawn per signal. Longer signals are evenly downsampled (every Nth point), which speeds
        up rendering of large logs. Default is None (draw every point).
    log_x : bool, optional
        Use a logarithmic time (x) axis. Default is False (linear).
    log_y : bool, optional
        Use logarithmic scaling on every y-axis, left and right. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int | None, optional
        Figure height in pixels. Default is None, which gives PLOT_HEIGHT (700) for one subplot and SUBPLOT_HEIGHT (350)
        per subplot otherwise.
    vertical_spacing : float, optional
        Gap between subplots as a fraction of the figure height (0 to 1). Default is SUBPLOT_VERTICAL_SPACING (0.12).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with
        ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling
        ``fig.show()``. Target one subplot with ``row=i, col=1``, and its right axis with ``secondary_y=True``, e.g.
        ``fig.update_yaxes(type="log", row=2, col=1, secondary_y=True)``.

    Examples
    --------
    >>> from perda.plotting import plot_time_series
    >>> fig = plot_time_series(
    ...     rows=[[speed_di], [roll_di, pitch_di]],
    ...     right_rows=[[], [steering_di]],
    ...     y_axis_titles=["Speed (m/s)", "Attitude (deg)"],
    ...     right_y_axis_titles=[None, "Steering (deg)"],
    ...     timestamp_unit=aly.data.timestamp_unit,
    ... )
    """
    n_rows = len(rows)
    if right_rows is None:
        right_rows = [[] for _ in rows]
    if len(right_rows) != n_rows:
        raise ValueError(f"right_rows has {len(right_rows)} entries but rows has {n_rows}")
    if not any(rows) and not any(right_rows):
        raise ValueError("No data instances provided for plotting")

    left_titles = _axis_titles_per_subplot(y_axis_titles, n_rows)
    right_titles = _axis_titles_per_subplot(right_y_axis_titles, n_rows)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing if n_rows > 1 else 0.0,
        specs=[[{"secondary_y": bool(right_dis)}] for right_dis in right_rows],
    )

    for row, (left_dis, right_dis) in enumerate(zip(rows, right_rows), start=1):
        _add_row_traces(fig, left_dis, row, False, timestamp_unit, max_points)
        _add_row_traces(fig, right_dis, row, True, timestamp_unit, max_points)
        fig.update_yaxes(title_text=left_titles[row - 1], row=row, col=1)
        if right_dis:
            fig.update_yaxes(title_text=right_titles[row - 1], row=row, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    fig.update_xaxes(
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        type="log" if log_x else "linear",
    )
    fig.update_yaxes(
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        type="log" if log_y else "linear",
    )

    if height is None:
        height = PLOT_HEIGHT if n_rows == 1 else SUBPLOT_HEIGHT * n_rows

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
