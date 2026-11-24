import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale, to_seconds
from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    MAX_DISPLAY_RESOLUTION_HZ,
    PLOT_BGCOLOR,
    PLOT_WIDTH,
    SUBPLOT_HEIGHT,
    SUBPLOT_VERTICAL_SPACING,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def data_instance_subplots(
    rows: list[list[DataInstance]],
    timestamp_unit: Timescale,
    title: str | None = None,
    row_y_labels: list[str | None] | None = None,
    show_legend: bool = True,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int | None = None,
    vertical_spacing: float = SUBPLOT_VERTICAL_SPACING,
) -> go.Figure:
    """Plot groups of DataInstances as vertically stacked subplots sharing one time axis.

    Each element of ``rows`` becomes one subplot row. Multiple DataInstances in the same row are overlaid on that row's y-axis, which is useful for comparing signals that share the same units or scale. Zooming the time axis zooms every row together.

    Parameters
    ----------
    rows : list[list[DataInstance]]
        Outer list is the subplot rows, top to bottom. Inner list is the signals overlaid within that row.
    timestamp_unit : Timescale
        Unit of the raw timestamps in the DataInstances. They are converted to seconds for the x-axis. Pass ``aly.data.timestamp_unit``.
    title : str | None, optional
        Title shown above the whole figure. Default is None (no title).
    row_y_labels : list[str | None] | None, optional
        Y-axis label for each row, in the same order as ``rows``. A ``None`` entry falls back to the signal labels in that row. Default is None (every row uses its signal labels).
    show_legend : bool, optional
        Whether to show the legend listing each signal. Default is True.
    log_x : bool, optional
        Use a logarithmic time (x) axis. Default is False (linear).
    log_y : bool, optional
        Use logarithmic y-axes on every row. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int | None, optional
        Figure height in pixels. Default is None, which gives SUBPLOT_HEIGHT (350) pixels per row.
    vertical_spacing : float, optional
        Gap between rows as a fraction of the figure height (0 to 1). Default is SUBPLOT_VERTICAL_SPACING (0.12).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``. Target one row with ``row=i, col=1``.

    Examples
    --------
    >>> from perda.plotting import data_instance_subplots
    >>> fig = data_instance_subplots(
    ...     rows=[[speed_di], [torque_di, motor_di]],
    ...     title="Run Overview",
    ...     row_y_labels=["Speed (mph)", "Torque / Motor"],
    ...     timestamp_unit=aly.data.timestamp_unit,
    ... )
    """
    if not rows:
        print("Warning: No rows provided for subplot figure")
        return go.Figure()

    n = len(rows)

    subplot_titles = []
    for i, row_dis in enumerate(rows):
        if row_y_labels and i < len(row_y_labels) and row_y_labels[i] is not None:
            subplot_titles.append(row_y_labels[i])  # type: ignore[arg-type]
        else:
            labels = [di.label for di in row_dis if di.label]
            subplot_titles.append(", ".join(labels) if labels else "")

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    for row_idx, row_dis in enumerate(rows, start=1):
        for di in row_dis:
            if len(di) == 0:
                print(f"Warning: No data points in DataInstance for {di.label}")
                continue

            timestamps_s = to_seconds(
                di.timestamp_np.astype(np.float64), timestamp_unit
            )

            fig.add_trace(
                go.Scattergl(
                    x=timestamps_s,
                    y=di.value_np,
                    mode="lines",
                    name=di.label,
                    legendgroup=str(row_idx),
                ),
                row=row_idx,
                col=1,
            )

        if row_y_labels and row_idx - 1 < len(row_y_labels):
            y_label = row_y_labels[row_idx - 1]
        else:
            labels = [di.label for di in row_dis if di.label]
            y_label = ", ".join(labels) if labels else None

        fig.update_yaxes(
            title_text=y_label,
            title_font=dict(size=FONT_SIZE_MEDIUM),
            tickfont=dict(size=FONT_SIZE_SMALL),
            row=row_idx,
            col=1,
        )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        row=n,
        col=1,
    )
    fig.update_xaxes(type="log" if log_x else "linear")
    fig.update_yaxes(type="log" if log_y else "linear")

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        height=height if height is not None else SUBPLOT_HEIGHT * n,
        width=width,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        showlegend=show_legend,
        hovermode="x unified",
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )

    return fig


def stride_downsample(
    timestamps_s: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    max_display_resolution: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Keep every Nth sample so the effective sample rate is at most ``max_display_resolution``.

    Parameters
    ----------
    timestamps_s : npt.NDArray[np.float64]
        Sample times in seconds.
    values : npt.NDArray[np.float64]
        Sample values, same length as ``timestamps_s``.
    max_display_resolution : float
        Maximum sample rate to keep, in Hz. Data already at or below this rate is returned unchanged.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Downsampled ``(timestamps_s, values)``.

    Examples
    --------
    >>> from perda.plotting import stride_downsample
    >>> ts_small, values_small = stride_downsample(time_s, di.value_np, 1000.0)
    """
    n = len(timestamps_s)
    if n <= 1:
        return timestamps_s, values
    duration_s = float(timestamps_s[-1] - timestamps_s[0])
    if duration_s <= 0:
        return timestamps_s, values
    actual_hz = n / duration_s
    if actual_hz <= max_display_resolution:
        print(
            "Data resolution is below the max display resolution, skipping downsampling."
        )
        return timestamps_s, values
    step = max(1, int(round(actual_hz / max_display_resolution)))
    idx = np.arange(0, n, step)
    return timestamps_s[idx], values[idx]


def plot_multi_log_subplots(
    logs: list[SingleRunData],
    var_names: list[str],
    timestamp_unit: Timescale,
    row_y_labels: list[str | None] | None = None,
    title: str | None = None,
    max_display_resolution_hz: float = MAX_DISPLAY_RESOLUTION_HZ,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int | None = None,
    vertical_spacing: float = SUBPLOT_VERTICAL_SPACING,
) -> go.Figure:
    """Compare the same variables across multiple logs, one stacked subplot per variable.

    Each subplot overlays one variable from every log. Variables missing from a log are skipped for that log.

    Parameters
    ----------
    logs : list[SingleRunData]
        Logs to compare, e.g. ``[aly1.data, aly2.data]``.
    var_names : list[str]
        Variable names to plot, one subplot row per name, top to bottom.
    timestamp_unit : Timescale
        Unit of the raw timestamps in the logs. They are converted to seconds for the x-axis. Pass ``aly.data.timestamp_unit``.
    row_y_labels : list[str | None] | None, optional
        Y-axis label for each row, in the same order as ``var_names``. Ignored unless its length matches ``var_names``. Default is None (no y-axis labels; subplot titles show the variable names).
    title : str | None, optional
        Title shown above the whole figure. Default is None (no title).
    max_display_resolution_hz : float, optional
        Signals sampled faster than this rate (Hz) are downsampled before plotting to keep the figure responsive. Default is MAX_DISPLAY_RESOLUTION_HZ (50).
    log_x : bool, optional
        Use a logarithmic time (x) axis. Default is False (linear).
    log_y : bool, optional
        Use logarithmic y-axes on every row. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int | None, optional
        Figure height in pixels. Default is None, which gives SUBPLOT_HEIGHT (350) pixels per row.
    vertical_spacing : float, optional
        Gap between rows as a fraction of the figure height (0 to 1). Default is SUBPLOT_VERTICAL_SPACING (0.12).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``.

    Examples
    --------
    >>> from perda.plotting import plot_multi_log_subplots
    >>> fig = plot_multi_log_subplots(
    ...     logs=[aly1.data, aly2.data],
    ...     var_names=["pcm.wheelSpeeds.frontLeft", "ludwig.steeringWheel.angle", "pcm.pedals.brakePressure.front"],
    ...     title="Speed Comparison",
    ...     timestamp_unit=aly1.data.timestamp_unit,
    ... )
    """
    if not logs or not var_names:
        print("No logs or variable names provided.")
        return go.Figure()

    n_vars = len(var_names)

    fig = make_subplots(
        rows=n_vars,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=var_names,
    )

    for var_idx, var_name in enumerate(var_names, start=1):
        first_valid_log_idx = 0
        for log_idx, srd in enumerate(logs):
            if var_name not in srd or len(srd[var_name]) == 0:
                continue
            elif first_valid_log_idx == 0:
                first_valid_log_idx = log_idx

            di = srd[var_name]
            timestamps_s = to_seconds(
                di.timestamp_np.astype(np.float64), timestamp_unit
            )
            values = di.value_np
            if max_display_resolution_hz:
                timestamps_s, values = stride_downsample(
                    timestamps_s, values, max_display_resolution_hz
                )

            fig.add_trace(
                go.Scattergl(
                    x=timestamps_s,
                    y=values,
                    mode="lines",
                    line=dict(width=1),
                    name=var_name,
                    legendgroup="Log " + str(log_idx),
                    showlegend=(log_idx == first_valid_log_idx),
                ),
                row=var_idx,
                col=1,
            )

    if row_y_labels and len(row_y_labels) == n_vars:
        for var_idx, y_label in enumerate(row_y_labels, start=1):
            fig.update_yaxes(
                title_text=y_label,
                title_font=dict(size=FONT_SIZE_MEDIUM),
                tickfont=dict(size=FONT_SIZE_SMALL),
                row=var_idx,
                col=1,
            )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=FONT_SIZE_MEDIUM),
        tickfont=dict(size=FONT_SIZE_SMALL),
        row=n_vars,
        col=1,
    )
    fig.update_xaxes(type="log" if log_x else "linear")
    fig.update_yaxes(type="log" if log_y else "linear")

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        height=height if height is not None else SUBPLOT_HEIGHT * n_vars,
        width=width,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        hovermode="x unified",
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )

    return fig
