from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale, _to_seconds
from .plotting_constants import *


def data_instance_subplots(
    rows: List[List[DataInstance]],
    title: str | None = None,
    row_y_labels: List[str | None] | None = None,
    show_legend: bool = True,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    subplot_config: SubplotConfig = DEFAULT_SUBPLOT_CONFIG,
    timestamp_unit: Timescale = Timescale.MS,
) -> go.Figure:
    """Plot groups of DataInstances as stacked subplots on a shared time axis.

    Each element of ``rows`` becomes one subplot row. Multiple DataInstances
    in the same row are overlaid on that row's y-axis — useful for comparing
    signals that share the same units or scale.

    Parameters
    ----------
    rows : List[List[DataInstance]]
        Outer list defines subplot rows (top to bottom); inner list defines
        the DataInstances overlaid within that row.
    title : str | None, optional
        Figure-level title. Default is None (no title).
    row_y_labels : List[str | None] | None, optional
        Y-axis label for each row. Must match the length of ``rows`` when
        provided. ``None`` entries fall back to auto-labelling from the
        DataInstance labels in that row. Default is None (all rows auto-label).
    show_legend : bool, optional
        Whether to show the figure legend. Default is True.
    font_config : FontConfig, optional
        Font sizes for title, axis labels, tick labels, and legend.
    subplot_config : SubplotConfig, optional
        Dimensions, spacing, and style for the subplot grid.
    timestamp_unit : Timescale, optional
        Timestamp unit of the underlying DataInstances. Converted to seconds
        for x-axis display. Default is Timescale.MS.

    Returns
    -------
    go.Figure
        Plotly figure containing the stacked subplot grid.

    Examples
    --------
    >>> fig = data_instance_subplots(
    ...     rows=[[speed_di], [torque_di, motor_di]],
    ...     title="Run Overview",
    ...     row_y_labels=["Speed (mph)", "Torque / Motor"],
    ... )
    >>> fig.show()
    """
    if not rows:
        print("Warning: No rows provided for subplot figure")
        return go.Figure()

    n = len(rows)

    subplot_titles: List[str] | None = None
    if subplot_config.show_subplot_titles:
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
        vertical_spacing=subplot_config.vertical_spacing,
        subplot_titles=subplot_titles,
    )

    for row_idx, row_dis in enumerate(rows, start=1):
        for di in row_dis:
            if len(di) == 0:
                print(f"Warning: No data points in DataInstance for {di.label}")
                continue

            timestamps_s = _to_seconds(di.timestamp_np.astype(float), timestamp_unit)

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

        # Resolve y-axis label for this row
        if row_y_labels and row_idx - 1 < len(row_y_labels):
            y_label = row_y_labels[row_idx - 1]
        else:
            labels = [di.label for di in row_dis if di.label]
            y_label = ", ".join(labels) if labels else None

        fig.update_yaxes(
            title_text=y_label,
            title_font=dict(size=font_config.medium),
            tickfont=dict(size=font_config.small),
            row=row_idx,
            col=1,
        )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=font_config.medium),
        tickfont=dict(size=font_config.small),
        row=n,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=font_config.large),
        ),
        height=subplot_config.height_per_row * n,
        width=subplot_config.width,
        margin=subplot_config.margin,
        plot_bgcolor=subplot_config.plot_bgcolor,
        showlegend=show_legend,
        hovermode="x unified",
        legend=dict(font=dict(size=font_config.small)),
    )

    return fig


def stride_downsample(
    timestamps_s: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    max_display_resolution: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return stride-downsampled arrays capped at *max_display_resolution*.

    Parameters
    ----------
    timestamps_s : np.ndarray
        Time values in seconds.
    values : np.ndarray
        Corresponding sample values.
    max_display_resolution : float
        Target display sample rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Downsampled ``(timestamps_s, values)``.
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
    logs: List[SingleRunData],
    var_names: List[str],
    row_y_labels: List[str | None] | None = None,
    title: str | None = None,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    config: MultiLogSubplotConfig = DEFAULT_MULTI_LOG_SUBPLOT_CONFIG,
    timestamp_unit: Timescale = Timescale.MS,
) -> go.Figure:
    """Plot the same set of variables across multiple logs as stacked subplots, with each
    variable on one subplot.

    Parameters
    ----------
    logs : List[SingleRunData]
    var_names : List[str]
        Ordered list of variable names to plot, one per subplot row.  Variables
        absent from log are silently skipped.
    row_y_labels : List[str | None] | None, optional
        Y-axis label for each row. Must match the length of ``var_names``
    title : str | None, optional
        Base figure title.  When multiple figures are produced a ``(1/N)``
        suffix is appended automatically.
    font_config : FontConfig, optional
    config : MultiLogSubplotConfig, optional
    timestamp_unit : Timescale, optional
        Timestamp unit of the underlying DataInstances. Default is Timescale.MS.

    Returns
    -------
    go.Figure
        The combined figure with all subplots.

    Examples
    --------
    >>> fig = plot_multi_log_subplots(
    ...     logs=[aly1.data, aly2.data],
    ...     var_names=["pcm."pcm.wheelSpeeds.frontLeft", "ludwig.steeringWheel.angle", "pcm.pedals.brakePressure.front"],
    ...     title="Speed Comparison",
    ... )
    >>> fig.show()
    """
    if not logs or not var_names:
        print("No logs or variable names provided.")
        return go.Figure()

    n_vars = len(var_names)

    fig = make_subplots(
        rows=n_vars,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=config.vertical_spacing,
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
            timestamps_s = _to_seconds(di.timestamp_np.astype(float), timestamp_unit)
            values = di.value_np
            if config.max_display_resolution:
                timestamps_s, values = stride_downsample(
                    timestamps_s, values, config.max_display_resolution
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
                title_font=dict(size=font_config.medium),
                tickfont=dict(size=font_config.small),
                row=var_idx,
                col=1,
            )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=font_config.medium),
        tickfont=dict(size=font_config.small),
        row=n_vars,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=font_config.large),
        ),
        height=config.height_per_row * n_vars,
        width=config.width,
        margin=config.margin,
        plot_bgcolor=config.plot_bgcolor,
        hovermode="x unified",
        legend=dict(font=dict(size=font_config.small)),
    )

    return fig
