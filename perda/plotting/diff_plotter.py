import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from .plotting_constants import *


def _bin_timestamps(
    timestamps_ms: npt.NDArray[np.int64],
    t_min: int,
    bucket_size_ms: int,
    n_buckets: int,
) -> npt.NDArray[np.int64]:
    """Count timestamps falling into each fixed-width time bucket.

    Parameters
    ----------
    timestamps_ms : npt.NDArray[np.int64]
        int64 array of event timestamps in milliseconds.
    t_min : int
        Start of the first bucket (ms).
    bucket_size_ms : int
        Width of each bucket (ms).
    n_buckets : int
        Total number of buckets.

    Returns
    -------
    int64 array of length ``n_buckets`` with per-bucket counts.
    """
    if timestamps_ms.size == 0:
        return np.zeros(n_buckets, dtype=np.int64)
    indices = ((timestamps_ms - t_min) // bucket_size_ms).clip(0, n_buckets - 1)
    return np.bincount(indices.astype(np.intp), minlength=n_buckets).astype(np.int64)


def plot_diff_bars(
    base_extra_ts: npt.NDArray[np.int64],
    incom_extra_ts: npt.NDArray[np.int64],
    value_mismatch_ts: npt.NDArray[np.int64],
    total_present_ts: npt.NDArray[np.int64],
    title: str | None = "Diff Counts Over Time (bucketed)",
    y_axis_title: str | None = "Event Count",
    show_legend: bool = True,
    diff_plot_config: DiffPlotConfig = DEFAULT_DIFF_PLOT_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """Plot diff results as bucketed bar charts.

    Renders four overlaid bar traces — total datapoints present (background),
    base-only extras, incoming-only extras, and value mismatches — bucketed
    into fixed-width time windows.

    Parameters
    ----------
    base_extra_ts:
        Timestamps (ms) of datapoints present only in the base run.
    incom_extra_ts:
        Timestamps (ms) of datapoints present only in the incoming run.
    value_mismatch_ts:
        Timestamps (ms) of matched points whose values differ.
    total_present_ts:
        Timestamps (ms) of all datapoints seen in either run.
    title:
        Plot title.
    y_axis_title:
        Label for the y-axis.
    show_legend:
        Whether to show the legend.
    diff_plot_config:
        Colors and bucket size configuration.
    layout_config:
        Plot dimensions and margin configuration.
    font_config:
        Font sizes for titles, labels, and ticks.

    Returns
    -------
    go.Figure
    """
    if total_present_ts.size == 0:
        print("No diff events to plot.")
        return go.Figure()

    bucket_size_ms = diff_plot_config.bucket_size_ms
    t_min = int(total_present_ts.min())
    t_max = int(total_present_ts.max())
    n_buckets = max((t_max - t_min) // bucket_size_ms + 1, 1)

    midpoints_s = (
        np.arange(n_buckets, dtype=np.float64) * bucket_size_ms
        + t_min
        + bucket_size_ms * 0.5
    ) / 1000.0

    total_counts = _bin_timestamps(total_present_ts, t_min, bucket_size_ms, n_buckets)
    base_counts = _bin_timestamps(base_extra_ts, t_min, bucket_size_ms, n_buckets)
    incom_counts = _bin_timestamps(incom_extra_ts, t_min, bucket_size_ms, n_buckets)
    diff_counts = _bin_timestamps(value_mismatch_ts, t_min, bucket_size_ms, n_buckets)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=midpoints_s,
            y=total_counts,
            name="All Datapoints Present",
            marker_color=diff_plot_config.color_total,
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Bar(
            x=midpoints_s,
            y=base_counts,
            name="Base-Only Points",
            marker_color=diff_plot_config.color_base_extra,
        )
    )
    fig.add_trace(
        go.Bar(
            x=midpoints_s,
            y=incom_counts,
            name="Incoming-Only Points",
            marker_color=diff_plot_config.color_incom_extra,
        )
    )
    fig.add_trace(
        go.Bar(
            x=midpoints_s,
            y=diff_counts,
            name="Value Mismatch Points",
            marker_color=diff_plot_config.color_value_mismatch,
        )
    )

    fig.update_layout(
        barmode="overlay",
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
