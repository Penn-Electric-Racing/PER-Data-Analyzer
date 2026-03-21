import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from ..analyzer.data_instance import DataInstance, left_join_data_instances
from ..utils.types import Timescale
from .plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    LayoutConfig,
)


def plot_gps_trimmer(
    lat: DataInstance,
    lon: DataInstance,
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    timestamp_unit: Timescale = Timescale.MS,
) -> widgets.VBox:
    """
    Interactive GPS coordinate trimmer with a dual-handle range slider.

    Displays a 2D scatter plot of GPS coordinates (longitude vs latitude) alongside
    an ipywidgets IntRangeSlider that trims which points are shown. Designed for use
    in Jupyter notebooks.

    Parameters
    ----------
    lat : DataInstance
        Latitude values. Must have the same length and identical timestamps as `lon`.
    lon : DataInstance
        Longitude values. Must have the same length and identical timestamps as `lat`.
    title : str | None, optional
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional
    timestamp_unit : Timescale, optional
        Timestamp unit used for the slider timestamp labels.

    Returns
    -------
    ipywidgets.VBox
    """
    if len(lat) != len(lon):
        raise ValueError(
            f"lat and lon DataInstances have different lengths ({len(lat)} vs {len(lon)}). "
            "Align them first using a join (e.g., inner_join_data_instances, left_join_data_instances)."
        )

    if not np.array_equal(lat.timestamp_np, lon.timestamp_np):
        raise ValueError(
            "lat and lon DataInstances have mismatched timestamps. "
            "Align them first using a join (e.g., inner_join_data_instances, left_join_data_instances)."
        )

    n = len(lat)

    # Main GPS visualization
    fig = go.FigureWidget()
    fig.add_trace(
        go.Scatter(
            x=lon.value_np,
            y=lat.value_np,
            mode="lines+markers",
            marker=dict(size=2),
            line=dict(width=1),
            name="GPS Path",
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
        xaxis_title=dict(text="Longitude", font=dict(size=font_config.medium)),
        yaxis_title=dict(text="Latitude", font=dict(size=font_config.medium)),
        showlegend=False,
        width=int(layout_config.width),
        height=int(layout_config.height),
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
    )

    # Range Slider
    range_slider = widgets.IntRangeSlider(
        value=[0, n - 1],
        min=0,
        max=n - 1,
        step=1,
        description="Trim:",
        continuous_update=True,
        layout=widgets.Layout(width="100%"),
    )

    # Index labels
    start_label = widgets.Label(
        value=f"Start index: 0 | timestamp: {lat.timestamp_np[0]} {timestamp_unit.value}"
    )
    end_label = widgets.Label(
        value=f"End index: {n - 1} | timestamp: {lat.timestamp_np[-1]} {timestamp_unit.value}"
    )
    label_box = widgets.HBox(
        [start_label, end_label],
        layout=widgets.Layout(justify_content="space-between", width="90%"),
    )

    # Callback
    def on_range_change(change: dict) -> None:
        lo, hi = change["new"]
        with fig.batch_update():
            fig.data[0].x = lon.value_np[lo : hi + 1]
            fig.data[0].y = lat.value_np[lo : hi + 1]
        start_label.value = f"Start index: {lo} | timestamp: {lat.timestamp_np[lo]} {timestamp_unit.value}"
        end_label.value = f"End index: {hi} | timestamp: {lat.timestamp_np[hi]} {timestamp_unit.value}"

    range_slider.observe(on_range_change, names="value")

    return widgets.VBox(
        [fig, range_slider, label_box],
    )


def create_representative_gps_image(
    lat_raw: DataInstance,
    lon_raw: DataInstance,
    vel_raw: DataInstance | None = None,
    vel_thresh: float = 0.5,
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """
    Generate a GPS image plot from raw latitude, longitude, and velocity data.

    Parameters
    ----------
    lat_raw : DataInstance
        Raw latitude data instance, timestamp alignment base
    lon_raw : DataInstance
        Raw longitude data instance
    vel_raw : DataInstance | None, optional
        Raw velocity data instance
    vel_thresh : float, optional
        Velocity threshold for trimming, by default 0.5, same unit as velocity
    title : str | None, optional
        Title for the plot, by default None
    layout_config : LayoutConfig, optional
        Layout configuration for the plot, by default DEFAULT_LAYOUT_CONFIG
    font_config : FontConfig, optional
        Font configuration for the plot, by default DEFAULT_FONT_CONFIG

    Returns
    -------
    go.Figure
        Plotly figure object containing the GPS image
    """

    # Align timestamps
    lat_aligned, lon_aligned = left_join_data_instances(lat_raw, lon_raw)
    if vel_raw is not None:
        _, vel_aligned = left_join_data_instances(lat_aligned, vel_raw)

        # Get first/last ts vel>vel_thresh
        mask = vel_aligned.value_np > vel_thresh
        ts_first_idx = np.argmax(mask) if mask.any() else -1
        ts_last_idx = len(mask) - 1 - np.argmax(mask[::-1]) if mask.any() else -1

        if ts_first_idx == -1 or ts_last_idx == -1:
            print("Velocity never exceeds threshold, skip graphing")
            return

        # Get trimmed data
        lon_trimmed = lon_aligned.value_np[ts_first_idx : ts_last_idx + 1]
        lat_trimmed = lat_aligned.value_np[ts_first_idx : ts_last_idx + 1]
    else:
        lon_trimmed = lon_aligned.value_np
        lat_trimmed = lat_aligned.value_np

    # Get data for graph
    xmin, xmax = lon_trimmed.min(), lon_trimmed.max()
    ymin, ymax = lat_trimmed.min(), lat_trimmed.max()

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    xspan = xmax - xmin
    yspan = ymax - ymin
    span = max(xspan, yspan) * 1.1
    half = span / 2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lon_trimmed,
            y=lat_trimmed,
            mode="lines+markers",
            marker=dict(size=2),
            line=dict(width=1),
            name="GPS Path",
        )
    )

    fig.update_xaxes(
        range=[xmid - half, xmid + half],
    )
    fig.update_yaxes(
        range=[ymid - half, ymid + half],
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        xaxis_title=dict(text="Longitude", font=dict(size=font_config.medium)),
        yaxis_title=dict(text="Latitude", font=dict(size=font_config.medium)),
        showlegend=False,
        width=int(layout_config.height),  # make square
        height=int(layout_config.height),
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
    )

    return fig
