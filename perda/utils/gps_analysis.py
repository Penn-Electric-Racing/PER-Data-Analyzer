import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from ..analyzer.data_instance import DataInstance, left_join_data_instances
from ..plotting.parametric_plot import (
    plot_parametric_curve_square,
    plot_parametric_trimmer,
)
from ..plotting.plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    LayoutConfig,
)
from .types import Timescale


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

    return plot_parametric_trimmer(
        x=lon.value_np,
        y=lat.value_np,
        timestamps=lat.timestamp_np,
        x_label="Longitude",
        y_label="Latitude",
        title=title,
        timestamp_unit=timestamp_unit,
        layout_config=layout_config,
        font_config=font_config,
    )


def create_representative_gps_image(
    lat_raw: DataInstance,
    lon_raw: DataInstance,
    vel_raw: DataInstance | None = None,
    vel_thresh: float = 0.5,
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure | None:
    """
    Generate a GPS image plot from raw latitude, longitude, and velocity data.

    Parameters
    ----------
    lat_raw : DataInstance
        Raw latitude data instance, timestamp alignment base.
    lon_raw : DataInstance
        Raw longitude data instance.
    vel_raw : DataInstance | None, optional
        Raw velocity data instance.
    vel_thresh : float, optional
        Velocity threshold for trimming, by default 0.5, same unit as velocity.
    title : str | None, optional
        Title for the plot.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    go.Figure | None
        Plotly figure, or None if velocity never exceeds the threshold.
    """
    lat_aligned, lon_aligned = left_join_data_instances(lat_raw, lon_raw)

    if vel_raw is not None:
        _, vel_aligned = left_join_data_instances(lat_aligned, vel_raw)

        mask = vel_aligned.value_np > vel_thresh
        ts_first_idx = np.argmax(mask) if mask.any() else -1
        ts_last_idx = len(mask) - 1 - np.argmax(mask[::-1]) if mask.any() else -1

        if ts_first_idx == -1 or ts_last_idx == -1:
            print("Velocity never exceeds threshold, skip graphing")
            return None

        lon_trimmed = lon_aligned.value_np[ts_first_idx : ts_last_idx + 1]
        lat_trimmed = lat_aligned.value_np[ts_first_idx : ts_last_idx + 1]
    else:
        lon_trimmed = lon_aligned.value_np
        lat_trimmed = lat_aligned.value_np

    return plot_parametric_curve_square(
        x=lon_trimmed,
        y=lat_trimmed,
        x_label="Longitude",
        y_label="Latitude",
        title=title,
        layout_config=layout_config,
        font_config=font_config,
    )
