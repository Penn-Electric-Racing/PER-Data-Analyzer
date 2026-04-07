import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..plotting.parametric_plot import (
    plot_parametric_curve_square,
    plot_parametric_trimmer,
)
from ..plotting.plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_GPS_MAP_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    GpsMapConfig,
    LayoutConfig,
)
from ..units import Timescale


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

    Examples
    --------
    >>> widget = plot_gps_trimmer(lat_di, lon_di, title="GPS Track")
    >>> display(widget)
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


def _filter_gps_bounds(
    lat: NDArray[float64],
    lon: NDArray[float64],
    gps_map_config: GpsMapConfig,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Remove GPS points outside valid lat/lon bounds.

    Parameters
    ----------
    lat : NDArray[float64]
        Latitude values.
    lon : NDArray[float64]
        Longitude values.
    gps_map_config : GpsMapConfig
        Configuration containing lat_range and lon_range bounds.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        Filtered (lat, lon) arrays.
    """
    lat_lo, lat_hi = gps_map_config.lat_range
    lon_lo, lon_hi = gps_map_config.lon_range
    mask = (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
    return lat[mask], lon[mask]


def create_representative_gps_image(
    lat_raw: DataInstance,
    lon_raw: DataInstance,
    vel_raw: DataInstance | None = None,
    vel_thresh: float = 0.5,
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    gps_map_config: GpsMapConfig = DEFAULT_GPS_MAP_CONFIG,
) -> go.Figure | None:
    """
    Generate a GPS trace on an interactive map background.

    Points outside the configured lat/lon bounds are filtered out.
    Uses CARTO Positron tiles by default (free, no API key required).

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
    gps_map_config : GpsMapConfig, optional
        Map overlay and GPS filtering configuration.

    Returns
    -------
    go.Figure | None
        Plotly figure with map background, or None if velocity never exceeds
        the threshold or no points remain after filtering.

    Examples
    --------
    >>> fig = create_representative_gps_image(lat_di, lon_di, vel_raw=vel_di, title="Lap Track")
    >>> fig.show()
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

    lat_filtered, lon_filtered = _filter_gps_bounds(
        lat_trimmed, lon_trimmed, gps_map_config
    )

    if len(lat_filtered) == 0:
        print("No GPS points remain after bounds filtering, skip graphing")
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=lat_filtered,
            lon=lon_filtered,
            mode="lines+markers",
            marker=dict(size=gps_map_config.marker_size),
            line=dict(
                width=gps_map_config.line_width,
                color=gps_map_config.line_color,
            ),
        )
    )

    center_lat = (lat_filtered.min() + lat_filtered.max()) / 2
    center_lon = (lon_filtered.min() + lon_filtered.max()) / 2

    lat_span = lat_filtered.max() - lat_filtered.min()
    lon_span = lon_filtered.max() - lon_filtered.min()
    max_span = max(lat_span, lon_span) * (1 + gps_map_config.zoom_padding)
    zoom = np.log2(360 / max_span) if max_span > 0 else 15

    mapbox_kwargs: dict[str, object] = dict(
        style=gps_map_config.mapbox_style,
        center=dict(lat=float(center_lat), lon=float(center_lon)),
        zoom=float(zoom),
    )
    if gps_map_config.mapbox_token:
        mapbox_kwargs["accesstoken"] = gps_map_config.mapbox_token

    fig.update_layout(
        title=dict(
            text=title,
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        mapbox=mapbox_kwargs,
        showlegend=False,
        width=int(layout_config.height),
        height=int(layout_config.height),
        margin=layout_config.margin,
    )

    return fig
