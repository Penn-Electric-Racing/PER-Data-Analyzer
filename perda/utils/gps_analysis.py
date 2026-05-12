from typing import List

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from ..constants import R_EARTH_M
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


def gps_to_enu(
    lat: NDArray[float64],
    lon: NDArray[float64],
    lat_ref: float | None = None,
    lon_ref: float | None = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Project lat/lon coordinates to a local East-North-Up (ENU) Cartesian frame.

    Uses a flat-Earth approximation, accurate to within ~0.1 % for distances
    up to ~10 km from the reference point.

    Parameters
    ----------
    lat : NDArray[float64]
        Latitude values in degrees.
    lon : NDArray[float64]
        Longitude values in degrees.
    lat_ref : float | None, optional
        Reference latitude in degrees.  Defaults to the median of ``lat``.
    lon_ref : float | None, optional
        Reference longitude in degrees.  Defaults to the median of ``lon``.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        ``(east_m, north_m)`` arrays giving signed distance in metres from the
        reference point along the East and North axes respectively.

    Examples
    --------
    >>> east, north = gps_to_enu(lat_array, lon_array)
    """
    lat_ref_rad = np.radians(lat_ref if lat_ref is not None else np.median(lat))
    lon_ref_rad = np.radians(lon_ref if lon_ref is not None else np.median(lon))
    east_m = (np.radians(lon) - lon_ref_rad) * R_EARTH_M * np.cos(lat_ref_rad)
    north_m = (np.radians(lat) - lat_ref_rad) * R_EARTH_M
    return east_m.astype(np.float64), north_m.astype(np.float64)


def filter_gps_cartesian(
    lat: NDArray[float64],
    lon: NDArray[float64],
    gps_map_config: GpsMapConfig = DEFAULT_GPS_MAP_CONFIG,
) -> NDArray[np.bool_]:
    """
    Build a boolean mask that removes Cartesian GPS outliers.

    Two independent filters are applied and combined with logical-AND:

    * **Radius filter** — discards points whose straight-line distance from the
      centroid of all points exceeds ``gps_map_config.max_radius_m``.
    * **Jump filter** — discards points where the step distance from the previous
      point exceeds ``gps_map_config.max_jump_m``.

    Parameters
    ----------
    lat : NDArray[float64]
        Latitude values in degrees.
    lon : NDArray[float64]
        Longitude values in degrees.
    gps_map_config : GpsMapConfig, optional

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask; ``True`` where a point should be *kept*.

    Examples
    --------
    >>> mask = filter_gps_cartesian(lat_array, lon_array)
    >>> lat_clean, lon_clean = lat_array[mask], lon_array[mask]
    """
    east, north = gps_to_enu(lat, lon)
    radius_mask = np.sqrt(east**2 + north**2) <= gps_map_config.max_radius_m
    dx = np.diff(east, prepend=east[0])
    dy = np.diff(north, prepend=north[0])
    jump_mask = np.sqrt(dx**2 + dy**2) <= gps_map_config.max_jump_m
    return (radius_mask & jump_mask).astype(np.bool_)


def plot_gps_comparison(
    runs: List[tuple[NDArray[float64], NDArray[float64], str]],
    title: str | None = None,
    gps_map_config: GpsMapConfig = DEFAULT_GPS_MAP_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure | None:
    """
    Overlay GPS tracks from multiple runs on a shared East/North Cartesian frame.

    Each run is projected to ENU coordinates using a single shared reference
    point (median of all points across all runs), then filtered with
    :func:`filter_gps_cartesian` before plotting.

    Parameters
    ----------
    runs : List[tuple[NDArray[float64], NDArray[float64], str]]
        Each element is ``(lat_array, lon_array, label)`` for one run.
        ``lat_array`` and ``lon_array`` must be the same length.
    title : str | None, optional
    gps_map_config : GpsMapConfig, optional
        Controls Cartesian outlier thresholds (``max_radius_m``, ``max_jump_m``)
        and marker appearance.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    go.Figure
        Plotly figure with one scatter trace per run

    Examples
    --------
    >>> fig = plot_gps_comparison(
    ...     [(lat1, lon1, "Run 1"), (lat2, lon2, "Run 2")],
    ...     title="GPS Comparison",
    ... )
    >>> fig.show()
    """

    all_lat = np.concatenate([lat for lat, _, _ in runs])
    all_lon = np.concatenate([lon for _, lon, _ in runs])
    lat_ref = float(np.median(all_lat))
    lon_ref = float(np.median(all_lon))

    fig = go.Figure()

    for lat, lon, label in runs:
        east, north = gps_to_enu(lat, lon, lat_ref=lat_ref, lon_ref=lon_ref)
        mask = filter_gps_cartesian(lat, lon, gps_map_config)
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=east[mask],
                y=north[mask],
                mode="markers",
                marker=dict(size=gps_map_config.marker_size),
                name=label,
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
        xaxis_title=dict(text="East (m)", font=dict(size=font_config.medium)),
        yaxis_title=dict(text="North (m)", font=dict(size=font_config.medium)),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=layout_config.height,
        height=layout_config.height,
        margin=layout_config.margin,
        legend=dict(font=dict(size=font_config.small)),
    )
    return fig


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

    Outliers (large radius from centroid or sudden jumps) are removed via
    :func:`filter_gps_cartesian`.  Uses CARTO Positron tiles by default
    (free, no API key required).

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
        Map overlay and outlier filtering configuration.

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

    # Trim to when the vehicle is moving if velocity data is provided, otherwise show all points
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

    # Filter outliers
    cartesian_mask = filter_gps_cartesian(lat_trimmed, lon_trimmed, gps_map_config)
    lat_filtered = lat_trimmed[cartesian_mask]
    lon_filtered = lon_trimmed[cartesian_mask]

    if len(lat_filtered) == 0:
        print("No GPS points remain after outlier filtering, skip graphing")
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
