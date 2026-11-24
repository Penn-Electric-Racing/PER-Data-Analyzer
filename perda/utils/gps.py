from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

# Only needed for the plot_gps_trimmer return type; the notebook extra provides it.
try:
    import ipywidgets as widgets
except ModuleNotFoundError:
    pass

from ..constants import EARTH_RADIUS_METERS
from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..plotting.defaults import MAP_STYLE, PLOT_HEIGHT, PLOT_WIDTH
from ..plotting.map_trace import plot_map_trace
from ..plotting.parametric_overlay import plot_parametric_overlay
from ..units import Timescale

# Outlier filtering thresholds (m). float("inf") disables a filter.
GPS_MAX_RADIUS_M = 5_000.0
GPS_MAX_JUMP_M = 30.0


def gps_to_enu(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    lat_ref: float | None = None,
    lon_ref: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Project lat/lon coordinates to a local East-North-Up (ENU) Cartesian frame.

    Uses a flat-Earth approximation, accurate to within ~0.1 % for distances
    up to ~10 km from the reference point.

    Parameters
    ----------
    lat : NDArray[np.float64]
        Latitude values in degrees.
    lon : NDArray[np.float64]
        Longitude values in degrees.
    lat_ref : float | None, optional
        Reference latitude in degrees.  Defaults to the median of ``lat``.
    lon_ref : float | None, optional
        Reference longitude in degrees.  Defaults to the median of ``lon``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``(east_m, north_m)`` arrays giving signed distance in metres from the
        reference point along the East and North axes respectively.

    Examples
    --------
    >>> from perda.utils import gps_to_enu
    >>> east, north = gps_to_enu(lat_array, lon_array)
    """
    lat_ref_rad = np.radians(lat_ref if lat_ref is not None else np.median(lat))
    lon_ref_rad = np.radians(lon_ref if lon_ref is not None else np.median(lon))
    east_m = (np.radians(lon) - lon_ref_rad) * EARTH_RADIUS_METERS * np.cos(lat_ref_rad)
    north_m = (np.radians(lat) - lat_ref_rad) * EARTH_RADIUS_METERS
    return east_m.astype(np.float64), north_m.astype(np.float64)


def filter_gps_cartesian(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    max_radius_m: float = GPS_MAX_RADIUS_M,
    max_jump_m: float = GPS_MAX_JUMP_M,
) -> NDArray[np.bool_]:
    """
    Build a boolean mask that removes Cartesian GPS outliers.

    Two independent filters are applied and combined with logical-AND:

    * **Radius filter** — discards points whose straight-line distance from the
      centroid of all points exceeds ``max_radius_m``.
    * **Jump filter** — discards points where the step distance from the previous
      point exceeds ``max_jump_m``.

    Parameters
    ----------
    lat : NDArray[np.float64]
        Latitude values in degrees.
    lon : NDArray[np.float64]
        Longitude values in degrees.
    max_radius_m : float, optional
        Points farther than this many metres from the centre of all points are dropped. Use ``float("inf")`` to disable. Default is GPS_MAX_RADIUS_M (5000).
    max_jump_m : float, optional
        Points more than this many metres from the previous point are dropped as GPS glitches. Use ``float("inf")`` to disable. Default is GPS_MAX_JUMP_M (30).

    Returns
    -------
    NDArray[``np.bool_``]
        Boolean mask; ``True`` where a point should be *kept*.

    Examples
    --------
    >>> from perda.utils import filter_gps_cartesian
    >>> mask = filter_gps_cartesian(lat_array, lon_array, max_radius_m=1500.0)
    >>> lat_clean, lon_clean = lat_array[mask], lon_array[mask]
    """
    east, north = gps_to_enu(lat, lon)
    radius_mask = np.sqrt(east**2 + north**2) <= max_radius_m
    dx = np.diff(east, prepend=east[0])
    dy = np.diff(north, prepend=north[0])
    jump_mask = np.sqrt(dx**2 + dy**2) <= max_jump_m
    return (radius_mask & jump_mask).astype(np.bool_)


def plot_gps_comparison(
    runs: list[tuple[NDArray[np.float64], NDArray[np.float64], str]],
    title: str | None = None,
    max_radius_m: float = GPS_MAX_RADIUS_M,
    max_jump_m: float = GPS_MAX_JUMP_M,
    width: int = PLOT_HEIGHT,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Overlay GPS tracks from multiple runs on a shared East/North frame in metres.

    Each run is projected to East/North coordinates around one shared reference point (median of all points across all runs), outliers are removed with :func:`filter_gps_cartesian`, and the result is plotted with :func:`perda.plotting.plot_parametric_overlay`. Runs with no points left after filtering are skipped.

    Parameters
    ----------
    runs : list[tuple[NDArray[np.float64], NDArray[np.float64], str]]
        One ``(lat_array, lon_array, label)`` tuple per run, with latitude and longitude in degrees. ``lat_array`` and ``lon_array`` must be the same length. ``label`` is the legend name.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    max_radius_m : float, optional
        Outlier filter: points farther than this many metres from the centre of the run are dropped. Use ``float("inf")`` to disable. Default is GPS_MAX_RADIUS_M (5000).
    max_jump_m : float, optional
        Outlier filter: points more than this many metres from the previous point are dropped. Use ``float("inf")`` to disable. Default is GPS_MAX_JUMP_M (30).
    width : int, optional
        Figure width in pixels. Default is PLOT_HEIGHT (700), making a square plot.
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure with one marker trace per run, in metres East/North of a shared reference point. Change anything not exposed here (fonts, marker size, colors, ...) with ``fig.update_layout`` or ``fig.update_traces`` before calling ``fig.show()``.

    Raises
    ------
    ValueError
        If two runs share the same label.

    Examples
    --------
    >>> from perda.utils import plot_gps_comparison
    >>> fig = plot_gps_comparison(
    ...     [(lat1, lon1, "Run 1"), (lat2, lon2, "Run 2")],
    ...     title="GPS Comparison",
    ... )
    """
    labels = [label for _, _, label in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels must be unique, got {labels}.")

    all_lat = np.concatenate([lat for lat, _, _ in runs])
    all_lon = np.concatenate([lon for _, lon, _ in runs])
    lat_ref = float(np.median(all_lat))
    lon_ref = float(np.median(all_lon))

    east_by_run: dict[str, NDArray[np.float64]] = {}
    north_by_run: dict[str, NDArray[np.float64]] = {}
    for lat, lon, label in runs:
        east, north = gps_to_enu(lat, lon, lat_ref=lat_ref, lon_ref=lon_ref)
        mask = filter_gps_cartesian(lat, lon, max_radius_m, max_jump_m)
        if not mask.any():
            continue
        east_by_run[label] = east[mask]
        north_by_run[label] = north[mask]

    return plot_parametric_overlay(
        east_by_run,
        north_by_run,
        x_label="East (m)",
        y_label="North (m)",
        title=title,
        mode="markers",
        equal_aspect=True,
        width=width,
        height=height,
    )


def create_representative_gps_image(
    lat_raw: DataInstance,
    lon_raw: DataInstance,
    vel_raw: DataInstance | None = None,
    vel_thresh: float = 0.5,
    title: str | None = None,
    max_radius_m: float = GPS_MAX_RADIUS_M,
    max_jump_m: float = GPS_MAX_JUMP_M,
    map_style: str = MAP_STYLE,
    width: int = PLOT_HEIGHT,
    height: int = PLOT_HEIGHT,
) -> go.Figure | None:
    """
    Draw a log's GPS track on an interactive map background.

    Aligns longitude (and speed) onto latitude's timestamps, trims away the stationary start and end of the log if speed is given, removes outliers with :func:`filter_gps_cartesian`, then plots with :func:`perda.plotting.plot_map_trace`.

    Parameters
    ----------
    lat_raw : DataInstance
        Latitude in degrees. Its timestamps are the base that longitude and velocity are aligned to.
    lon_raw : DataInstance
        Longitude in degrees.
    vel_raw : DataInstance | None, optional
        Vehicle speed, used to cut off the stationary start and end of the log. Default is None (show every point).
    vel_thresh : float, optional
        Only the span between the first and last moment speed exceeds this value is shown. Same unit as ``vel_raw``. Ignored if ``vel_raw`` is None. Default is 0.5.
    title : str | None, optional
        Title shown above the map. Default is None (no title).
    max_radius_m : float, optional
        Outlier filter: points farther than this many metres from the centre of the track are dropped. Use ``float("inf")`` to disable. Default is GPS_MAX_RADIUS_M (5000).
    max_jump_m : float, optional
        Outlier filter: points more than this many metres from the previous point are dropped. Use ``float("inf")`` to disable. Default is GPS_MAX_JUMP_M (30).
    map_style : str, optional
        Background map tiles, any Plotly MapLibre style name such as ``"open-street-map"``, ``"carto-positron"`` or ``"satellite"``. Default is MAP_STYLE (``"open-street-map"``, needs no API key).
    width : int, optional
        Figure width in pixels. Default is PLOT_HEIGHT (700), making a square map.
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure | None
        Plotly figure with a map background, or None if speed never exceeds ``vel_thresh`` or no points remain after filtering. Change anything not exposed here (line color, zoom, ...) with ``fig.update_layout`` or ``fig.update_traces`` before calling ``fig.show()``.

    Examples
    --------
    >>> from perda.utils import create_representative_gps_image
    >>> fig = create_representative_gps_image(lat_di, lon_di, vel_raw=vel_di, title="Lap Track")
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
    cartesian_mask = filter_gps_cartesian(
        lat_trimmed, lon_trimmed, max_radius_m, max_jump_m
    )
    lat_filtered = lat_trimmed[cartesian_mask]
    lon_filtered = lon_trimmed[cartesian_mask]

    if len(lat_filtered) == 0:
        print("No GPS points remain after outlier filtering, skip graphing")
        return None

    return plot_map_trace(
        lat_filtered,
        lon_filtered,
        title=title,
        map_style=map_style,
        width=width,
        height=height,
    )


def plot_gps_trimmer(
    lat: DataInstance,
    lon: DataInstance,
    timestamp_unit: Timescale,
    title: str | None = None,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> widgets.VBox:
    """
    Interactive GPS track plot (longitude vs latitude) with a range slider for picking a time window of the log.

    Drag the two slider handles to hide points outside the selected range. The labels under the slider show the sample index and the raw timestamp (in the log's own unit, e.g. ms or us, not seconds) of the first and last visible point. Pass those timestamps straight to ``trim_single_run_data(data, ts_start=..., ts_end=...)`` or ``DataInstance.trim`` to cut the log to that window. Designed for Jupyter notebooks, and requires the ``notebook`` extra (ipywidgets).

    Parameters
    ----------
    lat : DataInstance
        Latitude in degrees. Its timestamps define the points shown and the timestamps in the slider labels.
    lon : DataInstance
        Longitude in degrees. Does not need to share timestamps with ``lat``: it is automatically aligned onto ``lat``'s timestamps with ``left_join_data_instances`` (linear interpolation).
    timestamp_unit : Timescale
        Unit of the raw timestamps, shown next to the timestamps in the slider labels. Pass ``aly.data.timestamp_unit``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    ipywidgets.VBox
        Widget containing the figure, the slider and the labels. Display it as the last expression in a notebook cell or with ``display(widget)``.

    Examples
    --------
    >>> from perda.core_data_structures import trim_single_run_data
    >>> from perda.utils import plot_gps_trimmer
    >>> widget = plot_gps_trimmer(
    ...     aly.data["pcm.vnav.posLla.latitude"],
    ...     aly.data["pcm.vnav.posLla.longitude"],
    ...     aly.data.timestamp_unit,
    ...     title="GPS Track",
    ... )
    >>> display(widget)
    >>> # After reading the start/end timestamps off the slider:
    >>> trimmed = trim_single_run_data(aly.data, ts_start=436_000, ts_end=1_552_000)
    """
    from ..plotting.parametric_trimmer import plot_parametric_trimmer

    lat_aligned, lon_aligned = left_join_data_instances(lat, lon)

    return plot_parametric_trimmer(
        x=lon_aligned.value_np,
        y=lat_aligned.value_np,
        timestamps=lat_aligned.timestamp_np,
        x_label="Longitude",
        y_label="Latitude",
        title=title,
        timestamp_unit=timestamp_unit,
        width=width,
        height=height,
    )
