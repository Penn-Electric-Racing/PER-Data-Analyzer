import numpy as np
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from .defaults import (
    FONT_SIZE_LARGE,
    MAP_LINE_COLOR,
    MAP_LINE_WIDTH,
    MAP_MARKER_SIZE,
    MAP_MAX_ZOOM,
    MAP_STYLE,
    MAP_ZOOM_PADDING,
    MARGIN,
    PLOT_HEIGHT,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def plot_map_trace(
    lat: NDArray[float64],
    lon: NDArray[float64],
    title: str | None = None,
    zoom_padding: float = MAP_ZOOM_PADDING,
    width: int = PLOT_HEIGHT,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Draw a path of latitude/longitude points on top of an interactive background map. The map is
    automatically centered and zoomed to fit all points.

    Parameters
    ----------
    lat : NDArray[float64]
        Latitude of each point, in degrees.
    lon : NDArray[float64]
        Longitude of each point, in degrees. Must have the same length as ``lat``.
    title : str | None, optional
        Title shown above the map. Default is None (no title).
    zoom_padding : float, optional
        Extra space shown around the points when auto-zooming, as a fraction of the path's extent.
        Default is MAP_ZOOM_PADDING (0.4).
    width : int, optional
        Figure width in pixels. Default is PLOT_HEIGHT (700), making a square map.
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure with a map background. Change anything not exposed here (line color, marker size, zoom, ...) with ``fig.update_layout`` or ``fig.update_traces`` before calling ``fig.show()``, e.g. ``fig.update_layout(map_zoom=16)``.

    Raises
    ------
    ValueError
        If ``lat`` and ``lon`` have different lengths or are empty.

    Examples
    --------
    >>> from perda.plotting import plot_map_trace
    >>> fig = plot_map_trace(lat_array, lon_array, title="Lap Track")
    """
    if len(lat) != len(lon):
        raise ValueError(
            f"lat and lon have different lengths ({len(lat)} vs {len(lon)})."
        )
    if len(lat) == 0:
        raise ValueError("lat and lon are empty.")

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=lat,
            lon=lon,
            mode="lines+markers",
            marker=dict(size=MAP_MARKER_SIZE),
            line=dict(width=MAP_LINE_WIDTH, color=MAP_LINE_COLOR),
        )
    )

    center_lat = (lat.min() + lat.max()) / 2
    center_lon = (lon.min() + lon.max()) / 2
    max_span = max(lat.max() - lat.min(), lon.max() - lon.min()) * (1 + zoom_padding)
    zoom = np.log2(360 / max_span) if max_span > 0 else MAP_MAX_ZOOM

    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        map=dict(
            style=MAP_STYLE,
            center=dict(lat=float(center_lat), lon=float(center_lon)),
            zoom=float(zoom),
        ),
        showlegend=False,
        width=width,
        height=height,
        margin=MARGIN,
    )

    return fig
