from typing import Dict

from pydantic import BaseModel, Field


class FontConfig(BaseModel):
    """Font configuration for plot elements.

    Simplified to three sizes:
    - Large: Titles
    - Medium: Axis labels, legend titles
    - Small: Tick labels, legend items
    """

    large: int = 20
    medium: int = 14
    small: int = 10


class LayoutConfig(BaseModel):
    """Layout configuration for plot dimensions and spacing."""

    width: int = 1200
    height: int = 700

    margin: Dict[str, int] = Field(default_factory=lambda: dict(l=70, r=50, t=90, b=70))

    plot_bgcolor: str = "white"

    title_x: float = 0.5
    title_xanchor: str = "center"
    title_yanchor: str = "top"


class DiffPlotConfig(BaseModel):
    """Configuration for diff bar chart visualization."""

    bucket_size_s: float = 1.0
    color_base_extra: str = "blue"
    color_incom_extra: str = "darkorange"
    color_value_mismatch: str = "crimson"
    color_total: str = "gray"


class ScatterHistogramPlotConfig(BaseModel):
    """Configuration for frequency analysis visualization."""

    color_scatter: str = "blue"
    color_line: str = "crimson"
    color_histogram: str = "blue"
    histogram_bins: int = 80


class VLineConfig(BaseModel):
    """Configuration for vertical marker lines on plots."""

    color: str = "gray"
    width: int = 2
    dash: str = "dash"
    opacity: float = 0.7


class GpsMapConfig(BaseModel):
    """Configuration for GPS plotting and outlier filtering.

    Parameters
    ----------
    mapbox_style : str
        Mapbox tile style. ``"carto-positron"`` is free and requires no token.
    mapbox_token : str
        Mapbox access token. Only needed for Mapbox-hosted styles.
    marker_size : int
        Marker size on GPS traces.
    line_width : int
        Line width on the map trace.
    line_color : str
        Color of the GPS trace line.
    zoom_padding : float
        Extra fraction added around the data extent when auto-zooming.
    max_radius_m : float
        Maximum allowed distance in metres from the centroid of all GPS points.
        Points further than this are discarded as outliers.
        Set to ``float("inf")`` to disable.
    max_jump_m : float
        Maximum allowed step distance in metres between consecutive points.
        Points that jump further than this are discarded as outliers.
        Set to ``float("inf")`` to disable.
    """

    mapbox_style: str = "carto-positron"
    mapbox_token: str = ""
    marker_size: int = 3
    line_width: int = 2
    line_color: str = "red"
    zoom_padding: float = 0.4
    max_radius_m: float = 5_000.0
    max_jump_m: float = 30.0


class SubplotConfig(BaseModel):
    """Layout configuration for stacked subplot figures.

    Parameters
    ----------
    height_per_row : int
        Pixel height allocated to each subplot row.
    vertical_spacing : float
        Vertical gap between subplot rows as a fraction of the total figure height.
    width : int
        Total figure width in pixels.
    show_subplot_titles : bool
        If True, each row receives a subtitle taken from the DataInstance label.
    plot_bgcolor : str
        Background color of each subplot panel.
    margin : Dict[str, int]
        Figure margins as a dict with keys ``l``, ``r``, ``t``, ``b``.
    """

    height_per_row: int = 250
    vertical_spacing: float = 0.04
    width: int = 1200
    show_subplot_titles: bool = True
    plot_bgcolor: str = "white"
    margin: Dict[str, int] = Field(default_factory=lambda: dict(l=70, r=50, t=90, b=70))


class FFTPlotConfig(BaseModel):
    """Configuration for FFT magnitude spectrum plots.

    Parameters
    ----------
    log_x : bool
        Logarithmic x-axis (frequency). Default is True.
    log_y : bool
        Logarithmic y-axis (magnitude). Default is False.
    line_color : str
        Color of the spectrum trace(s). Default is ``"steelblue"``.
    height_single : int
        Figure height in pixels when ``stacked=False`` (single-panel view).
        Default is 500.
    """

    log_x: bool = True
    log_y: bool = False
    line_color: str = "steelblue"
    height_single: int = 500


class MultiLogSubplotConfig(BaseModel):
    """Configuration for multi-log variable comparison subplot figures.

    Parameters
    ----------
    max_display_resolution : float | None
        Maximum display sample rate in Hz. Traces are stride-downsampled to
        approximately this rate before rendering.  ``None`` disables downsampling.
    height_per_row : int
        Pixel height allocated to each subplot row.
    vertical_spacing : float
        Vertical gap between subplot rows as a fraction of the total figure height.
    width : int
        Total figure width in pixels.
    plot_bgcolor : str
        Background color of each subplot panel.
    margin : Dict[str, int]
        Figure margins as a dict with keys ``l``, ``r``, ``t``, ``b``.
    """

    max_display_resolution: float | None = 50.0
    height_per_row: int = 250
    vertical_spacing: float = 0.05
    width: int = 1200
    plot_bgcolor: str = "white"
    margin: Dict[str, int] = Field(default_factory=lambda: dict(l=70, r=50, t=90, b=70))


# Default configuration instances
DEFAULT_FONT_CONFIG = FontConfig()
DEFAULT_LAYOUT_CONFIG = LayoutConfig()
DEFAULT_DIFF_PLOT_CONFIG = DiffPlotConfig()
DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG = ScatterHistogramPlotConfig()
DEFAULT_VLINE_CONFIG = VLineConfig()
DEFAULT_GPS_MAP_CONFIG = GpsMapConfig()
DEFAULT_SUBPLOT_CONFIG = SubplotConfig()
DEFAULT_FFT_PLOT_CONFIG = FFTPlotConfig()
DEFAULT_MULTI_LOG_SUBPLOT_CONFIG = MultiLogSubplotConfig()
