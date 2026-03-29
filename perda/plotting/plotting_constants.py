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

    bucket_size_ms: int = 1000
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


# Default configuration instances
DEFAULT_FONT_CONFIG = FontConfig()
DEFAULT_LAYOUT_CONFIG = LayoutConfig()
DEFAULT_DIFF_PLOT_CONFIG = DiffPlotConfig()
DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG = ScatterHistogramPlotConfig()
DEFAULT_VLINE_CONFIG = VLineConfig()
