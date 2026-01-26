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


# Default configuration instances
DEFAULT_FONT_CONFIG = FontConfig()
DEFAULT_LAYOUT_CONFIG = LayoutConfig()
