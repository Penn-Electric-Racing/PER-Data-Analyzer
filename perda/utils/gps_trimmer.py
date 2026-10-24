import ipywidgets as widgets
import numpy as np

from ..core_data_structures.data_instance import DataInstance
from ..plotting.parametric_trimmer import plot_parametric_trimmer
from ..plotting.plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    LayoutConfig,
)
from ..units import Timescale


def plot_gps_trimmer(
    lat: DataInstance,
    lon: DataInstance,
    timestamp_unit: Timescale,
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
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
    timestamp_unit : Timescale
        Timestamp unit used for the slider timestamp labels. Pass ``aly.data.timestamp_unit``.
    title : str | None, optional
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    ipywidgets.VBox

    Examples
    --------
    >>> from perda.utils import plot_gps_trimmer
    >>> widget = plot_gps_trimmer(lat_di, lon_di, aly.data.timestamp_unit, title="GPS Track")
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
