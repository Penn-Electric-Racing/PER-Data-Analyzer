import ipywidgets as widgets

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..plotting.defaults import PLOT_HEIGHT, PLOT_WIDTH
from ..plotting.parametric_trimmer import plot_parametric_trimmer
from ..units import Timescale


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

    Drag the two slider handles to hide points outside the selected range. The labels under the slider show the sample index and the raw timestamp (in the log's own unit, e.g. ms or us, not seconds) of the first and last visible point. Pass those timestamps straight to ``trim_single_run_data(data, ts_start=..., ts_end=...)`` or ``DataInstance.trim`` to cut the log to that window. Designed for Jupyter notebooks.

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
