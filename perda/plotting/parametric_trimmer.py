import ipywidgets as widgets
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from ..units import Timescale
from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def plot_parametric_trimmer(
    x: NDArray[float64],
    y: NDArray[float64],
    timestamp_unit: Timescale,
    timestamps: NDArray[float64] | None = None,
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> widgets.VBox:
    """
    Interactive parametric curve (y against x) with a range slider for picking a sub-window of samples.

    Drag the two slider handles to hide points outside the selected range. The labels under the slider show the index and raw timestamp of the first and last visible point, which can be passed directly to trimming functions such as ``trim_single_run_data(ts_start=..., ts_end=...)``. Designed for Jupyter notebooks.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values, e.g. longitude or East position.
    y : NDArray[float64]
        Y-axis values, e.g. latitude or North position. Must have the same length as ``x``.
    timestamp_unit : Timescale
        Unit of ``timestamps``, shown next to the timestamp in the slider labels. Ignored if ``timestamps`` is None. Pass ``aly.data.timestamp_unit``.
    timestamps : NDArray[float64] | None, optional
        Raw timestamp of each point (in ``timestamp_unit``, not converted to seconds), shown in the slider labels. Default is None (labels show sample index only).
    x_label : str, optional
        X-axis label. Default is ``"X"``.
    y_label : str, optional
        Y-axis label. Default is ``"Y"``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    ipywidgets.VBox
        Widget containing the figure, the slider and the labels. Display it as the last expression in a notebook cell or with ``display(widget)``. The figure is ``widget.children[0]`` and can be customized with ``update_layout`` like any Plotly figure.

    Raises
    ------
    ValueError
        If ``x`` is empty, or ``y`` (or ``timestamps``, when given) does not have the same length as ``x``.

    Examples
    --------
    >>> from perda.plotting import plot_parametric_trimmer
    >>> widget = plot_parametric_trimmer(
    ...     lon_arr, lat_arr, aly.data.timestamp_unit, timestamps=ts_arr, x_label="Longitude", y_label="Latitude"
    ... )
    >>> display(widget)
    """
    n = len(x)
    if n == 0:
        raise ValueError("x and y are empty; there is nothing to trim.")
    if len(y) != n:
        raise ValueError(f"x and y have different lengths ({n} vs {len(y)}).")
    if timestamps is not None and len(timestamps) != n:
        raise ValueError(
            f"timestamps has {len(timestamps)} values but x and y have {n}; there must be one timestamp per point."
        )

    fig = go.FigureWidget()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(size=2),
            line=dict(width=1),
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=TITLE_X,
            xanchor=TITLE_XANCHOR,
            yanchor=TITLE_YANCHOR,
            font=dict(size=FONT_SIZE_LARGE),
        ),
        xaxis_title=dict(text=x_label, font=dict(size=FONT_SIZE_MEDIUM)),
        yaxis_title=dict(text=y_label, font=dict(size=FONT_SIZE_MEDIUM)),
        showlegend=False,
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        xaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL)),
    )

    range_slider = widgets.IntRangeSlider(
        value=[0, n - 1],
        min=0,
        max=n - 1,
        step=1,
        description="Trim:",
        continuous_update=True,
        layout=widgets.Layout(width="100%"),
    )

    def _label(idx: int) -> str:
        if timestamps is not None:
            return f"index: {idx} | timestamp: {timestamps[idx]} {timestamp_unit.value}"
        return f"index: {idx}"

    start_label = widgets.Label(value=f"Start {_label(0)}")
    end_label = widgets.Label(value=f"End {_label(n - 1)}")
    label_box = widgets.HBox(
        [start_label, end_label],
        layout=widgets.Layout(justify_content="space-between", width="90%"),
    )

    def on_range_change(change: dict) -> None:
        lo, hi = change["new"]
        with fig.batch_update():
            fig.data[0].x = x[lo : hi + 1]
            fig.data[0].y = y[lo : hi + 1]
        start_label.value = f"Start {_label(lo)}"
        end_label.value = f"End {_label(hi)}"

    range_slider.observe(on_range_change, names="value")

    return widgets.VBox([fig, range_slider, label_box])
