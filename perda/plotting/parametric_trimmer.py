import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from ..units import Timescale, to_seconds
from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    TIME_COLORSCALE,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
    TRIMMER_MARKER_SIZE,
)


def plot_parametric_trimmer(
    x: NDArray[float64],
    y: NDArray[float64],
    timestamps: NDArray[np.int64],
    timestamp_unit: Timescale,
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    equal_aspect: bool = True,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> widgets.VBox:
    """
    Interactive parametric curve (y against x), colored by time, with a range slider for picking a time window.

    Drag the two slider handles to hide points outside the selected window. The label under the slider shows the window in
    seconds since the first point, the number of points, and the raw timestamps of the first and last visible point, which
    can be passed directly to ``trim_single_run_data(ts_start=..., ts_end=...)`` or ``DataInstance.trim``. Designed for
    Jupyter notebooks.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values, e.g. East position.
    y : NDArray[float64]
        Y-axis values, e.g. North position. Must have the same length as ``x``.
    timestamps : NDArray[np.int64]
        Raw timestamp of each point, in ``timestamp_unit``. Must have the same length as ``x``.
    timestamp_unit : Timescale
        Unit of ``timestamps``. Pass ``aly.data.timestamp_unit``.
    x_label : str, optional
        X-axis label. Default is ``"X"``.
    y_label : str, optional
        Y-axis label. Default is ``"Y"``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    equal_aspect : bool, optional
        Draw one unit in x the same length as one unit in y, so spatial data is not stretched. Default is True.
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    ipywidgets.VBox
        Widget containing the figure, the slider and the label. Display it as the last expression in a notebook cell or
        with ``display(widget)``. The figure is ``widget.children[0]`` and can be customized with ``update_layout``.

    Raises
    ------
    ValueError
        If ``x`` is empty, or ``y`` or ``timestamps`` does not have the same length as ``x``.

    Examples
    --------
    >>> from perda.plotting import plot_parametric_trimmer
    >>> pos_x, pos_y = aly.data["posX", "posY"]
    >>> widget = plot_parametric_trimmer(
    ...     pos_x.value_np, pos_y.value_np, pos_x.timestamp_np, aly.data.timestamp_unit, x_label="X (m)", y_label="Y (m)"
    ... )
    >>> display(widget)
    """
    n = len(x)
    if n == 0:
        raise ValueError("x and y are empty; there is nothing to trim.")
    if len(y) != n:
        raise ValueError(f"x and y have different lengths ({n} vs {len(y)}).")
    if len(timestamps) != n:
        raise ValueError(
            f"timestamps has {len(timestamps)} values but x and y have {n}; there must be one timestamp per point."
        )

    elapsed_s = to_seconds(
        (timestamps - timestamps[0]).astype(np.float64), timestamp_unit
    )

    fig = go.FigureWidget()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=TRIMMER_MARKER_SIZE,
                color=elapsed_s,
                colorscale=TIME_COLORSCALE,
                colorbar=dict(title="Time (s)"),
                showscale=True,
            ),
            hovertemplate="x: %{x}<br>y: %{y}<br>t: %{marker.color:.2f} s<extra></extra>",
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
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    range_slider = widgets.IntRangeSlider(
        value=[0, n - 1],
        min=0,
        max=n - 1,
        step=1,
        description="Trim:",
        continuous_update=True,
        readout=False,
        layout=widgets.Layout(width="100%"),
    )

    def window_text(first: int, last: int) -> str:
        return (
            f"{elapsed_s[first]:.2f} s to {elapsed_s[last]:.2f} s "
            f"({last - first + 1} points, {elapsed_s[last] - elapsed_s[first]:.1f} s) | "
            f"ts_start={timestamps[first]}, ts_end={timestamps[last]} ({timestamp_unit.value})"
        )

    window_label = widgets.Label(value=window_text(0, n - 1))

    def on_range_change(change: dict) -> None:
        first, last = change["new"]
        with fig.batch_update():
            fig.data[0].x = x[first : last + 1]
            fig.data[0].y = y[first : last + 1]
            fig.data[0].marker.color = elapsed_s[first : last + 1]
        window_label.value = window_text(first, last)

    range_slider.observe(on_range_change, names="value")

    return widgets.VBox([fig, range_slider, window_label])
