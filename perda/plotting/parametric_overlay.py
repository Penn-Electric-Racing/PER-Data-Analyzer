from typing import Literal

import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from .defaults import (
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    PARAMETRIC_MARKER_SIZE,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)

TraceMode = Literal["lines", "markers", "lines+markers"]


def plot_parametric_overlay(
    x_by_series: dict[str, NDArray[float64]],
    y_by_series: dict[str, NDArray[float64]],
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    mode: TraceMode = "lines",
    equal_aspect: bool = False,
    show_legend: bool = True,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Overlay several 2D parametric curves (y against x, in sample order) on one plot, e.g. tracks from multiple runs.

    Parameters
    ----------
    x_by_series : dict[str, NDArray[float64]]
        Maps each curve's legend name to its x values. Curves are drawn in dictionary order.
    y_by_series : dict[str, NDArray[float64]]
        Maps each curve's legend name to its y values. Must have the same keys as ``x_by_series``, and each array the same length as the matching x array.
    x_label : str, optional
        X-axis label. Default is ``"X"``.
    y_label : str, optional
        Y-axis label. Default is ``"Y"``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
    mode : {"lines", "markers", "lines+markers"}, optional
        Draw each curve as connected lines, as separate dots, or both. Use ``"markers"`` for noisy data where connecting lines would be misleading. Default is ``"lines"``.
    equal_aspect : bool, optional
        Lock the axes so one unit in x is drawn the same length as one unit in y. Use this for spatial data such as positions in metres, so shapes are not stretched. Default is False.
    show_legend : bool, optional
        Whether to show the legend listing each curve. Default is True.
    log_x : bool, optional
        Use a logarithmic x-axis. Default is False (linear).
    log_y : bool, optional
        Use a logarithmic y-axis. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int, optional
        Figure height in pixels. Default is PLOT_HEIGHT (700).

    Returns
    -------
    go.Figure
        Plotly figure with one trace per curve, named by its key. Change anything not exposed here (fonts, colors, marker size, ...) with ``fig.update_layout`` or ``fig.update_traces`` before calling ``fig.show()``.

    Raises
    ------
    ValueError
        If the two dictionaries have different keys, or a curve's x and y arrays have different lengths.

    Examples
    --------
    >>> from perda.plotting import plot_parametric_overlay
    >>> fig = plot_parametric_overlay(
    ...     {"Run 1": east_1, "Run 2": east_2},
    ...     {"Run 1": north_1, "Run 2": north_2},
    ...     x_label="East (m)",
    ...     y_label="North (m)",
    ...     equal_aspect=True,
    ... )
    """
    if x_by_series.keys() != y_by_series.keys():
        raise ValueError(
            f"x_by_series and y_by_series have different keys: {sorted(x_by_series)} vs {sorted(y_by_series)}."
        )
    for name, x in x_by_series.items():
        if len(x) != len(y_by_series[name]):
            raise ValueError(
                f"Curve '{name}' has {len(x)} x values but {len(y_by_series[name])} y values."
            )

    fig = go.Figure()
    for name, x in x_by_series.items():
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y_by_series[name],
                mode=mode,
                marker=dict(size=PARAMETRIC_MARKER_SIZE),
                name=name,
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
        showlegend=show_legend,
        width=width,
        height=height,
        margin=MARGIN,
        plot_bgcolor=PLOT_BGCOLOR,
        xaxis=dict(
            tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_x else "linear"
        ),
        yaxis=dict(
            tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_y else "linear"
        ),
        legend=dict(font=dict(size=FONT_SIZE_SMALL)),
    )
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig
