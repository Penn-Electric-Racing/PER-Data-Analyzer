import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

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


def plot_parametric_curve(
    x: NDArray[float64],
    y: NDArray[float64],
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """
    Plot a 2D parametric curve (y against x, in sample order), e.g. a GPS track.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values, e.g. longitude.
    y : NDArray[float64]
        Y-axis values, e.g. latitude. Must have the same length as ``x``.
    x_label : str, optional
        X-axis label. Default is ``"X"``.
    y_label : str, optional
        Y-axis label. Default is ``"Y"``.
    title : str | None, optional
        Title shown above the plot. Default is None (no title).
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
        Plotly figure. Change anything not exposed here (fonts, colors, margins, equal axis scaling, ...) with
        ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling
        ``fig.show()``.

    Examples
    --------
    >>> from perda.plotting import plot_parametric_curve
    >>> fig = plot_parametric_curve(lon_arr, lat_arr, x_label="Longitude", y_label="Latitude")
    """
    fig = go.Figure()
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
        xaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_x else "linear"),
        yaxis=dict(tickfont=dict(size=FONT_SIZE_SMALL), type="log" if log_y else "linear"),
    )
    return fig
