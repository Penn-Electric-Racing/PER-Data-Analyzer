import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray

from .plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FontConfig,
    LayoutConfig,
)


def plot_parametric_curve(
    x: NDArray[float64],
    y: NDArray[float64],
    x_label: str = "X",
    y_label: str = "Y",
    title: str | None = None,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """
    Plot a 2D parametric curve as a static Plotly figure.

    Parameters
    ----------
    x : NDArray[float64]
        X-axis values.
    y : NDArray[float64]
        Y-axis values. Must have the same length as `x`.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    title : str | None, optional
        Plot title.
    layout_config : LayoutConfig, optional
    font_config : FontConfig, optional

    Returns
    -------
    go.Figure

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
            x=layout_config.title_x,
            xanchor=layout_config.title_xanchor,
            yanchor=layout_config.title_yanchor,
            font=dict(size=font_config.large),
        ),
        xaxis_title=dict(text=x_label, font=dict(size=font_config.medium)),
        yaxis_title=dict(text=y_label, font=dict(size=font_config.medium)),
        showlegend=False,
        width=int(layout_config.width),
        height=int(layout_config.height),
        margin=layout_config.margin,
        plot_bgcolor=layout_config.plot_bgcolor,
        xaxis=dict(tickfont=dict(size=font_config.small)),
        yaxis=dict(tickfont=dict(size=font_config.small)),
    )
    return fig
