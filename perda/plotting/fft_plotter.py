from typing import List

import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from .plotting_constants import (
    DEFAULT_FFT_PLOT_CONFIG,
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    FFTPlotConfig,
    FontConfig,
    LayoutConfig,
)


def plot_fft_spectrum(
    frequencies: List[NDArray[float64]],
    magnitudes: List[NDArray[float64]],
    series_names: List[str],
    title: str | None = None,
    x_label: str = "Frequency (Hz)",
    y_label: str = "Magnitude",
    stacked: bool = True,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    fft_config: FFTPlotConfig = DEFAULT_FFT_PLOT_CONFIG,
) -> go.Figure:
    """Plot one or more pre-computed FFT magnitude spectra.

    Parameters
    ----------
    frequencies : list[NDArray[float64]]
        Frequency axis arrays, one per series.
    magnitudes : list[NDArray[float64]]
        Magnitude arrays (same length as corresponding ``frequencies``
        entry), one per series.
    series_names : list[str]
        Display labels for each series, used as subplot titles when
        ``stacked=True`` and legend entries when ``stacked=False``.
    title : str | None, optional
        Overall figure title.
    stacked : bool, optional
        If ``True``, render one subplot per series sharing the x-axis.
        If ``False``, overlay all series on a single plot. Default is True.
    layout_config : LayoutConfig, optional
        Figure dimensions and spacing.
    font_config : FontConfig, optional
        Font sizes for plot elements.
    fft_config : FFTPlotConfig, optional
        Axis scaling, trace color

    Returns
    -------
    go.Figure

    Examples
    --------
    >>> freqs, mags = compute_fft(di)
    >>> fig = plot_fft_spectrum([freqs], [mags], [di.label])
    >>> fig.show()
    """
    n = len(frequencies)
    if n == 0:
        raise ValueError("At least one series must be provided.")
    if not (len(magnitudes) == n == len(series_names)):
        raise ValueError("Inconsistent array lengths.")

    x_axis_type = "log" if fft_config.log_x else "linear"
    y_axis_type = "log" if fft_config.log_y else "linear"

    if stacked:
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"FFT: {name}" for name in series_names],
            vertical_spacing=layout_config.grid_vertical_spacing,
        )

        for i, (xf, yf, name) in enumerate(
            zip(frequencies, magnitudes, series_names), 1
        ):
            fig.add_trace(
                go.Scattergl(
                    x=xf,
                    y=yf,
                    mode="lines",
                    name=name,
                    line=dict(color=fft_config.line_color),
                ),
                row=i,
                col=1,
            )

            fig.update_yaxes(
                title_text=y_label,
                title_font=dict(size=font_config.medium),
                tickfont=dict(size=font_config.small),
                type=y_axis_type,
                row=i,
                col=1,
            )
            fig.update_xaxes(type=x_axis_type, row=i, col=1)

        fig.update_xaxes(
            title_text=x_label,
            title_font=dict(size=font_config.medium),
            tickfont=dict(size=font_config.small),
            type=x_axis_type,
            row=n,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_config.large),
            ),
            height=layout_config.grid_height_per_row * n,
            width=layout_config.width,
            margin=layout_config.margin,
            plot_bgcolor=layout_config.plot_bgcolor,
            showlegend=False,
        )
    else:
        fig = go.Figure()
        for xf, yf, name in zip(frequencies, magnitudes, series_names):
            fig.add_trace(
                go.Scattergl(
                    x=xf,
                    y=yf,
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_config.large),
            ),
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=font_config.medium)),
                tickfont=dict(size=font_config.small),
                type=x_axis_type,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(size=font_config.medium)),
                tickfont=dict(size=font_config.small),
                type=y_axis_type,
            ),
            height=layout_config.height,
            width=layout_config.width,
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(font=dict(size=font_config.small)),
        )

    return fig
