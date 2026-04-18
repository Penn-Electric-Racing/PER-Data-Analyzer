"""Generic FFT magnitude spectrum plotter."""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from .plotting_constants import (
    DEFAULT_FFT_PLOT_CONFIG,
    DEFAULT_FONT_CONFIG,
    DEFAULT_SUBPLOT_CONFIG,
    FFTPlotConfig,
    FontConfig,
    SubplotConfig,
)


def plot_fft_spectrum(
    frequencies: List[NDArray[float64]],
    magnitudes: List[NDArray[float64]],
    series_names: List[str],
    freq_unit: str = "Hz",
    title: str | None = None,
    stacked: bool = True,
    cutoff: float | None = None,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
    subplot_config: SubplotConfig = DEFAULT_SUBPLOT_CONFIG,
    fft_config: FFTPlotConfig = DEFAULT_FFT_PLOT_CONFIG,
) -> go.Figure:
    """Plot one or more pre-computed FFT magnitude spectra.

    Accepts frequency and magnitude arrays produced by any computation
    (e.g. ``compute_fft`` from ``perda.utils.filtering``) and renders
    them as an interactive Plotly figure.

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
    freq_unit : str, optional
        Unit label for the x-axis (e.g. ``"Hz"`` or ``"1/m"``).
        Default is ``"Hz"``.
    title : str | None, optional
        Overall figure title. Default is None.
    stacked : bool, optional
        If ``True``, render one subplot per series sharing the x-axis.
        If ``False``, overlay all series on a single plot. Default is True.
    cutoff : float | None, optional
        If given, draw a vertical dashed line at this frequency value.
        Default is None.
    font_config : FontConfig, optional
        Font sizes for plot elements. Default is DEFAULT_FONT_CONFIG.
    subplot_config : SubplotConfig, optional
        Row height and spacing used when ``stacked=True``.
        Default is DEFAULT_SUBPLOT_CONFIG.
    fft_config : FFTPlotConfig, optional
        Axis scaling, trace color, and cutoff line appearance.
        Default is DEFAULT_FFT_PLOT_CONFIG.

    Returns
    -------
    go.Figure

    Examples
    --------
    >>> freqs, mags, unit = compute_fft(di)
    >>> fig = plot_fft_spectrum([freqs], [mags], [di.label], freq_unit=unit)
    >>> fig.show()
    """
    n = len(frequencies)
    x_axis_type = "log" if fft_config.log_x else "linear"
    y_axis_type = "log" if fft_config.log_y else "linear"
    x_title = f"Frequency ({freq_unit})"

    if stacked:
        subplot_titles = [f"FFT: {name}" for name in series_names]
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                subplot_titles if subplot_config.show_subplot_titles else None
            ),
            vertical_spacing=subplot_config.vertical_spacing,
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
            if cutoff is not None:
                fig.add_vline(
                    x=cutoff,
                    line_dash=fft_config.cutoff_dash,
                    line_color=fft_config.cutoff_color,
                    annotation_text=f"{cutoff} {freq_unit}",
                    annotation_font_size=font_config.small,
                    row=i,
                    col=1,
                )
            fig.update_yaxes(
                title_text="Magnitude",
                title_font=dict(size=font_config.medium),
                tickfont=dict(size=font_config.small),
                type=y_axis_type,
                row=i,
                col=1,
            )

        fig.update_xaxes(
            title_text=x_title,
            title_font=dict(size=font_config.medium),
            tickfont=dict(size=font_config.small),
            type=x_axis_type,
            row=n,
            col=1,
        )
        # Apply x-axis type to all rows (shared_xaxes only links range, not type)
        for i in range(1, n):
            fig.update_xaxes(type=x_axis_type, row=i, col=1)

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_config.large),
            ),
            height=subplot_config.height_per_row * n,
            width=subplot_config.width,
            margin=subplot_config.margin,
            plot_bgcolor=subplot_config.plot_bgcolor,
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
        if cutoff is not None:
            fig.add_vline(
                x=cutoff,
                line_dash=fft_config.cutoff_dash,
                line_color=fft_config.cutoff_color,
                annotation_text=f"{cutoff} {freq_unit}",
                annotation_font_size=font_config.small,
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
                title=dict(text=x_title, font=dict(size=font_config.medium)),
                tickfont=dict(size=font_config.small),
                type=x_axis_type,
            ),
            yaxis=dict(
                title=dict(text="Magnitude", font=dict(size=font_config.medium)),
                tickfont=dict(size=font_config.small),
                type=y_axis_type,
            ),
            height=fft_config.height_single,
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(font=dict(size=font_config.small)),
        )

    return fig
