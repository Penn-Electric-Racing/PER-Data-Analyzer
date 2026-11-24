from collections.abc import Sequence

import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from .defaults import (
    FFT_LINE_COLOR,
    FONT_SIZE_LARGE,
    FONT_SIZE_MEDIUM,
    FONT_SIZE_SMALL,
    MARGIN,
    PLOT_BGCOLOR,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    SUBPLOT_HEIGHT,
    SUBPLOT_VERTICAL_SPACING,
    TITLE_X,
    TITLE_XANCHOR,
    TITLE_YANCHOR,
)


def plot_fft_spectrum(
    frequencies: Sequence[NDArray[float64]],
    magnitudes: Sequence[NDArray[float64]],
    series_names: Sequence[str],
    title: str | None = None,
    x_label: str = "Frequency (Hz)",
    y_label: str = "Magnitude",
    stacked: bool = True,
    log_x: bool = True,
    log_y: bool = False,
    width: int = PLOT_WIDTH,
    height: int | None = None,
    vertical_spacing: float = SUBPLOT_VERTICAL_SPACING,
) -> go.Figure:
    """Plot one or more pre-computed FFT magnitude spectra.

    Parameters
    ----------
    frequencies : Sequence[NDArray[float64]]
        Frequency axis (Hz) for each series, e.g. the first output of ``compute_fft``.
    magnitudes : Sequence[NDArray[float64]]
        Magnitude array for each series, same length as the matching ``frequencies`` entry.
    series_names : Sequence[str]
        Display name for each series. Used as subplot titles when ``stacked=True`` and as legend entries when ``stacked=False``.
    title : str | None, optional
        Title shown above the whole figure. Default is None (no title).
    x_label : str, optional
        X-axis label. Default is ``"Frequency (Hz)"``.
    y_label : str, optional
        Y-axis label. Default is ``"Magnitude"``.
    stacked : bool, optional
        If True, draw each series in its own subplot, stacked vertically with a shared frequency axis. If False, overlay all series on one plot. Default is True.
    log_x : bool, optional
        Use a logarithmic frequency (x) axis. Default is True.
    log_y : bool, optional
        Use a logarithmic magnitude (y) axis. Default is False (linear).
    width : int, optional
        Figure width in pixels. Default is PLOT_WIDTH (1200).
    height : int | None, optional
        Figure height in pixels. Default is None, which gives SUBPLOT_HEIGHT (350) pixels per series when stacked, or PLOT_HEIGHT (700) when overlaid.
    vertical_spacing : float, optional
        Gap between stacked subplots as a fraction of the figure height (0 to 1). Ignored when ``stacked=False``. Default is SUBPLOT_VERTICAL_SPACING (0.12).

    Returns
    -------
    go.Figure
        Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``.

    Examples
    --------
    >>> from perda.plotting import plot_fft_spectrum
    >>> from perda.utils import compute_fft
    >>> freqs, mags = compute_fft(di, source_time_unit=aly.data.timestamp_unit)
    >>> fig = plot_fft_spectrum([freqs], [mags], [di.label], log_x=False, log_y=True)
    """
    n = len(frequencies)
    if n == 0:
        raise ValueError("At least one series must be provided.")
    if not (len(magnitudes) == n == len(series_names)):
        raise ValueError("Inconsistent array lengths.")

    x_axis_type = "log" if log_x else "linear"
    y_axis_type = "log" if log_y else "linear"

    if stacked:
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"FFT: {name}" for name in series_names],
            vertical_spacing=vertical_spacing,
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
                    line=dict(color=FFT_LINE_COLOR),
                ),
                row=i,
                col=1,
            )

            fig.update_yaxes(
                title_text=y_label,
                title_font=dict(size=FONT_SIZE_MEDIUM),
                tickfont=dict(size=FONT_SIZE_SMALL),
                type=y_axis_type,
                row=i,
                col=1,
            )
            fig.update_xaxes(type=x_axis_type, row=i, col=1)

        fig.update_xaxes(
            title_text=x_label,
            title_font=dict(size=FONT_SIZE_MEDIUM),
            tickfont=dict(size=FONT_SIZE_SMALL),
            type=x_axis_type,
            row=n,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text=title,
                x=TITLE_X,
                xanchor=TITLE_XANCHOR,
                yanchor=TITLE_YANCHOR,
                font=dict(size=FONT_SIZE_LARGE),
            ),
            height=height if height is not None else SUBPLOT_HEIGHT * n,
            width=width,
            margin=MARGIN,
            plot_bgcolor=PLOT_BGCOLOR,
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
                x=TITLE_X,
                xanchor=TITLE_XANCHOR,
                yanchor=TITLE_YANCHOR,
                font=dict(size=FONT_SIZE_LARGE),
            ),
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=FONT_SIZE_MEDIUM)),
                tickfont=dict(size=FONT_SIZE_SMALL),
                type=x_axis_type,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(size=FONT_SIZE_MEDIUM)),
                tickfont=dict(size=FONT_SIZE_SMALL),
                type=y_axis_type,
            ),
            height=height if height is not None else PLOT_HEIGHT,
            width=width,
            plot_bgcolor=PLOT_BGCOLOR,
            showlegend=True,
            legend=dict(font=dict(size=FONT_SIZE_SMALL)),
        )

    return fig
