import os
from typing import Any, Dict, List
import numpy as np
import plotly.graph_objects as go

from ..plotting.plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
)
from ..units import _to_seconds


def plot_comparison(
    analyzers: List[Any], cpp_name: str, max_points: int | None = None
) -> go.Figure:
    """Compare and overlay a variable from multiple runs on a single Plotly figure.

    Aligns each run's timestamps to start at relative seconds (t = 0).

    Parameters
    ----------
    analyzers : List[Analyzer]
        List of Analyzer instances to compare.
    cpp_name : str
        C++ variable name to plot and overlay.
    max_points : int | None, optional
        Maximum number of data points to plot per run (enables zero-copy step downsampling). Default is None (full fidelity).

    Returns
    -------
    go.Figure
        Plotly Figure with overlaid traces from all matching runs.
    """
    fig = go.Figure()

    for aly in analyzers:
        if cpp_name not in aly.data:
            continue
        di = aly.data[cpp_name]
        if len(di) == 0:
            continue

        # Shift timestamps relative to start time and convert to seconds
        raw_start = aly.data.data_start_time
        rel_timestamps_raw = di.timestamp_np - raw_start

        # Downsample coordinates if data points exceed the limit
        n_points = len(di)
        if max_points is not None and n_points > max_points and max_points > 0:
            step = (n_points + max_points - 1) // max_points
            timestamps_s = _to_seconds(rel_timestamps_raw[::step], aly.data.timestamp_unit)
            values_np = di.value_np[::step]
        else:
            timestamps_s = _to_seconds(rel_timestamps_raw, aly.data.timestamp_unit)
            values_np = di.value_np

        # Get file name for legend label
        file_path = getattr(aly, "file_path", None) or getattr(aly, "filename", None) or "Run"
        run_label = os.path.basename(str(file_path))

        fig.add_trace(
            go.Scattergl(
                x=timestamps_s,
                y=values_np,
                mode="lines",
                name=f"{run_label} ({di.label})",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Multi-Log Comparison: {cpp_name}",
            font=dict(size=DEFAULT_FONT_CONFIG.large)
        ),
        xaxis_title=dict(
            text="Time (seconds from start)",
            font=dict(size=DEFAULT_FONT_CONFIG.medium)
        ),
        yaxis_title=dict(
            text=cpp_name,
            font=dict(size=DEFAULT_FONT_CONFIG.medium)
        ),
        template="plotly_dark",
        width=DEFAULT_LAYOUT_CONFIG.width,
        height=DEFAULT_LAYOUT_CONFIG.height,
        margin=DEFAULT_LAYOUT_CONFIG.margin,
    )
    return fig


def compare_summary(analyzers: List[Any], cpp_name: str) -> Dict[str, Dict[str, float]]:
    """Compare basic statistics of a variable across multiple logs.

    Parameters
    ----------
    analyzers : List[Analyzer]
        List of Analyzer instances to compare.
    cpp_name : str
        C++ variable name to summarize.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary mapping run filenames to statistical descriptors:
        {"run_filename.csv": {"min": x, "max": y, "mean": z, "std": w, "len": n}}
    """
    comparison = {}
    for aly in analyzers:
        if cpp_name not in aly.data:
            continue
        di = aly.data[cpp_name]
        if len(di) == 0:
            continue

        vals = di.value_np
        file_path = getattr(aly, "file_path", None) or getattr(aly, "filename", None) or "Run"
        run_label = os.path.basename(str(file_path))

        comparison[run_label] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "len": int(len(vals)),
        }
    return comparison
