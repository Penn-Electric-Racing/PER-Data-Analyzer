import numba
import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go
from tqdm import tqdm

from ..constants import DELIMITER, title_block
from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..plotting.diff_plotter import plot_diff_bars
from ..plotting.plotting_constants import *
from ..units import _to_seconds


def _pct(num: float, den: float) -> float:
    """Return num/den as a percentage, or 0.0 if den is non-positive."""
    return 0.0 if den <= 0 else (num / den) * 100.0


@numba.njit(cache=True)
def _get_diff_timestamps_core(
    ts_a: npt.NDArray[np.float64],
    va: npt.NDArray[np.float64],
    ts_b: npt.NDArray[np.float64],
    vb: npt.NDArray[np.float64],
    tol: np.float64,
    diff_rtol: float,
    diff_atol: float,
) -> tuple[
    numba.typed.List,
    numba.typed.List,
    numba.typed.List,
    numba.typed.List,
    int,
    int,
]:
    """JIT-compiled two-pointer core; returns typed lists and final (i, j) indices."""
    rpi_extra = numba.typed.List.empty_list(numba.float64)
    server_extra = numba.typed.List.empty_list(numba.float64)
    diff_ts = numba.typed.List.empty_list(numba.float64)
    matched_ts = numba.typed.List.empty_list(numba.float64)

    i = 0
    j = 0
    n = len(ts_a)
    m = len(ts_b)
    while i < n and j < m:
        if ts_a[i] < ts_b[j] - tol:
            rpi_extra.append(ts_a[i])
            i += 1
            continue
        if ts_b[j] < ts_a[i] - tol:
            server_extra.append(ts_b[j])
            j += 1
            continue

        # Matched pair within timestamp tolerance — inline np.isclose
        a_val = va[i]
        b_val = vb[j]
        a_nan = a_val != a_val  # noqa: PLR0124
        b_nan = b_val != b_val  # noqa: PLR0124
        if a_nan and b_nan:
            values_close = True
        elif a_nan or b_nan:
            values_close = False
        else:
            values_close = abs(a_val - b_val) <= diff_atol + diff_rtol * abs(b_val)
        if not values_close:
            diff_ts.append(ts_a[i])
        else:
            matched_ts.append(ts_a[i])
        i += 1
        j += 1

    return rpi_extra, server_extra, diff_ts, matched_ts, i, j


def _get_diff_timestamps(
    base_ts: npt.NDArray[np.float64],
    base_vals: npt.NDArray[np.float64],
    incom_ts: npt.NDArray[np.float64],
    incom_vals: npt.NDArray[np.float64],
    timestamp_tolerance_s: float = 0.002,
    diff_rtol: float = 1e-3,
    diff_atol: float = 1e-3,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compare two time-series point-by-point and classify each point.

    Uses a two-pointer walk over sorted timestamps. Points within
    ``timestamp_tolerance_s`` of each other are considered matched; their
    values are then compared with ``diff_rtol``/``diff_atol``. Points with no
    match in the other series are counted as extras.

    Parameters
    ----------
    base_ts:
        Sorted float64 timestamp array for the base run (seconds).
    base_vals:
        float64 value array aligned with ``base_ts``.
    incom_ts:
        Sorted float64 timestamp array for the incoming run (seconds).
    incom_vals:
        float64 value array aligned with ``incom_ts``.
    timestamp_tolerance_s:
        Maximum timestamp difference (seconds) to treat two points as matching.
    diff_rtol:
        Relative tolerance for value comparison.
    diff_atol:
        Absolute tolerance for value comparison.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (base_extra_ts, incom_extra_ts, value_mismatch_ts, matched_ts)
    """
    ts_a = base_ts.astype(np.float64, copy=False)
    va = base_vals.astype(np.float64, copy=False)
    ts_b = incom_ts.astype(np.float64, copy=False)
    vb = incom_vals.astype(np.float64, copy=False)

    # Fast exact match shortcut
    if np.array_equal(ts_a, ts_b) and np.array_equal(va, vb, equal_nan=True):
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, ts_a.copy()

    tol = np.float64(max(timestamp_tolerance_s, 0.0))

    rpi_extra_list, server_extra_list, diff_ts_list, matched_ts_list, i, j = (
        _get_diff_timestamps_core(ts_a, va, ts_b, vb, tol, diff_rtol, diff_atol)
    )

    rpi_extra = np.array(rpi_extra_list, dtype=np.float64)
    server_extra = np.array(server_extra_list, dtype=np.float64)
    diff_ts = np.array(diff_ts_list, dtype=np.float64)
    matched_ts = np.array(matched_ts_list, dtype=np.float64)

    tail_a = ts_a[i:]
    tail_b = ts_b[j:]
    return (
        np.concatenate([rpi_extra, tail_a]),
        np.concatenate([server_extra, tail_b]),
        diff_ts,
        np.concatenate([matched_ts, tail_a, tail_b]),
    )


def diff(
    rpi_data: SingleRunData,
    server_data: SingleRunData,
    timestamp_tolerance_s: float = 0.002,
    diff_rtol: float = 1e-3,
    diff_atol: float = 1e-3,
    diff_plot_config: DiffPlotConfig = DEFAULT_DIFF_PLOT_CONFIG,
    layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    font_config: FontConfig = DEFAULT_FONT_CONFIG,
) -> go.Figure:
    """Compare two SingleRunData objects and report differences.

    Performs a three-stage comparison:

    1. Variable-name alignment: reports C++ names present in one run but not the other
    2. Point-level diff: for each common variable, classifies every data point
       as a base-only extra, incoming-only extra, value mismatch, or match.
    3. Summary + plot: prints a diff summary table and displays an interactive
       Plotly bar chart of per-bucket diff counts.

    Timestamps from each run are converted to seconds using the run's
    ``timestamp_unit`` metadata before comparison, so runs with different
    logging units are handled correctly.

    Parameters
    ----------
    rpi_data:
        The reference (baseline) run.
    server_data:
        The incoming run to compare against the baseline.
    timestamp_tolerance_s:
        Maximum timestamp delta (seconds) to consider two points as matching.
        Defaults to 0.002 (2 ms).
    diff_rtol:
        Relative tolerance for value comparison via ``np.isclose``. Defaults to 1e-3.
    diff_atol:
        Absolute tolerance for value comparison via ``np.isclose``. Defaults to 1e-3.
    """
    base_cpp_name_to_id = rpi_data.cpp_name_to_id
    base_id_to_instance = rpi_data.id_to_instance

    server_cpp_name_to_id = server_data.cpp_name_to_id
    server_id_to_instance = server_data.id_to_instance

    # ===== Stage 1: Compare Variables =====
    in_rpi_not_in_server = []
    in_server_not_in_rpi = []
    # Shared C++ names -> (base instance, incoming instance)
    shared_cpp_name_to_instances: dict[str, tuple[DataInstance, DataInstance]] = {}

    for rpi_cpp_name, base_id in base_cpp_name_to_id.items():
        if rpi_cpp_name in server_cpp_name_to_id:
            shared_cpp_name_to_instances[rpi_cpp_name] = (
                base_id_to_instance[base_id],
                server_id_to_instance[server_cpp_name_to_id[rpi_cpp_name]],
            )
        else:
            in_rpi_not_in_server.append(rpi_cpp_name)
    for server_cpp_name in server_cpp_name_to_id:
        if server_cpp_name not in base_cpp_name_to_id:
            in_server_not_in_rpi.append(server_cpp_name)

    has_mismatch = bool(in_rpi_not_in_server or in_server_not_in_rpi)
    if has_mismatch:
        print(
            title_block(
                f"Mismatched Variables: {len(in_rpi_not_in_server) + len(in_server_not_in_rpi)}"
            )
        )
        if in_rpi_not_in_server:
            print(f"  Only in RPI:     {in_rpi_not_in_server}")
        if in_server_not_in_rpi:
            print(f"  Only in server: {in_server_not_in_rpi}")
        print()
    else:
        print(title_block("All C++ names match") + "\n")

    # ===== Stage 2: Compare DataInstances =====
    rpi_extra_ts_list: list[npt.NDArray[np.float64]] = []
    server_extra_ts_list: list[npt.NDArray[np.float64]] = []
    diff_ts_list: list[npt.NDArray[np.float64]] = []
    matched_ts_list: list[npt.NDArray[np.float64]] = []
    total_rpi_entries = 0
    total_server_entries = 0
    total_rpi_extra_entries = 0
    total_server_extra_entries = 0
    total_matched_entries = 0
    total_diff_entries = 0

    timestamps_compared = 0
    pbar = tqdm(
        shared_cpp_name_to_instances.items(),
        desc="Comparing matching variables",
        unit=" vars",
        total=len(shared_cpp_name_to_instances),
    )
    for _, (rpi_di, server_di) in pbar:
        rpi_ts_s = _to_seconds(
            rpi_di.timestamp_np.astype(np.float64), rpi_data.timestamp_unit
        )
        server_ts_s = _to_seconds(
            server_di.timestamp_np.astype(np.float64), server_data.timestamp_unit
        )
        rpi_extra_ts, server_extra_ts, var_diff_ts, var_matched_ts = (
            _get_diff_timestamps(
                rpi_ts_s,
                rpi_di.value_np,
                server_ts_s,
                server_di.value_np,
                timestamp_tolerance_s=timestamp_tolerance_s,
                diff_rtol=diff_rtol,
                diff_atol=diff_atol,
            )
        )
        rpi_extra_ts_list.append(rpi_extra_ts)
        server_extra_ts_list.append(server_extra_ts)
        diff_ts_list.append(var_diff_ts)
        matched_ts_list.append(var_matched_ts)
        total_rpi_entries += rpi_di.timestamp_np.size
        total_server_entries += server_di.timestamp_np.size
        total_rpi_extra_entries += rpi_extra_ts.size
        total_server_extra_entries += server_extra_ts.size
        total_matched_entries += var_matched_ts.size
        total_diff_entries += var_diff_ts.size
        timestamps_compared += rpi_di.timestamp_np.size + server_di.timestamp_np.size
        pbar.set_postfix({"timestamps": timestamps_compared})
    pbar.clear()
    pbar.close()

    rpi_extra_all = (
        np.concatenate(rpi_extra_ts_list)
        if rpi_extra_ts_list
        else np.array([], dtype=np.float64)
    )
    server_extra_all = (
        np.concatenate(server_extra_ts_list)
        if server_extra_ts_list
        else np.array([], dtype=np.float64)
    )
    diff_all = (
        np.concatenate(diff_ts_list) if diff_ts_list else np.array([], dtype=np.float64)
    )
    total_present_all = (
        np.concatenate(
            rpi_extra_ts_list + server_extra_ts_list + matched_ts_list + diff_ts_list
        )
        if (rpi_extra_ts_list + server_extra_ts_list + matched_ts_list + diff_ts_list)
        else np.array([], dtype=np.float64)
    )

    # ===== Stage 3: Summary + Plot =====
    rows = [
        ("Matched variables:", str(len(shared_cpp_name_to_instances))),
        ("Total RPI entries:", str(total_rpi_entries)),
        ("Total server entries:", str(total_server_entries)),
        ("Matched entries:", str(total_matched_entries)),
        ("RPI-only entries:", str(total_rpi_extra_entries)),
        ("Server-only entries:", str(total_server_extra_entries)),
        ("Value mismatch entries:", str(total_diff_entries)),
        (
            "RPI entries lost:",
            f"{_pct(total_rpi_extra_entries, total_rpi_entries):.3f}%",
        ),
        (
            "Server entries lost:",
            f"{_pct(total_server_extra_entries, total_server_entries):.3f}%",
        ),
    ]
    col_width = max(len(label) for label, _ in rows) + 2
    print(title_block("Diff Summary"))
    for label, value in rows:
        print(f"{label:<{col_width}}  {value}")
    print(DELIMITER)

    fig = plot_diff_bars(
        base_extra_ts=rpi_extra_all,
        incom_extra_ts=server_extra_all,
        value_mismatch_ts=diff_all,
        total_present_ts=total_present_all,
        title="Diff Counts Over Time",
        diff_plot_config=diff_plot_config,
        layout_config=layout_config,
        font_config=font_config,
    )

    return fig
