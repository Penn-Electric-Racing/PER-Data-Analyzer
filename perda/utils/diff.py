from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..analyzer.single_run_data import SingleRunData
from ..analyzer.data_instance import DataInstance
from ..plotting.data_instance_plotter import plot_single_axis

def diff(
    base_data: SingleRunData,
    incom_data: SingleRunData,
    force_compare: bool = False
) -> None:
    base_cpp_name_to_id = base_data.cpp_name_to_id
    base_id_to_instance = base_data.id_to_instance

    incom_cpp_name_to_id = incom_data.cpp_name_to_id
    incom_id_to_instance = incom_data.id_to_instance

    # Block 1: Compare C++ names

    # String list of mismatching C++ names
    in_base_not_in_incom = []
    in_incom_not_in_base = []

    # Common C++ names to their instances
    union_cpp_name_to_instances: defaultdict[
        str, tuple[DataInstance, DataInstance]
    ] = {}

    # Find common C++ names to their instances AND mismatch
    for base_cpp_name in base_cpp_name_to_id.keys():
        # Put matching instances into the union
        if base_cpp_name in incom_cpp_name_to_id:
            union_cpp_name_to_instances[base_cpp_name] = (
                base_id_to_instance[base_cpp_name_to_id[base_cpp_name]],
                incom_id_to_instance[incom_cpp_name_to_id[base_cpp_name]]
            )
        else:
            in_base_not_in_incom.append(base_cpp_name)
    for incom_cpp_name in incom_cpp_name_to_id.keys():
        if incom_cpp_name not in base_cpp_name_to_id:
            in_incom_not_in_base.append(incom_cpp_name)

    # Print mismatch information if any
    if in_base_not_in_incom or in_incom_not_in_base:
        print(f"Total mismatch C++ names: {len(in_base_not_in_incom) + len(in_incom_not_in_base)}")
        if in_base_not_in_incom:
            print(f" - In base but not in incoming: {in_base_not_in_incom}")
        if in_incom_not_in_base:
            print(f" - In incoming but not in base: {in_incom_not_in_base}")
        if not force_compare:
            # Abort if mismatch occur and force_compare is False
            print(" - Skipping data comparison due to mismatch")
            return
        else:
            print(" - Forcing data comparison despite mismatch")
    else:
        print("All C++ names match.")

    # Block 2: Compare datainstance
    print("Comparing DataInstances...")

    base_extra_pairs = []
    incom_extra_pairs = []
    diff_pairs = []
    matched_base_points = 0
    matched_incom_points = 0

    pbar = tqdm(
        union_cpp_name_to_instances.items(),
        desc="Comparing variables",
        unit=" vars",
        total=len(union_cpp_name_to_instances),
    )
    for cpp_name, (base_di, incom_di) in pbar:
        base_extra_pair, incom_extra_pair, diff_pair = _get_diff_pairs(base_di, incom_di)
        base_extra_pairs.append(base_extra_pair)
        incom_extra_pairs.append(incom_extra_pair)
        diff_pairs.append(diff_pair)
        matched_base_points += len(base_di)
        matched_incom_points += len(incom_di)
    
    base_extra_di, incom_extra_di, diff_di, all_diff_di = _get_diff_dis(base_extra_pairs, incom_extra_pairs, diff_pairs)

    # Block 3: Display
    base_extra_total = float(np.sum(base_extra_di.value_np))
    incom_extra_total = float(np.sum(incom_extra_di.value_np))
    diff_total = float(np.sum(diff_di.value_np))
    all_diff_total = float(np.sum(all_diff_di.value_np))

    matched_base_points_f = float(matched_base_points)
    matched_incom_points_f = float(matched_incom_points)
    matched_pairs_from_base = max(matched_base_points_f - base_extra_total, 0.0)
    matched_pairs_from_incom = max(matched_incom_points_f - incom_extra_total, 0.0)
    matched_pairs = min(matched_pairs_from_base, matched_pairs_from_incom)

    def _pct(num: float, den: float) -> float:
        return 0.0 if den <= 0 else (num / den) * 100.0

    print("==== Diff Summary ====")
    print(f"Matched C++ names:                {len(union_cpp_name_to_instances)}")
    print(f"Matched base points:              {matched_base_points}")
    print(f"Matched incoming points:          {matched_incom_points}")
    print(f"Base-only points:                 {base_extra_total:.0f}")
    print(f"Incoming-only points:             {incom_extra_total:.0f}")
    print(f"Value-mismatch points:            {diff_total:.0f}")
    print(f"Total diff points (all categories): {all_diff_total:.0f}")
    print(f"Base-only % of base:              {_pct(base_extra_total, matched_base_points_f):.2f}%")
    print(f"Incoming-only % of incoming:      {_pct(incom_extra_total, matched_incom_points_f):.2f}%")
    print(f"Value-mismatch % of base:         {_pct(diff_total, matched_base_points_f):.2f}%")
    print(f"Value-mismatch % of incoming:     {_pct(diff_total, matched_incom_points_f):.2f}%")
    print(f"Value-mismatch % of matched pairs:{_pct(diff_total, matched_pairs):.2f}%")
    print(f"All-diff % of base:               {_pct(all_diff_total, matched_base_points_f):.2f}%")
    print(f"All-diff % of incoming:           {_pct(all_diff_total, matched_incom_points_f):.2f}%")
    print("======================")

    fig = plot_single_axis(
        data_instances=[base_extra_di, incom_extra_di, diff_di, all_diff_di],
        title="Diff Counts Over Time",
        y_axis_title="Count per Timestamp",
        show_legend=True,
    )
    if fig is not None:
        fig.show()

    return

def _get_diff_pairs(
    base_di: DataInstance,
    incom_di: DataInstance,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    # Keep full union timestamp axis so zero-diff timestamps are preserved.
    all_ts = np.union1d(base_di.timestamp_np, incom_di.timestamp_np)

    base_unique_ts, base_first_idx, base_counts = np.unique(
        base_di.timestamp_np, return_index=True, return_counts=True
    )
    incom_unique_ts, incom_first_idx, incom_counts = np.unique(
        incom_di.timestamp_np, return_index=True, return_counts=True
    )

    base_cnt_aligned = np.zeros(all_ts.shape[0], dtype=np.float64)
    incom_cnt_aligned = np.zeros(all_ts.shape[0], dtype=np.float64)
    diff_cnt_aligned = np.zeros(all_ts.shape[0], dtype=np.float64)

    if base_unique_ts.size:
        base_idx = np.searchsorted(all_ts, base_unique_ts)
        base_cnt_aligned[base_idx] = base_counts.astype(np.float64)
    if incom_unique_ts.size:
        incom_idx = np.searchsorted(all_ts, incom_unique_ts)
        incom_cnt_aligned[incom_idx] = incom_counts.astype(np.float64)

    # Only compute value mismatches where timestamp exists in both.
    i = 0
    j = 0
    while i < base_unique_ts.size and j < incom_unique_ts.size:
        base_ts = int(base_unique_ts[i])
        incom_ts = int(incom_unique_ts[j])

        if base_ts < incom_ts:
            i += 1
            continue
        if incom_ts < base_ts:
            j += 1
            continue

        base_start = int(base_first_idx[i])
        base_end = base_start + int(base_counts[i])
        incom_start = int(incom_first_idx[j])
        incom_end = incom_start + int(incom_counts[j])

        base_values = base_di.value_np[base_start:base_end]
        incom_values = incom_di.value_np[incom_start:incom_end]
        paired_count = min(base_values.size, incom_values.size)

        if paired_count > 0:
            base_sorted = np.sort(base_values.astype(np.float64, copy=False))
            incom_sorted = np.sort(incom_values.astype(np.float64, copy=False))
            same_mask = np.isclose(
                base_sorted[:paired_count],
                incom_sorted[:paired_count],
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )
            mismatch_count = np.count_nonzero(~same_mask)
            diff_idx = np.searchsorted(all_ts, base_ts)
            diff_cnt_aligned[diff_idx] = float(mismatch_count)

        i += 1
        j += 1

    base_extra_cnt = np.maximum(base_cnt_aligned - incom_cnt_aligned, 0.0)
    incom_extra_cnt = np.maximum(incom_cnt_aligned - base_cnt_aligned, 0.0)

    return (
        (all_ts, base_extra_cnt),
        (all_ts, incom_extra_cnt),
        (all_ts, diff_cnt_aligned),
    )

def _get_diff_dis(
    base_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    incom_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    diff_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[DataInstance, DataInstance, DataInstance, DataInstance]:
    all_ts = np.array([], dtype=np.int64)
    for ts_np, _ in base_extra_pairs:
        all_ts = np.union1d(all_ts, ts_np)
    for ts_np, _ in incom_extra_pairs:
        all_ts = np.union1d(all_ts, ts_np)
    for ts_np, _ in diff_pairs:
        all_ts = np.union1d(all_ts, ts_np)

    base_total = np.zeros(all_ts.shape[0], dtype=np.float64)
    incom_total = np.zeros(all_ts.shape[0], dtype=np.float64)
    diff_total = np.zeros(all_ts.shape[0], dtype=np.float64)

    for ts_np, cnt_np in base_extra_pairs:
        idx = np.searchsorted(all_ts, ts_np)
        base_total[idx] += cnt_np
    for ts_np, cnt_np in incom_extra_pairs:
        idx = np.searchsorted(all_ts, ts_np)
        incom_total[idx] += cnt_np
    for ts_np, cnt_np in diff_pairs:
        idx = np.searchsorted(all_ts, ts_np)
        diff_total[idx] += cnt_np

    all_total = base_total + incom_total + diff_total

    base_extra_di = DataInstance(
        timestamp_np=all_ts,
        value_np=base_total,
        label="All Variables Base Extra",
    )
    incom_extra_di = DataInstance(
        timestamp_np=all_ts,
        value_np=incom_total,
        label="All Variables Incoming Extra",
    )
    diff_di = DataInstance(
        timestamp_np=all_ts,
        value_np=diff_total,
        label="All Variables Value Diff",
    )
    all_diff_di = DataInstance(
        timestamp_np=all_ts,
        value_np=all_total,
        label="All Variables Total Diff",
    )

    return base_extra_di, incom_extra_di, diff_di, all_diff_di
