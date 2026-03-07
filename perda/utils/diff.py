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

    if len(all_diff_di) == 0:
        print("No differences found.")
        return

    fig = plot_single_axis(
        data_instances=[
            base_extra_di,
            incom_extra_di,
            diff_di,
            all_diff_di,
        ],
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
    if np.array_equal(base_di.timestamp_np, incom_di.timestamp_np) and np.array_equal(base_di.value_np, incom_di.value_np, equal_nan=True):
        empty_ts = np.array([], dtype=np.int64)
        empty_cnt = np.array([], dtype=np.float64)
        return (
            (empty_ts, empty_cnt),
            (empty_ts, empty_cnt),
            (empty_ts, empty_cnt),
        )
    
    base_unique_ts, base_first_idx, base_counts = np.unique(
        base_di.timestamp_np, return_index=True, return_counts=True
    )
    incom_unique_ts, incom_first_idx, incom_counts = np.unique(
        incom_di.timestamp_np, return_index=True, return_counts=True
    )

    base_extra_ts: list[int] = []
    base_extra_cnt: list[float] = []
    incom_extra_ts: list[int] = []
    incom_extra_cnt: list[float] = []
    diff_ts: list[int] = []
    diff_cnt: list[float] = []

    i = 0
    j = 0
    while i < base_unique_ts.size and j < incom_unique_ts.size:
        base_ts = int(base_unique_ts[i])
        incom_ts = int(incom_unique_ts[j])

        if base_ts < incom_ts:
            base_extra_ts.append(base_ts)
            base_extra_cnt.append(float(base_counts[i]))
            i += 1
            continue
        if incom_ts < base_ts:
            incom_extra_ts.append(incom_ts)
            incom_extra_cnt.append(float(incom_counts[j]))
            j += 1
            continue

        # same timestamp
        base_count = int(base_counts[i])
        incom_count = int(incom_counts[j])
        paired_count = min(base_count, incom_count)

        if base_count > paired_count:
            base_extra_ts.append(base_ts)
            base_extra_cnt.append(float(base_count - paired_count))
        if incom_count > paired_count:
            incom_extra_ts.append(incom_ts)
            incom_extra_cnt.append(float(incom_count - paired_count))

        if paired_count > 0:
            base_start = int(base_first_idx[i])
            base_end = base_start + base_count
            incom_start = int(incom_first_idx[j])
            incom_end = incom_start + incom_count

            base_values = np.sort(base_di.value_np[base_start:base_end].astype(np.float64, copy=False))
            incom_values = np.sort(incom_di.value_np[incom_start:incom_end].astype(np.float64, copy=False))
            same_mask = np.isclose(
                base_values[:paired_count],
                incom_values[:paired_count],
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )
            mismatch_count = int(np.count_nonzero(~same_mask))
            if mismatch_count > 0:
                diff_ts.append(base_ts)
                diff_cnt.append(float(mismatch_count))

        i += 1
        j += 1

    while i < base_unique_ts.size:
        base_extra_ts.append(int(base_unique_ts[i]))
        base_extra_cnt.append(float(base_counts[i]))
        i += 1

    while j < incom_unique_ts.size:
        incom_extra_ts.append(int(incom_unique_ts[j]))
        incom_extra_cnt.append(float(incom_counts[j]))
        j += 1

    return (
        (
            np.asarray(base_extra_ts, dtype=np.int64),
            np.asarray(base_extra_cnt, dtype=np.float64),
        ),
        (
            np.asarray(incom_extra_ts, dtype=np.int64),
            np.asarray(incom_extra_cnt, dtype=np.float64),
        ),
        (
            np.asarray(diff_ts, dtype=np.int64),
            np.asarray(diff_cnt, dtype=np.float64),
        ),
    )

def _get_diff_dis(
    base_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    incom_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    diff_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[DataInstance, DataInstance, DataInstance, DataInstance]:
    def _accumulate_sparse(
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[int, float]:
        counter: defaultdict[int, float] = defaultdict(float)
        for ts_np, cnt_np in pairs:
            for ts, cnt in zip(ts_np, cnt_np):
                counter[int(ts)] += float(cnt)
        return dict(counter)

    def _insert_boundary_zeros(
        ts_np: np.ndarray,
        cnt_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ts_np.size <= 1:
            return ts_np, cnt_np

        out_ts: list[int] = [int(ts_np[0])]
        out_cnt: list[float] = [float(cnt_np[0])]

        for idx in range(1, ts_np.size):
            prev_ts = int(ts_np[idx - 1])
            cur_ts = int(ts_np[idx])
            cur_cnt = float(cnt_np[idx])

            if cur_ts - prev_ts > 1:
                left_zero_ts = prev_ts + 1
                right_zero_ts = cur_ts - 1

                if left_zero_ts > prev_ts and left_zero_ts < cur_ts:
                    out_ts.append(left_zero_ts)
                    out_cnt.append(0.0)
                if right_zero_ts > out_ts[-1] and right_zero_ts < cur_ts:
                    out_ts.append(right_zero_ts)
                    out_cnt.append(0.0)

            out_ts.append(cur_ts)
            out_cnt.append(cur_cnt)

        return (
            np.asarray(out_ts, dtype=np.int64),
            np.asarray(out_cnt, dtype=np.float64),
        )

    def _counter_to_di(counter: dict[int, float], label: str) -> DataInstance:
        if not counter:
            return DataInstance(
                timestamp_np=np.array([], dtype=np.int64),
                value_np=np.array([], dtype=np.float64),
                label=label,
            )

        ts_np = np.asarray(sorted(counter.keys()), dtype=np.int64)
        cnt_np = np.asarray([counter[int(ts)] for ts in ts_np], dtype=np.float64)
        ts_plot, cnt_plot = _insert_boundary_zeros(ts_np, cnt_np)
        return DataInstance(
            timestamp_np=ts_plot,
            value_np=cnt_plot,
            label=label,
        )

    base_counter = _accumulate_sparse(base_extra_pairs)
    incom_counter = _accumulate_sparse(incom_extra_pairs)
    diff_counter = _accumulate_sparse(diff_pairs)

    all_counter: defaultdict[int, float] = defaultdict(float)
    for counter in (base_counter, incom_counter, diff_counter):
        for ts, cnt in counter.items():
            all_counter[int(ts)] += float(cnt)

    base_extra_di = _counter_to_di(base_counter, "All Variables Base Extra")
    incom_extra_di = _counter_to_di(incom_counter, "All Variables Incoming Extra")
    diff_di = _counter_to_di(diff_counter, "All Variables Value Diff")
    all_diff_di = _counter_to_di(dict(all_counter), "All Variables Total Diff")

    return base_extra_di, incom_extra_di, diff_di, all_diff_di
