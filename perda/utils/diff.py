from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..analyzer.single_run_data import SingleRunData
from ..analyzer.data_instance import DataInstance
from ..plotting.data_instance_plotter import plot_single_axis

def diff(
    base_data: SingleRunData,
    incom_data: SingleRunData,
    force_compare: bool = False,
    timestamp_tolerance_ms: int = 2,
    diff_rtol: float = 1e-3,
    diff_atol: float = 1e-3,
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
    union_cpp_name_to_instances: dict[str, tuple[DataInstance, DataInstance]] = {}

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
    point_present_pairs = []
    matched_base_points = 0
    matched_incom_points = 0

    total_points_to_compare = sum(
        len(base_di) + len(incom_di)
        for base_di, incom_di in union_cpp_name_to_instances.values()
    )
    pbar = tqdm(
        desc="Comparing points",
        unit=" pts",
        total=total_points_to_compare,
    )
    for cpp_name, (base_di, incom_di) in union_cpp_name_to_instances.items():
        base_extra_pair, incom_extra_pair, diff_pair = _get_diff_pairs(
            base_di,
            incom_di,
            timestamp_tolerance_ms=timestamp_tolerance_ms,
            diff_rtol=diff_rtol,
            diff_atol=diff_atol,
        )
        base_extra_pairs.append(base_extra_pair)
        incom_extra_pairs.append(incom_extra_pair)
        diff_pairs.append(diff_pair)
        present_ts, present_cnt = np.unique(
            np.concatenate((base_di.timestamp_np, incom_di.timestamp_np)),
            return_counts=True,
        )
        present_cnt = present_cnt.astype(np.float64)
        point_present_pairs.append((present_ts, present_cnt))
        matched_base_points += len(base_di)
        matched_incom_points += len(incom_di)
        pbar.update(len(base_di) + len(incom_di))
    pbar.close()
    
    def _accumulate_sparse_pairs(
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[int, float]:
        counter: defaultdict[int, float] = defaultdict(float)
        for ts_np, cnt_np in pairs:
            for ts, cnt in zip(ts_np, cnt_np):
                counter[int(ts)] += float(cnt)
        return dict(counter)

    point_present_counter = _accumulate_sparse_pairs(point_present_pairs)
    loss_counter: defaultdict[int, float] = defaultdict(float)
    for counter in (
        _accumulate_sparse_pairs(base_extra_pairs),
        _accumulate_sparse_pairs(incom_extra_pairs),
        _accumulate_sparse_pairs(diff_pairs),
    ):
        for ts, cnt in counter.items():
            loss_counter[int(ts)] += float(cnt)

    total_present_timestamps = sum(1 for cnt in point_present_counter.values() if cnt > 0)
    loss_timestamps = sum(
        1 for ts in point_present_counter if float(loss_counter.get(ts, 0.0)) > 0.0
    )
    no_loss_timestamps = total_present_timestamps - loss_timestamps

    base_extra_di, incom_extra_di, diff_di, all_point_count_di = _get_diff_dis(
        base_extra_pairs,
        incom_extra_pairs,
        diff_pairs,
        point_present_pairs,
    )

    # Block 3: Display
    base_extra_total = float(np.sum(base_extra_di.value_np))
    incom_extra_total = float(np.sum(incom_extra_di.value_np))
    diff_total = float(np.sum(diff_di.value_np))
    all_point_count_total = float(np.sum(all_point_count_di.value_np))

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
    print(f"Total point-presence points:      {all_point_count_total:.0f}")
    print(f"Base-only % of base:              {_pct(base_extra_total, matched_base_points_f):.2f}%")
    print(f"Incoming-only % of incoming:      {_pct(incom_extra_total, matched_incom_points_f):.2f}%")
    print(f"Value-mismatch % of base:         {_pct(diff_total, matched_base_points_f):.2f}%")
    print(f"Value-mismatch % of incoming:     {_pct(diff_total, matched_incom_points_f):.2f}%")
    print(f"Value-mismatch % of matched pairs:{_pct(diff_total, matched_pairs):.2f}%")
    print(f"Base-only % of point-presence:    {_pct(base_extra_total, all_point_count_total):.2f}%")
    print(f"Incoming-only % of point-presence:{_pct(incom_extra_total, all_point_count_total):.2f}%")
    print(f"Value-diff % of point-presence:   {_pct(diff_total, all_point_count_total):.2f}%")
    print(
        f"Timestamps no-loss/loss:          "
        f"{_pct(no_loss_timestamps, total_present_timestamps):.2f}% / "
        f"{_pct(loss_timestamps, total_present_timestamps):.2f}%"
    )
    print(
        f"Matcher params: tol={timestamp_tolerance_ms}ms, "
        f"rtol={diff_rtol}, atol={diff_atol}"
    )
    print("======================")

    if len(all_point_count_di) == 0:
        print("No differences found.")
        return

    fig = plot_single_axis(
        data_instances=[
            all_point_count_di,
            base_extra_di,
            incom_extra_di,
            diff_di,
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
    timestamp_tolerance_ms: int = 2,
    diff_rtol: float = 1e-3,
    diff_atol: float = 1e-3,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    ts_a = base_di.timestamp_np.astype(np.int64, copy=False)
    va = base_di.value_np.astype(np.float64, copy=False)
    ts_b = incom_di.timestamp_np.astype(np.int64, copy=False)
    vb = incom_di.value_np.astype(np.float64, copy=False)

    # Fast exact match shortcut
    if np.array_equal(ts_a, ts_b) and np.array_equal(va, vb, equal_nan=True):
        empty_ts = np.array([], dtype=np.int64)
        empty_cnt = np.array([], dtype=np.float64)
        return (empty_ts, empty_cnt), (empty_ts, empty_cnt), (empty_ts, empty_cnt)

    tol = max(int(timestamp_tolerance_ms), 0)

    base_extra_counter: defaultdict[int, float] = defaultdict(float)
    incom_extra_counter: defaultdict[int, float] = defaultdict(float)
    diff_counter: defaultdict[int, float] = defaultdict(float)

    def same_val(a: float, b: float) -> bool:
        return bool(np.isclose(a, b, rtol=diff_rtol, atol=diff_atol, equal_nan=True))

    i = 0
    j = 0
    n = ts_a.size
    m = ts_b.size

    while i < n and j < m:
        ta = int(ts_a[i])
        tb = int(ts_b[j])

        # Too far apart: count as extras
        if ta < tb - tol:
            base_extra_counter[ta] += 1.0
            i += 1
            continue
        if tb < ta - tol:
            incom_extra_counter[tb] += 1.0
            j += 1
            continue

        # Within tolerance: match points directly (monotonic two-pointer).
        if not same_val(va[i], vb[j]):
            diff_counter[ta] += 1.0
        i += 1
        j += 1

    # Remaining tails are extras
    while i < n:
        base_extra_counter[int(ts_a[i])] += 1.0
        i += 1
    while j < m:
        incom_extra_counter[int(ts_b[j])] += 1.0
        j += 1

    def to_pair(counter: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        if not counter:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        ts = np.fromiter(counter.keys(), dtype=np.int64)
        cnt = np.fromiter(counter.values(), dtype=np.float64)
        order = np.argsort(ts, kind="stable")
        return ts[order], cnt[order]

    return to_pair(base_extra_counter), to_pair(incom_extra_counter), to_pair(diff_counter)

def _get_diff_dis(
    base_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    incom_extra_pairs: list[tuple[np.ndarray, np.ndarray]],
    diff_pairs: list[tuple[np.ndarray, np.ndarray]],
    point_present_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
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

    if point_present_pairs is None:
        all_counter: defaultdict[int, float] = defaultdict(float)
        for counter in (base_counter, incom_counter, diff_counter):
            for ts, cnt in counter.items():
                all_counter[int(ts)] += float(cnt)
        point_counter = dict(all_counter)
    else:
        point_counter = _accumulate_sparse(point_present_pairs)

    base_extra_di = _counter_to_di(base_counter, "All Points Base Extra")
    incom_extra_di = _counter_to_di(incom_counter, "All Points Incoming Extra")
    diff_di = _counter_to_di(diff_counter, "All Points Value Diff")
    all_diff_di = _counter_to_di(point_counter, "All Points Present")

    return base_extra_di, incom_extra_di, diff_di, all_diff_di
