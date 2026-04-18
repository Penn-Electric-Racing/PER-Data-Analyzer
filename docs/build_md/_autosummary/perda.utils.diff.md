# perda.utils.diff

### perda.utils.diff.aggregate_timestamps(ts_list)

Concatenate timestamp arrays and count occurrences per unique timestamp.

* **Parameters:**
  **ts_list** (*list* *[**npt.NDArray* *[**np.int64* *]* *]*) – Each element is a 1-D int64 array of timestamps (one per event).
* **Returns:**
  (unique_timestamps, counts), both sorted by timestamp.
* **Return type:**
  tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]

### perda.utils.diff.diff(rpi_data, server_data, timestamp_tolerance_ms=2, diff_rtol=0.001, diff_atol=0.001)

Compare two SingleRunData objects and report differences.

Performs a three-stage comparison:

1. Variable-name alignment: reports C++ names present in one run but not the other
2. Point-level diff: for each common variable, classifies every data point
   as a base-only extra, incoming-only extra, value mismatch, or match.
3. Summary + plot: prints a diff summary table and displays an interactive
   Plotly bar chart of per-bucket diff counts.

* **Parameters:**
  * **rpi_data** ([`SingleRunData`](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)) – The reference (baseline) run.
  * **server_data** ([`SingleRunData`](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)) – The incoming run to compare against the baseline.
  * **timestamp_tolerance_ms** (`int`) – Maximum timestamp delta (ms) to consider two points as matching.
    Defaults to 2.
  * **diff_rtol** (`float`) – Relative tolerance for value comparison via `np.isclose`. Defaults to 1e-3.
  * **diff_atol** (`float`) – Absolute tolerance for value comparison via `np.isclose`. Defaults to 1e-3.
* **Return type:**
  `Figure`
