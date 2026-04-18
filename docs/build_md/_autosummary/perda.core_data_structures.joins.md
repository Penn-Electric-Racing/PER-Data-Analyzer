# perda.core_data_structures.joins

### perda.core_data_structures.joins.inner_join(left_ts, left_val, right_ts, right_val, , tolerance, method=ResampleMethod.LINEAR)

Inner join: keep only left timestamps that have a right timestamp within tolerance,
then interpolate right values onto those timestamps.

* **Parameters:**
  * **left_ts** (*NDArray*) – Timestamps for left series
  * **left_val** (*NDArray*) – Values for left series
  * **right_ts** (*NDArray*) – Timestamps for right series
  * **right_val** (*NDArray*) – Values for right series
  * **tolerance** (*float*) – Maximum allowed distance to the nearest right timestamp for a left
    timestamp to be kept.
  * **method** ([*ResampleMethod*](perda.core_data_structures.resampling.md#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method for right values. Default is LINEAR.
* **Return type:**
  `Tuple`[`ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]]
* **Returns:**
  * **timestamps** (*NDArray*) – Subset of left timestamps within tolerance of a right timestamp
  * **left_values** (*NDArray*) – Left values at the kept timestamps
  * **right_values** (*NDArray*) – Right values interpolated onto the kept timestamps

### perda.core_data_structures.joins.left_join(left_ts, left_val, right_ts, right_val, , method=ResampleMethod.LINEAR)

Left join: interpolate right values onto the left timestamp grid.

* **Parameters:**
  * **left_ts** (*NDArray*) – Timestamps for left series (used as the target grid)
  * **left_val** (*NDArray*) – Values for left series
  * **right_ts** (*NDArray*) – Timestamps for right series
  * **right_val** (*NDArray*) – Values for right series
  * **method** ([*ResampleMethod*](perda.core_data_structures.resampling.md#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method. Default is LINEAR.
* **Return type:**
  `Tuple`[`ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]]
* **Returns:**
  * **timestamps** (*NDArray*) – Left timestamps (unchanged)
  * **left_values** (*NDArray*) – Left values (unchanged)
  * **right_values** (*NDArray*) – Right values interpolated onto left timestamps

### perda.core_data_structures.joins.outer_join(left_ts, left_val, right_ts, right_val, , method=ResampleMethod.LINEAR, drop_nan=True, fill=0.0)

Outer join: union of timestamps with interpolation.

* **Parameters:**
  * **left_ts** (*NDArray*) – Timestamps for left series
  * **left_val** (*NDArray*) – Values for left series
  * **right_ts** (*NDArray*) – Timestamps for right series
  * **right_val** (*NDArray*) – Values for right series
  * **method** ([*ResampleMethod*](perda.core_data_structures.resampling.md#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method. Default is LINEAR.
  * **drop_nan** (*bool* *,* *optional*) – If True, drop rows where either series has NaN after interpolation.
    Default is True.
  * **fill** (*float* *,* *optional*) – Fill value for NaNs when drop_nan is False. Default is 0.0.
* **Return type:**
  `Tuple`[`ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]]
* **Returns:**
  * **timestamps** (*NDArray*) – Union of all timestamps
  * **left_values** (*NDArray*) – Left values interpolated to union timestamps
  * **right_values** (*NDArray*) – Right values interpolated to union timestamps
