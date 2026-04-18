# perda.core_data_structures.data_instance

### *pydantic model* perda.core_data_structures.data_instance.DataInstance

Bases: `BaseModel`

A single time-series variable, pairing a 1D timestamp array with a 1D value array.

* **Config:**
  - **arbitrary_types_allowed**: *bool = True*
* **Fields:**
  - [`cpp_name (str | None)`](#perda.core_data_structures.data_instance.DataInstance.cpp_name)
  - [`label (str | None)`](#perda.core_data_structures.data_instance.DataInstance.label)
  - [`timestamp_np (numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy._typing._array_like._ScalarT]])`](#perda.core_data_structures.data_instance.DataInstance.timestamp_np)
  - [`value_np (numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy._typing._array_like._ScalarT]])`](#perda.core_data_structures.data_instance.DataInstance.value_np)
  - [`var_id (int | None)`](#perda.core_data_structures.data_instance.DataInstance.var_id)
* **Validators:**
  - [`validate_timestamp`](#perda.core_data_structures.data_instance.DataInstance.validate_timestamp) » [`timestamp_np`](#perda.core_data_structures.data_instance.DataInstance.timestamp_np)
  - [`validate_value`](#perda.core_data_structures.data_instance.DataInstance.validate_value) » [`value_np`](#perda.core_data_structures.data_instance.DataInstance.value_np)

#### *field* timestamp_np *: `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]* *[Required]*

Timestamps as a 1D NumPy array

* **Validated by:**
  - [`validate_timestamp`](#perda.core_data_structures.data_instance.DataInstance.validate_timestamp)

#### *field* value_np *: `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]* *[Required]*

Values as a 1D NumPy array

* **Validated by:**
  - [`validate_value`](#perda.core_data_structures.data_instance.DataInstance.validate_value)

#### *field* label *: `str` | `None`* *= None*

Human-readable label for this variable

#### *field* var_id *: `int` | `None`* *= None*

Unique variable ID

#### *field* cpp_name *: `str` | `None`* *= None*

C++ variable name

#### *validator* validate_timestamp  *»*  [*timestamp_np*](#perda.core_data_structures.data_instance.DataInstance.timestamp_np)

Validate that timestamp array is 1-dimensional, positive, and strictly increasing.

* **Return type:**
  `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]
* **Parameters:**
  **v** (*Any*)

#### *validator* validate_value  *»*  [*value_np*](#perda.core_data_structures.data_instance.DataInstance.value_np)

Validate that value array is 1-dimensional

* **Return type:**
  `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]
* **Parameters:**
  **v** (*Any*)

#### model_post_init(\_DataInstance_\_context)

Post-initialization validation that timestamp and value arrays have the same length.

* **Return type:**
  `None`
* **Parameters:**
  **\_DataInstance_\_context** (*Any*)

#### trim(ts_start=None, ts_end=None)

Return a new DataInstance containing only points within the given timestamp range.

* **Parameters:**
  * **ts_start** (*float* *|* *None* *,* *optional*) – Lower bound in raw timestamp units (inclusive). Default is None (no lower bound).
  * **ts_end** (*float* *|* *None* *,* *optional*) – Upper bound in raw timestamp units (inclusive). Default is None (no upper bound).
* **Returns:**
  New DataInstance with only in-range data points.
* **Return type:**
  [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### Examples

```pycon
>>> clipped = di.trim(ts_start=10_000, ts_end=30_000)
```

### *class* perda.core_data_structures.data_instance.FilterOptions(\*values)

Bases: `Enum`

Specifies which array(s) a filter function receives as input.

#### BOTH *= 'both'*

#### TIMESTAMPS *= 'right_only'*

#### VALUES *= 'left_only'*

### perda.core_data_structures.data_instance.apply_ufunc_filter(data, filter_func, apply_to=FilterOptions.VALUES)

Apply a filter function to a DataInstance.

* **Parameters:**
  * **data** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Input DataInstance
  * **filter_func** (*Callable*) – Function that takes in values and/or timestamps and returns a boolean mask
  * **apply_to** ([*FilterOptions*](#perda.core_data_structures.data_instance.FilterOptions) *,* *optional*) – Whether to apply the filter to values, timestamps, or both. Default is values
* **Returns:**
  Filtered DataInstance
* **Return type:**
  [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### perda.core_data_structures.data_instance.apply_ufunc_inner_join(left, right, ufunc, , tolerance)

Apply a binary operation to two DataInstances using inner join.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Right DataInstance
  * **ufunc** (*Callable*) – NumPy universal function to apply (e.g., np.add, np.subtract)
  * **tolerance** (*float*) – Maximum allowed distance between left and right timestamps for a match.
* **Returns:**
  New DataInstance with combined values
* **Return type:**
  [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### perda.core_data_structures.data_instance.apply_ufunc_left_join(left, right, ufunc)

Apply a binary operation to two DataInstances using left join.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance (all timestamps are kept)
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Right DataInstance (values interpolated to left)
  * **ufunc** (*Callable*) – NumPy universal function to apply (e.g., np.add, np.subtract)
* **Returns:**
  New DataInstance with combined values
* **Return type:**
  [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### perda.core_data_structures.data_instance.apply_ufunc_outer_join(left, right, ufunc, , drop_nan=True, fill=0.0)

Apply a binary operation to two DataInstances using outer join.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Right DataInstance
  * **ufunc** (*Callable*) – NumPy universal function to apply (e.g., np.add, np.subtract)
  * **drop_nan** (*bool* *,* *optional*) – If True, drop rows where either series has NaN after interpolation.
    Default is True.
  * **fill** (*float* *,* *optional*) – Fill value for NaNs when drop_nan is False. Default is 0.0.
* **Returns:**
  New DataInstance with combined values
* **Return type:**
  [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### perda.core_data_structures.data_instance.inner_join_data_instances(left, right, , tolerance, method=ResampleMethod.LINEAR)

Inner join two DataInstances: keep only left timestamps with matching right timestamps.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Right DataInstance
  * **tolerance** (*float*) – Maximum allowed distance between left and right timestamps for a match.
    Timestamps with distance > tolerance are dropped.
  * **method** ([*ResampleMethod*](perda.core_data_structures.resampling.md#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method for right values. Default is LINEAR.
* **Return type:**
  `Tuple`[[`DataInstance`](#perda.core_data_structures.data_instance.DataInstance), [`DataInstance`](#perda.core_data_structures.data_instance.DataInstance)]
* **Returns:**
  * **left_result** (*DataInstance*) – Left DataInstance with only matched timestamps
  * **right_result** (*DataInstance*) – Right DataInstance with only matched timestamps

### perda.core_data_structures.data_instance.left_join_data_instances(left, right, , method=ResampleMethod.LINEAR)

Left join one or more DataInstances onto the left timestamp grid.

All right series are interpolated onto the left series timestamps.
To resample onto a uniform frequency grid first, call
`resample_to_freq` on the left instance before passing it here.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance (defines the target timestamp grid)
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance) *or* *list* *of* [*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – One or more right DataInstances to align to the left grid
  * **method** ([*ResampleMethod*](perda.core_data_structures.resampling.md#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method. Default is LINEAR.
* **Returns:**
  First element is the left DataInstance (unchanged), followed by one
  aligned DataInstance per right input, in the same order.
* **Return type:**
  tuple of [DataInstance](#perda.core_data_structures.data_instance.DataInstance)

### Examples

```pycon
>>> left_a, right_b, right_c = left_join_data_instances(a, [b, c])
>>> left_a, right_b = left_join_data_instances(a, b, method=ResampleMethod.ZOH)
```

### perda.core_data_structures.data_instance.outer_join_data_instances(left, right, , drop_nan=True, fill=0.0)

Outer join two DataInstances: union of timestamps with interpolation.

* **Parameters:**
  * **left** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Left DataInstance
  * **right** ([*DataInstance*](#perda.core_data_structures.data_instance.DataInstance)) – Right DataInstance
  * **drop_nan** (*bool* *,* *optional*) – If True, drop rows where either series has NaN after interpolation.
    Default is True.
  * **fill** (*float* *,* *optional*) – Fill value for NaNs when drop_nan is False. Default is 0.0.
* **Return type:**
  `Tuple`[[`DataInstance`](#perda.core_data_structures.data_instance.DataInstance), [`DataInstance`](#perda.core_data_structures.data_instance.DataInstance)]
* **Returns:**
  * **left_result** (*DataInstance*) – Left DataInstance with values interpolated to union timestamps
  * **right_result** (*DataInstance*) – Right DataInstance with values interpolated to union timestamps
