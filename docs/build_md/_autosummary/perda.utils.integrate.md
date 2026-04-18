# perda.utils.integrate

### perda.utils.integrate.average_over_time_range(data_instance, start_time=0, end_time=-1, source_time_unit=Timescale.MS, target_time_unit=Timescale.S)

Get average value over time using integral divided by time range.

* **Parameters:**
  * **data_instance** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – DataInstance to average
  * **start_time** (*int* *,* *optional*) – Start time for averaging. Default is 0
  * **end_time** (*int* *,* *optional*) – End time for averaging. -1 means end of data. Default is -1
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit of inputs. Default is Timescale.MS
  * **target_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Target time unit for averaging. Default is Timescale.S
* **Returns:**
  Time-weighted average value over the time range
* **Return type:**
  float

### Notes

Uses numpy.trapezoid (trapezoidal rule) for numerical integration of discrete data.

### perda.utils.integrate.get_data_slice_by_timestamp(original_instance, start_time=0, end_time=-1)

Get a new DataInstance with data in [start_time, end_time).

* **Parameters:**
  * **original_instance** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Original DataInstance to slice
  * **start_time** (*int* *,* *optional*) – Start time (inclusive). Default is 0
  * **end_time** (*int* *,* *optional*) – End time (exclusive). -1 means till end. Default is -1
* **Returns:**
  New DataInstance containing only data within the specified time range
* **Return type:**
  [DataInstance](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)

### perda.utils.integrate.integrate_over_time_range(data_instance, start_time=0, end_time=-1, source_time_unit=Timescale.MS, target_time_unit=Timescale.S)

Get integral of the value over time using the trapezoidal rule.

* **Parameters:**
  * **data_instance** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – DataInstance to integrate
  * **start_time** (*int* *,* *optional*) – Start time for integration. Default is 0
  * **end_time** (*int* *,* *optional*) – End time for integration. -1 means end of data. Default is -1
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit of inputs. Default is Timescale.MS
  * **target_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Target time unit for averaging. Default is Timescale.S
* **Returns:**
  Integral of the value over the time range
* **Return type:**
  float

### Notes

Uses numpy.trapezoid (trapezoidal rule) for numerical integration of discrete data.

### perda.utils.integrate.smoothed_filtered_integration(data, source_time_unit=Timescale.US, target_time_unit=Timescale.S, filter_window_size=10, n_sigmas=3, smoothing_window_len=11, smoothing_poly_order=2)

Integrate a time-series signal after outlier removal and smoothing.

Cleans spikes via rolling MAD-based outlier detection, applies Savitzky-Golay
smoothing, then computes the cumulative trapezoidal integral over time.

* **Parameters:**
  * **data** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Time-series signal to integrate.
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale)) – Time unit of inputs. Default is Timescale.US.
  * **target_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale)) – Target time unit for integration result. Default is Timescale.S.
  * **filter_window_size** (*int*) – Rolling window size used for median and MAD outlier detection.
  * **n_sigmas** (*float*) – Number of scaled MAD units beyond which a point is considered an outlier.
  * **smoothing_window_len** (*int*) – Window length for Savitzky-Golay smoothing filter.
  * **smoothing_poly_order** (*int*) – Polynomial order for Savitzky-Golay smoothing filter.
* **Returns:**
  A tuple of (timestamps, smoothed values, cumulative integral), all of the
  same length as the input signal.
* **Return type:**
  tuple[NDArray[Float64], NDArray[Float64], NDArray[Float64]]
