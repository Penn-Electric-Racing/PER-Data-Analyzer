# perda.core_data_structures.resampling

### *class* perda.core_data_structures.resampling.ResampleMethod(\*values)

Bases: `str`, `Enum`

Interpolation strategy used when aligning a series to a target timestamp grid.

#### CUBIC *= 'cubic'*

#### LINEAR *= 'linear'*

#### NEAREST *= 'nearest'*

#### ZOH *= 'zoh'*

### perda.core_data_structures.resampling.resample_to_freq(ts, val, freq_hz, timestamp_divisor, method=ResampleMethod.LINEAR)

Resample a time series onto a uniform frequency grid.

* **Parameters:**
  * **ts** (*NDArray*) – Source timestamps (int64)
  * **val** (*NDArray*) – Source values
  * **freq_hz** (*float*) – Target sampling frequency in Hz
  * **timestamp_divisor** (*float*) – Raw timestamp units per second (e.g. 1e6 for microseconds)
  * **method** ([*ResampleMethod*](#perda.core_data_structures.resampling.ResampleMethod) *,* *optional*) – Interpolation method. Default is LINEAR.
* **Return type:**
  `Tuple`[`ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]], `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]]
* **Returns:**
  * **target_ts** (*NDArray*) – Uniform timestamp grid (int64)
  * **resampled_val** (*NDArray*) – Values interpolated onto the uniform grid
