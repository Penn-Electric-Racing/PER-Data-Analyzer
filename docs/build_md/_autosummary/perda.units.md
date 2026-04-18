# perda.units

### *class* perda.units.Timescale(\*values)

Bases: `Enum`

Supported timestamp units for time-series data.

#### MS *= 'ms'*

#### S *= 's'*

#### US *= 'us'*

### perda.units.convert_time(timestamp, source_time_unit, target_time_unit)

Convert a timestamp between two timescale units.

* **Parameters:**
  * **timestamp** (*float* *|* *NDArray*) – Timestamp(s) to convert.
  * **source_time_unit** ([*Timescale*](#perda.units.Timescale)) – Unit of the input timestamp.
  * **target_time_unit** ([*Timescale*](#perda.units.Timescale)) – Desired output unit.
* **Returns:**
  Timestamp(s) in the target unit.
* **Return type:**
  float | NDArray

### Examples

```pycon
>>> convert_time(5000.0, Timescale.MS, Timescale.S)
5.0
```

### perda.units.in_to_m(value)

Convert a length value from inches to meters.

* **Parameters:**
  **value** (*float* *|* *NDArray*) – Length in inches.
* **Returns:**
  Length in meters.
* **Return type:**
  float | NDArray

### Examples

```pycon
>>> in_to_m(1.0)
0.0254
```

### perda.units.mph_to_m_per_s(value)

Convert a speed value from miles per hour to meters per second.

* **Parameters:**
  **value** (*float* *|* *NDArray*) – Speed in mph.
* **Returns:**
  Speed in m/s.
* **Return type:**
  float | NDArray

### Examples

```pycon
>>> mph_to_m_per_s(1.0)
0.44704
```
