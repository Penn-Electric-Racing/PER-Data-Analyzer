# perda.utils.accel_calculator

### *pydantic model* perda.utils.accel_calculator.AccelSegmentResult

Bases: `BaseModel`

Result of a single segment marked as an acceleration test

* **Fields:**
  - [`dist_reached (float)`](#perda.utils.accel_calculator.AccelSegmentResult.dist_reached)
  - [`start_time (float)`](#perda.utils.accel_calculator.AccelSegmentResult.start_time)
  - [`time_to_dist (float)`](#perda.utils.accel_calculator.AccelSegmentResult.time_to_dist)
  - [`timescale (perda.units.Timescale)`](#perda.utils.accel_calculator.AccelSegmentResult.timescale)

#### *field* start_time *: `float`* *[Required]*

Timestamp when the segment began.

#### *field* time_to_dist *: `float`* *[Required]*

Time to travel dist_reached meters from segment start.

#### *field* dist_reached *: `float`* *[Required]*

Distance of the segment in meters.

#### *field* timescale *: [`Timescale`](perda.units.md#perda.units.Timescale)* *[Required]*

Time unit for the output times.

### perda.utils.accel_calculator.compute_accel_results(signal_obj, distance_obj, target_dist=75, source_time_unit=Timescale.MS, target_time_unit=Timescale.S)

Compute time-to-distance results for each acceleration event.

* **Parameters:**
  * **signal_obj** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Binary accel event signal (0.0/1.0), typically from detect_accel_event.
  * **distance_obj** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Cumulative distance signal in meters.
  * **target_dist** (*float* *,* *optional*) – Target distance in meters. Default is 75.
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit of input timestamps. Default is Timescale.MS.
  * **target_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit for output times. Default is Timescale.S.
* **Returns:**
  One result per qualifying segment.
* **Return type:**
  list[[AccelSegmentResult](#perda.utils.accel_calculator.AccelSegmentResult)]

### Examples

```pycon
>>> results = compute_accel_results(signal_di, distance_di)
>>> for r in results:
...     print(r)
```

### perda.utils.accel_calculator.detect_accel_event(torque_obj, speed_obj, torque_threshold=100, speed_threshold=0.5)

Detect acceleration events based on torque and speed thresholds.

An event is active when torque exceeds torque_threshold and speed exceeds
speed_threshold, and ends when speed drops back to or below speed_threshold.

* **Parameters:**
  * **torque_obj** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Time-series of motor torque values.
  * **speed_obj** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – Time-series of wheel speed values. The output is aligned to these timestamps.
  * **torque_threshold** (*float* *,* *optional*) – Minimum torque (in Nm) required to trigger an acceleration event. Default is 100.
  * **speed_threshold** (*float* *,* *optional*) – Speed value used as the trigger floor and reset condition. Default is 0.5.
* **Returns:**
  Binary signal (0.0 or 1.0) on speed_obj timestamps, labeled “Accel Event”,
  where 1.0 indicates an active acceleration event.
* **Return type:**
  [DataInstance](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)

### Examples

```pycon
>>> signal = detect_accel_event(torque_di, speed_di)
```
