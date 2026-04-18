# perda.core_data_structures.single_run_data

### *pydantic model* perda.core_data_structures.single_run_data.SingleRunData

Bases: `BaseModel`

Pydantic model to store parsed CSV data with dictionary-like lookup.

* **Config:**
  - **arbitrary_types_allowed**: *bool = True*
* **Fields:**
  - [`concat_boundaries (List[int])`](#perda.core_data_structures.single_run_data.SingleRunData.concat_boundaries)
  - [`cpp_name_to_id (Dict[str, int])`](#perda.core_data_structures.single_run_data.SingleRunData.cpp_name_to_id)
  - [`data_end_time (int)`](#perda.core_data_structures.single_run_data.SingleRunData.data_end_time)
  - [`data_start_time (int)`](#perda.core_data_structures.single_run_data.SingleRunData.data_start_time)
  - [`id_to_cpp_name (Dict[int, str])`](#perda.core_data_structures.single_run_data.SingleRunData.id_to_cpp_name)
  - [`id_to_descript (Dict[int, str])`](#perda.core_data_structures.single_run_data.SingleRunData.id_to_descript)
  - [`id_to_instance (Dict[int, perda.core_data_structures.data_instance.DataInstance])`](#perda.core_data_structures.single_run_data.SingleRunData.id_to_instance)
  - [`timestamp_unit (perda.units.Timescale)`](#perda.core_data_structures.single_run_data.SingleRunData.timestamp_unit)
  - [`total_data_points (int)`](#perda.core_data_structures.single_run_data.SingleRunData.total_data_points)

#### *field* id_to_instance *: `Dict`[`int`, [`DataInstance`](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)]* *[Required]*

Mapping from variable ID to DataInstance

#### *field* cpp_name_to_id *: `Dict`[`str`, `int`]* *[Required]*

Mapping from variable name to variable ID

#### *field* id_to_cpp_name *: `Dict`[`int`, `str`]* *[Required]*

Mapping from variable ID to variable name

#### *field* id_to_descript *: `Dict`[`int`, `str`]* *[Required]*

Mapping from variable ID to variable description

#### *field* total_data_points *: `int`* *[Required]*

Total number of data points across all variables

#### *field* data_start_time *: `int`* *[Required]*

Start timestamp in log timestamp unit

#### *field* data_end_time *: `int`* *[Required]*

End timestamp in log timestamp unit

#### *field* timestamp_unit *: [`Timescale`](perda.units.md#perda.units.Timescale)* *= Timescale.MS*

Timestamp logging unit for this run (ms/us)

#### *field* concat_boundaries *: `List`[`int`]* *[Optional]*

Timestamps where concatenated runs begin (post-shift)
