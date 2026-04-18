# perda.utils.data_summary

### perda.utils.data_summary.data_instance_summary(data_instance, source_time_unit=Timescale.MS, target_time_unit=Timescale.S)

Print information about a DataInstance.

* **Parameters:**
  * **data_instance** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – The DataInstance to print info about
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit of data instance. Default is milliseconds.
  * **target_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit used by the summary. Default is Timescale.S.
* **Return type:**
  `None`

### perda.utils.data_summary.single_run_summary(data, time_unit=Timescale.S)

Print overall information about the SingleRunData.

* **Parameters:**
  * **data** ([*SingleRunData*](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)) – Data structure containing CSV file data
  * **time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Time unit used by the summary. Default is Timescale.S.
* **Return type:**
  `None`
