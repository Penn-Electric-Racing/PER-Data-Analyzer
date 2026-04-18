# perda.core_data_structures.deduplication

### perda.core_data_structures.deduplication.deduplicate(data)

Remove consecutive duplicate values from a DataInstance.

* **Parameters:**
  **data** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance))
* **Return type:**
  [DataInstance](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)

### Examples

```pycon
>>> gps_deduped = deduplicate(analyzer.data["gps.lat"])
```
