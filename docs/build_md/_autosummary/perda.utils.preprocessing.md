# perda.utils.preprocessing

### *class* perda.utils.preprocessing.Preprocessing(\*values)

Bases: `Enum`

#### CORRECT_MOTOR_DATA *= 'correct_motor_data'*

#### PATCH_NED_VELOCITY *= 'patch_ned_velocity'*

#### WHEELSPEEDS_TO_M_PER_S *= 'wheelspeeds_to_m_per_s'*

### perda.utils.preprocessing.apply_preprocessing(data, corrections)

Run a sequence of preprocessing steps on a parsed SingleRunData instance.

* **Parameters:**
  * **data** ([*SingleRunData*](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)) – Parsed run data to preprocess.
  * **corrections** (*list* *[*[*Preprocessing*](#perda.utils.preprocessing.Preprocessing) *]*) – Ordered list of preprocessing steps to apply.
* **Returns:**
  The same object, modified in-place and returned for chaining.
* **Return type:**
  [SingleRunData](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)

### perda.utils.preprocessing.convert_wheelspeeds_to_m_per_s(data)

* **Return type:**
  [`SingleRunData`](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)
* **Parameters:**
  **data** ([*SingleRunData*](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData))

### perda.utils.preprocessing.correct_motor_data(data)

* **Return type:**
  [`SingleRunData`](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)
* **Parameters:**
  **data** ([*SingleRunData*](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData))

### perda.utils.preprocessing.patch_ned_velocity(data)

Corrects a VectorNav configuration bug where velocityBody.x/y/z actually contains
NED (North/East/Down) velocities. Copies raw NED to velN/velE/velD, then rotates
into true body frame (FRD) using yaw. Includes sanity checks and ZOH duplicate warnings.

* **Parameters:**
  **data** ([*SingleRunData*](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData))
* **Return type:**
  [SingleRunData](perda.core_data_structures.single_run_data.md#perda.core_data_structures.single_run_data.SingleRunData)
