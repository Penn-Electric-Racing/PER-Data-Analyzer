from ..constants import DELIMITER, title_block
from ..models.data_instance import DataInstance
from ..models.single_run_data import SingleRunData
from .integrate import average_over_time_range
from .units import *


def data_instance_summary(
    data_instance: DataInstance,
    source_time_unit: Timescale = Timescale.MS,
    target_time_unit: Timescale = Timescale.S,
) -> None:
    """
    Print information about a DataInstance.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance to print info about
    source_time_unit : Timescale, optional
        Time unit of data instance. Default is milliseconds.
    target_time_unit : Timescale, optional
        Time unit used by the summary. Default is Timescale.S.
    """
    print(str(data_instance))
    print(DELIMITER)

    if len(data_instance) > 0:
        min_val = float(data_instance.value_np.min())
        max_val = float(data_instance.value_np.max())

        min_val_idx = int(data_instance.value_np.argmin())
        max_val_idx = int(data_instance.value_np.argmax())
        min_ts = float(data_instance.timestamp_np[min_val_idx])
        max_ts = float(data_instance.timestamp_np[max_val_idx])

        first_ts = float(data_instance.timestamp_np[0])
        last_ts = float(data_instance.timestamp_np[-1])

        first_ts = convert_time(first_ts, source_time_unit, target_time_unit)
        last_ts = convert_time(last_ts, source_time_unit, target_time_unit)
        min_ts = convert_time(min_ts, source_time_unit, target_time_unit)
        max_ts = convert_time(max_ts, source_time_unit, target_time_unit)

        avg_val = average_over_time_range(
            data_instance,
            source_time_unit=source_time_unit,
            target_time_unit=target_time_unit,
        )

        print(f"Count:      {len(data_instance)}")
        print(f"Time range: {first_ts:.4f} to {last_ts:.4f} ({target_time_unit.value})")
        print(f"Min:        {min_val:.4f} at {min_ts:.4f} ({target_time_unit.value})")
        print(f"Max:        {max_val:.4f} at {max_ts:.4f} ({target_time_unit.value})")
        print(f"Average:    {avg_val:.4f}")
    else:
        print("Empty DataInstance.")


def single_run_summary(
    data: SingleRunData,
    time_unit: Timescale = Timescale.S,
) -> None:
    """
    Print overall information about the SingleRunData.

    Parameters
    ----------
    data : SingleRunData
        Data structure containing CSV file data
    time_unit : Timescale, optional
        Time unit used by the summary. Default is Timescale.S.
    """
    start_time = float(data.data_start_time)
    end_time = float(data.data_end_time)
    start_time = convert_time(start_time, data.timestamp_unit, time_unit)
    end_time = convert_time(end_time, data.timestamp_unit, time_unit)

    print(title_block("Data Summary"))
    print(f"Logging unit:       {data.timestamp_unit.value}")
    print(f"Time range:         {start_time} to {end_time} ({time_unit.value})")
    print(f"Total Variable:     {len(data.id_to_instance)}")
    print(f"Total Data Points:  {data.total_data_points}")
    print(DELIMITER)
