from ..analyzer.data_instance import DataInstance
from ..analyzer.single_run_data import SingleRunData
from ..constants import DELIMITER, title_block
from .integrate import average_over_time_range
from .types import Timescale


def _to_seconds(timestamp: float, source_time_unit: Timescale) -> float:
    if source_time_unit == Timescale.US:
        return timestamp / 1e6
    if source_time_unit == Timescale.MS:
        return timestamp / 1e3
    return timestamp


def _from_seconds(timestamp_s: float, target_time_unit: Timescale) -> float:
    if target_time_unit == Timescale.US:
        return timestamp_s * 1e6
    if target_time_unit == Timescale.MS:
        return timestamp_s * 1e3
    return timestamp_s


def _convert_time(
    timestamp: float,
    source_time_unit: Timescale,
    target_time_unit: Timescale,
) -> float:
    return _from_seconds(_to_seconds(timestamp, source_time_unit), target_time_unit)


def data_instance_summary(
    data_instance: DataInstance,
    time_unit: Timescale = Timescale.S,
    source_time_unit: Timescale = Timescale.MS,
) -> None:
    """
    Print information about a DataInstance.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance to print info about
    time_unit : Timescale, optional
        Unit for time values. Default is seconds
    source_time_unit : Timescale, optional
        Timestamp unit used in `data_instance.timestamp_np`. Default is milliseconds.
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

        first_ts = _convert_time(first_ts, source_time_unit, time_unit)
        last_ts = _convert_time(last_ts, source_time_unit, time_unit)
        min_ts = _convert_time(min_ts, source_time_unit, time_unit)
        max_ts = _convert_time(max_ts, source_time_unit, time_unit)

        avg_val = average_over_time_range(data_instance, time_unit=time_unit)

        print(f"Count:      {len(data_instance)}")
        print(f"Time range: {first_ts:.4f} to {last_ts:.4f} ({time_unit.value})")
        print(f"Min:        {min_val:.4f} at {min_ts:.4f} ({time_unit.value})")
        print(f"Max:        {max_val:.4f} at {max_ts:.4f} ({time_unit.value})")
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
        Unit for time values. Default is seconds
    """
    start_time = float(data.data_start_time)
    end_time = float(data.data_end_time)
    start_time = _convert_time(start_time, data.timestamp_unit, time_unit)
    end_time = _convert_time(end_time, data.timestamp_unit, time_unit)

    print(title_block("Data Summary"))
    print(f"Logging unit:       {data.timestamp_unit.value}")
    print(f"Time range:         {start_time} to {end_time} ({time_unit.value})")
    print(f"Total Variable:     {len(data.id_to_instance)}")
    print(f"Total Data Points:  {data.total_data_points}")
    print(DELIMITER)
