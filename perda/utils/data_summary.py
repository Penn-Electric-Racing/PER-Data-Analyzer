from ..analyzer.data_instance import DataInstance
from ..analyzer.single_run_data import SingleRunData
from .integrate import average_over_time_range
from .types import Timescale


def data_instance_summary(
    data_instance: DataInstance,
    time_unit: Timescale = Timescale.S,
) -> None:
    """
    Print information about a DataInstance.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance to print info about
    time_unit : Timescale, optional
        Unit for time values. Default is seconds
    """
    print(
        f"{data_instance.label} | ID: {data_instance.var_id} | C++ Name: {data_instance.cpp_name}"
    )
    print("-" * 40)

    if len(data_instance) > 0:
        min_val = float(data_instance.value_np.min())
        max_val = float(data_instance.value_np.max())

        min_val_idx = int(data_instance.value_np.argmin())
        max_val_idx = int(data_instance.value_np.argmax())
        min_ts = float(data_instance.timestamp_np[min_val_idx])
        max_ts = float(data_instance.timestamp_np[max_val_idx])

        first_ts = float(data_instance.timestamp_np[0])
        last_ts = float(data_instance.timestamp_np[-1])

        if time_unit == Timescale.S:
            first_ts = first_ts / 1e3
            last_ts = last_ts / 1e3
            min_ts = min_ts / 1e3
            max_ts = max_ts / 1e3

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
    if time_unit == Timescale.S:
        start_time = start_time / 1e3
        end_time = end_time / 1e3

    print("==== Data Summary ====")
    print(f"Time range:         {start_time} to {end_time} ({time_unit.value})")
    print(f"Total Variable:     {len(data.id_to_instance)}")
    print(f"Total Data Points:  {data.total_data_points}")
    print("======================")
