from pydantic import BaseModel, Field

from ..constants import DELIMITER, title_block
from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale, convert_time
from .integrate import average_over_time_range


class DataInstanceSummary(BaseModel):
    """Summary statistics for a single DataInstance."""

    header: str = Field(
        description="Label / ID / C++ name line of the source DataInstance"
    )
    count: int = Field(description="Number of data points")
    time_unit: Timescale = Field(
        description="Time unit that all timestamps below are expressed in"
    )
    first_timestamp: float = Field(
        default=0.0, description="Timestamp of the first data point"
    )
    last_timestamp: float = Field(
        default=0.0, description="Timestamp of the last data point"
    )
    min_value: float = Field(default=0.0, description="Smallest value")
    min_timestamp: float = Field(
        default=0.0, description="Timestamp at which the minimum occurs"
    )
    max_value: float = Field(default=0.0, description="Largest value")
    max_timestamp: float = Field(
        default=0.0, description="Timestamp at which the maximum occurs"
    )
    average: float = Field(
        default=0.0, description="Time-weighted average over the full range"
    )

    def __str__(self) -> str:
        """
        Human-readable multi-line summary block.

        Returns
        -------
        str
            The DataInstance header, followed by a delimiter and the statistics.
        """
        unit = self.time_unit.value
        body = (
            "Empty DataInstance."
            if self.count == 0
            else "\n".join(
                [
                    f"Count:      {self.count}",
                    f"Time range: {self.first_timestamp:.4f} to {self.last_timestamp:.4f} ({unit})",
                    f"Min:        {self.min_value:.4f} at {self.min_timestamp:.4f} ({unit})",
                    f"Max:        {self.max_value:.4f} at {self.max_timestamp:.4f} ({unit})",
                    f"Average:    {self.average:.4f}",
                ]
            )
        )
        return f"{self.header}\n{DELIMITER}\n{body}"


def data_instance_summary(
    data_instance: DataInstance,
    source_time_unit: Timescale = Timescale.MS,
    target_time_unit: Timescale = Timescale.S,
) -> DataInstanceSummary:
    """
    Summarize a DataInstance.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance to summarize
    source_time_unit : Timescale, optional
        Time unit of data instance. Default is milliseconds.
    target_time_unit : Timescale, optional
        Time unit used by the summary. Default is Timescale.S.

    Returns
    -------
    DataInstanceSummary
        Statistics for the DataInstance. Printing is left to the caller.

    Examples
    --------
    >>> print(data_instance_summary(aly.data["bms.pack.current"]))
    >>> data_instance_summary(aly.data["bms.pack.current"]).max_value
    """
    header = str(data_instance)

    if len(data_instance) == 0:
        return DataInstanceSummary(header=header, count=0, time_unit=target_time_unit)

    min_val_idx = int(data_instance.value_np.argmin())
    max_val_idx = int(data_instance.value_np.argmax())

    def to_target(timestamp: float) -> float:
        return convert_time(float(timestamp), source_time_unit, target_time_unit)

    return DataInstanceSummary(
        header=header,
        count=len(data_instance),
        time_unit=target_time_unit,
        first_timestamp=to_target(data_instance.timestamp_np[0]),
        last_timestamp=to_target(data_instance.timestamp_np[-1]),
        min_value=float(data_instance.value_np.min()),
        min_timestamp=to_target(data_instance.timestamp_np[min_val_idx]),
        max_value=float(data_instance.value_np.max()),
        max_timestamp=to_target(data_instance.timestamp_np[max_val_idx]),
        average=average_over_time_range(
            data_instance,
            source_time_unit=source_time_unit,
            target_time_unit=target_time_unit,
        ),
    )


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
