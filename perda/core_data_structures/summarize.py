import numpy as np
from numpy.typing import NDArray

from ..constants import SECTION_DIVIDER, title_block
from ..units import Timescale, convert_time


def summarize_time_series(
    header: str, timestamp_np: NDArray[np.int64], value_np: NDArray[np.float64]
) -> str:
    """
    Format count, time range, min, max and time-weighted average of a time series.

    Parameters
    ----------
    header : str
        First line of the summary, e.g. the variable's label and name.
    timestamp_np : NDArray[np.int64]
        Raw timestamps, in the log's native unit.
    value_np : NDArray[np.float64]
        Values, same length as ``timestamp_np``.

    Returns
    -------
    str
        Multi-line summary block. Timestamps are reported in raw units.
    """
    if len(timestamp_np) == 0:
        return f"{header}\n{SECTION_DIVIDER}\nEmpty DataInstance."

    min_index = int(value_np.argmin())
    max_index = int(value_np.argmax())
    duration = timestamp_np[-1] - timestamp_np[0]
    average = (
        float(np.trapezoid(value_np, timestamp_np) / duration)
        if duration > 0
        else float(np.mean(value_np))
    )
    lines = [
        f"Count:      {len(timestamp_np)}",
        f"Time range: {timestamp_np[0]} to {timestamp_np[-1]} (raw)",
        f"Min:        {value_np[min_index]:.4f} at {timestamp_np[min_index]} (raw)",
        f"Max:        {value_np[max_index]:.4f} at {timestamp_np[max_index]} (raw)",
        f"Average:    {average:.4f}",
    ]
    return "\n".join([header, SECTION_DIVIDER, *lines])


def summarize_run(
    timestamp_unit: Timescale,
    data_start_time: int,
    data_end_time: int,
    variable_count: int,
    total_data_points: int,
) -> str:
    """
    Format the overall information of a run.

    Parameters
    ----------
    timestamp_unit : Timescale
        Native timestamp unit of the run.
    data_start_time : int
        First timestamp of the run, in ``timestamp_unit``.
    data_end_time : int
        Last timestamp of the run, in ``timestamp_unit``.
    variable_count : int
        Number of variables in the run.
    total_data_points : int
        Number of data points across all variables.

    Returns
    -------
    str
        Multi-line summary block with the time range in seconds.
    """
    start_s = convert_time(float(data_start_time), timestamp_unit, Timescale.S)
    end_s = convert_time(float(data_end_time), timestamp_unit, Timescale.S)
    lines = [
        title_block("Data Summary"),
        f"Logging unit:       {timestamp_unit.value}",
        f"Time range:         {start_s} to {end_s} (s)",
        f"Total Variable:     {variable_count}",
        f"Total Data Points:  {total_data_points}",
        SECTION_DIVIDER,
    ]
    return "\n".join(lines)
