from .single_run_data import DataInstance, SingleRunData


def trim_single_run_data(
    data: SingleRunData,
    ts_start: float,
    ts_end: float,
) -> SingleRunData:
    """Return a new SingleRunData with every variable trimmed to [ts_start, ts_end].

    Parameters
    ----------
    data : SingleRunData
    ts_start, ts_end : float
        Timestamps in the same unit as data.timestamp_unit.

    Returns
    -------
    SingleRunData
        Fresh object; the original is not mutated.

    Examples
    --------
    >>> from perda.core_data_structures import trim_single_run_data
    >>> from perda.units import from_seconds
    >>> start = aly.data.data_start_time
    >>> first_minute = trim_single_run_data(
    ...     aly.data, start, start + from_seconds(60.0, aly.data.timestamp_unit)
    ... )
    """
    trimmed: dict[int, DataInstance] = {
        var_id: di.trim(ts_start, ts_end) for var_id, di in data.id_to_instance.items()
    }

    return SingleRunData(
        id_to_instance=trimmed,
        cpp_name_to_id=dict(data.cpp_name_to_id),
        id_to_cpp_name=dict(data.id_to_cpp_name),
        id_to_descript=dict(data.id_to_descript),
        total_data_points=sum(len(di.value_np) for di in trimmed.values()),
        data_start_time=int(ts_start),
        data_end_time=int(ts_end),
        timestamp_unit=data.timestamp_unit,
        concat_boundaries=[],
    )


def split_single_run_data(
    data: SingleRunData,
    split_timestamps: list[float],
) -> list[SingleRunData]:
    """Split a SingleRunData into segments defined by a list of boundary timestamps.

    Each consecutive pair of timestamps in ``split_timestamps`` defines one
    segment.  The result is keyed by 1-based segment number.

    Parameters
    ----------
    data : SingleRunData
    split_timestamps : list[float]
        Ordered boundary timestamps in the same unit as data.timestamp_unit.
        Must contain at least 2 values.

    Returns
    -------
    List[SingleRunData]
        One SingleRunData per consecutive pair of boundaries.

    Examples
    --------
    >>> from perda.core_data_structures import split_single_run_data
    >>> data = aly.data
    >>> middle = (data.data_start_time + data.data_end_time) / 2
    >>> first_half, second_half = split_single_run_data(
    ...     data, [data.data_start_time, middle, data.data_end_time]
    ... )
    """
    if len(split_timestamps) < 2:
        raise ValueError(f"Need at least 2 boundary timestamps")

    segments = []
    for t_start, t_end in zip(split_timestamps[:-1], split_timestamps[1:]):
        segments.append(trim_single_run_data(data, t_start, t_end))

    return segments
