"""Concatenate two SingleRunData objects sequentially."""

from typing import List

import numpy as np

from .data_instance import DataInstance
from .single_run_data import SingleRunData
from ..utils.types import Timescale


def _upscale_to_us(data: SingleRunData) -> SingleRunData:
    """
    Convert a millisecond-unit SingleRunData to microseconds in place.

    Parameters
    ----------
    data : SingleRunData
        Data with timestamp_unit == Timescale.MS

    Returns
    -------
    SingleRunData
        New SingleRunData with all timestamps scaled to microseconds
    """
    new_instances = {}
    for var_id, di in data.id_to_instance.items():
        new_instances[var_id] = DataInstance(
            timestamp_np=di.timestamp_np * 1000,
            value_np=di.value_np.copy(),
            label=di.label,
            var_id=di.var_id,
            cpp_name=di.cpp_name,
        )

    return SingleRunData(
        id_to_instance=new_instances,
        cpp_name_to_id=dict(data.cpp_name_to_id),
        id_to_cpp_name=dict(data.id_to_cpp_name),
        id_to_descript=dict(data.id_to_descript),
        total_data_points=data.total_data_points,
        data_start_time=data.data_start_time * 1000,
        data_end_time=data.data_end_time * 1000,
        timestamp_unit=Timescale.US,
        concat_boundaries=list(data.concat_boundaries),
    )


def concat_single_run_data(
    first: SingleRunData,
    second: SingleRunData,
    gap: int = 1,
) -> SingleRunData:
    """
    Concatenate two SingleRunData objects sequentially in time.

    The second run's timestamps are shifted so they start ``gap`` units after
    the first run ends.  Variables are matched by ``cpp_name``; unmatched
    variables appear in the result with data from only one run.

    If the two runs use different timestamp units (ms vs us), the ms run is
    upscaled to us so the result is in microseconds.

    Parameters
    ----------
    first : SingleRunData
        First run (earlier in time)
    second : SingleRunData
        Second run (appended after first)
    gap : int
        Gap in timestamp units inserted between the two runs. Default is 1.

    Returns
    -------
    SingleRunData
        New SingleRunData containing the concatenated data

    Examples
    --------
    >>> merged = concat_single_run_data(aly1.data, aly2.data)
    """
    # Reconcile timestamp units
    if first.timestamp_unit != second.timestamp_unit:
        if first.timestamp_unit == Timescale.MS:
            first = _upscale_to_us(first)
        else:
            second = _upscale_to_us(second)

    ts_unit = first.timestamp_unit
    shift = first.data_end_time - second.data_start_time + gap
    boundary_ts = second.data_start_time + shift

    # Build cpp_name -> DataInstance maps for both runs
    first_by_name: dict[str, DataInstance] = {}
    first_descript: dict[str, str] = {}
    for var_id, di in first.id_to_instance.items():
        name = first.id_to_cpp_name.get(var_id, di.cpp_name or "")
        first_by_name[name] = di
        first_descript[name] = first.id_to_descript.get(var_id, "")

    second_by_name: dict[str, DataInstance] = {}
    second_descript: dict[str, str] = {}
    for var_id, di in second.id_to_instance.items():
        name = second.id_to_cpp_name.get(var_id, di.cpp_name or "")
        second_by_name[name] = di
        second_descript[name] = second.id_to_descript.get(var_id, "")

    # Union of all cpp_names, preserving first-run order then second-run extras
    all_names: List[str] = list(first_by_name.keys())
    for name in second_by_name:
        if name not in first_by_name:
            all_names.append(name)

    # Build merged dictionaries with fresh sequential IDs
    id_to_instance: dict[int, DataInstance] = {}
    cpp_name_to_id: dict[str, int] = {}
    id_to_cpp_name: dict[int, str] = {}
    id_to_descript: dict[int, str] = {}
    total_points = 0

    for new_id, name in enumerate(all_names):
        di_first = first_by_name.get(name)
        di_second = second_by_name.get(name)

        if di_first is not None and di_second is not None:
            shifted_ts = di_second.timestamp_np + shift
            merged_ts = np.concatenate([di_first.timestamp_np, shifted_ts])
            merged_vals = np.concatenate([di_first.value_np, di_second.value_np])
            label = di_first.label
            descript = first_descript[name]
        elif di_first is not None:
            merged_ts = di_first.timestamp_np
            merged_vals = di_first.value_np
            label = di_first.label
            descript = first_descript[name]
        else:
            assert di_second is not None
            merged_ts = di_second.timestamp_np + shift
            merged_vals = di_second.value_np
            label = di_second.label
            descript = second_descript[name]

        id_to_instance[new_id] = DataInstance(
            timestamp_np=merged_ts,
            value_np=merged_vals,
            label=label,
            var_id=new_id,
            cpp_name=name,
        )
        cpp_name_to_id[name] = new_id
        id_to_cpp_name[new_id] = name
        id_to_descript[new_id] = descript
        total_points += len(merged_ts)

    # Carry forward existing boundaries and add the new one
    concat_boundaries = [
        *first.concat_boundaries,
        boundary_ts,
        *[b + shift for b in second.concat_boundaries],
    ]

    return SingleRunData(
        id_to_instance=id_to_instance,
        cpp_name_to_id=cpp_name_to_id,
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        total_data_points=total_points,
        data_start_time=first.data_start_time,
        data_end_time=second.data_end_time + shift,
        timestamp_unit=ts_unit,
        concat_boundaries=concat_boundaries,
    )
