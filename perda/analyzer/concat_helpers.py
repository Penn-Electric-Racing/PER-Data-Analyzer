from typing import List

import numpy as np

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.search_indexes import VariableSemanticIndex
from ..core_data_structures.semantic_embedding import (
    copy_semantic_vectors,
    empty_semantic_index,
    update_semantic_index,
)
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale
from .analyzer import Analyzer


def _upscale_to_us(data: SingleRunData) -> SingleRunData:
    """Convert a millisecond-unit SingleRunData to microseconds in place."""
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
        semantic_index=data.semantic_index,
    )


def _merge_semantic_indexes(
    first: SingleRunData,
    second: SingleRunData,
    first_id_to_new: dict[int, int],
    second_id_to_new: dict[int, int],
    merged_id_to_descript: dict[int, str],
) -> VariableSemanticIndex | None:
    """Combine both runs' semantic indexes under the merged variable IDs."""
    if first.semantic_index is None and second.semantic_index is None:
        return None

    merged = empty_semantic_index()
    for source, id_map in (
        (first.semantic_index, first_id_to_new),
        (second.semantic_index, second_id_to_new),
    ):
        if source is not None:
            copy_semantic_vectors(source, merged, id_map)
    update_semantic_index(merged, merged_id_to_descript)
    return merged


def concat_single_run_data(
    first: SingleRunData,
    second: SingleRunData,
    gap: int = 1,
) -> SingleRunData:
    """
    Concatenate two SingleRunData objects sequentially in time.

    The second run's timestamps are shifted so they start ``gap`` units after the first run ends.
    Variables are matched by ``cpp_name``; unmatched variables appear in the result with data from only one run.

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
    first_ids: dict[str, int] = {}
    for var_id, di in first.id_to_instance.items():
        name = first.id_to_cpp_name.get(var_id, di.cpp_name or "")
        first_by_name[name] = di
        first_descript[name] = first.id_to_descript.get(var_id, "")
        first_ids[name] = var_id

    second_by_name: dict[str, DataInstance] = {}
    second_descript: dict[str, str] = {}
    second_ids: dict[str, int] = {}
    for var_id, di in second.id_to_instance.items():
        name = second.id_to_cpp_name.get(var_id, di.cpp_name or "")
        second_by_name[name] = di
        second_descript[name] = second.id_to_descript.get(var_id, "")
        second_ids[name] = var_id

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
    first_id_to_new: dict[int, int] = {}
    second_id_to_new: dict[int, int] = {}
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
            first_id_to_new[first_ids[name]] = new_id
        elif di_first is not None:
            merged_ts = di_first.timestamp_np
            merged_vals = di_first.value_np
            label = di_first.label
            descript = first_descript[name]
            first_id_to_new[first_ids[name]] = new_id
        else:
            if di_second is None:
                raise KeyError(f"Variable '{name}' is missing from both runs")
            merged_ts = di_second.timestamp_np + shift
            merged_vals = di_second.value_np
            label = di_second.label
            descript = second_descript[name]
            second_id_to_new[second_ids[name]] = new_id

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
        semantic_index=_merge_semantic_indexes(
            first, second, first_id_to_new, second_id_to_new, id_to_descript
        ),
    )


def concat(
    first: Analyzer,
    second: Analyzer,
    gap: int = 1,
) -> Analyzer:
    """
    Concatenate two Analyzers sequentially in time by concatenating their SingleRunData objects.
    See :func:`concat_single_run_data` for details on how variables are matched and timestamps are shifted.

    Parameters
    ----------
    first : Analyzer
        First analyzer (earlier in time)
    second : Analyzer
        Second analyzer (appended after first)
    gap : int
        Gap in timestamp units between the two runs. Default is 1.

    Returns
    -------
    Analyzer

    Examples
    --------
    >>> from perda.analyzer import concat
    >>> merged = concat(aly1, aly2)
    >>> fig = merged.plot("bms.pack.voltage")
    """
    merged = object.__new__(Analyzer)
    merged.data = concat_single_run_data(first.data, second.data, gap=gap)
    return merged
