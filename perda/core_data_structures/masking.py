from __future__ import annotations

from enum import Enum
from typing import Callable

from .data_instance import DataInstance


class FilterOptions(Enum):
    """Specifies which array(s) a filter function receives as input."""

    VALUES = "left_only"
    TIMESTAMPS = "right_only"
    BOTH = "both"


def apply_ufunc_filter(
    data: DataInstance,
    filter_func: Callable,
    apply_to: FilterOptions = FilterOptions.VALUES,
) -> DataInstance:
    """
    Apply a filter function to a DataInstance.

    Parameters
    ----------
    data : DataInstance
        Input DataInstance
    filter_func : Callable
        Function that takes in values and/or timestamps and returns a boolean mask
    apply_to : FilterOptions, optional
        Whether to apply the filter to values, timestamps, or both. Default is values

    Returns
    -------
    DataInstance
        Filtered DataInstance
    """
    if apply_to == FilterOptions.VALUES:
        mask = filter_func(data.value_np)
    elif apply_to == FilterOptions.TIMESTAMPS:
        mask = filter_func(data.timestamp_np)
    else:
        mask = filter_func(data.timestamp_np, data.value_np)

    return DataInstance(
        timestamp_np=data.timestamp_np[mask],
        value_np=data.value_np[mask],
        label=data.label,
        var_id=data.var_id,
        cpp_name=data.cpp_name,
    )
