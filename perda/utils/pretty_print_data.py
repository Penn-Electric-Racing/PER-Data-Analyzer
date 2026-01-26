from typing import Optional

import numpy as np

from ..analyzer.models import DataInstance
from ..csv_parser import SingleRunData
from .utils import average_over_time_range, integrate_over_time_range


def pretty_print_data_instance_info(
    data_instance: DataInstance, time_unit: str = "s"
) -> None:
    """
    Print information about a DataInstance.

    Parameters
    ----------
    data_instance : DataInstance
        The DataInstance to print info about
    time_unit : str, optional
        Unit for time values: "ms" or "s". Default is "s"

    Returns
    -------
    None
    """
    print(f"DataInstance for | {data_instance.label} | canid={data_instance.canid}")
    if len(data_instance) > 0:
        min_val = float(data_instance.value_np.min())
        max_val = float(data_instance.value_np.max())
        min_val_idx = int(data_instance.value_np.argmin())
        max_val_idx = int(data_instance.value_np.argmax())
        min_ts = float(data_instance.timestamp_np[min_val_idx])
        max_ts = float(data_instance.timestamp_np[max_val_idx])
        first_ts = float(data_instance.timestamp_np[0])
        last_ts = float(data_instance.timestamp_np[-1])
        if time_unit == "s":
            first_ts = first_ts / 1e3
            last_ts = last_ts / 1e3
            min_ts = min_ts / 1e3
            max_ts = max_ts / 1e3
        avg_val = average_over_time_range(data_instance)
        integral = integrate_over_time_range(data_instance, time_unit=time_unit)
        # Set width for alignment
        w = 10
        print(f"  Data points:  {len(data_instance):>{w}}")
        print(f"  Time range:   {first_ts:>{w}.4f} to {last_ts:>{w}.4f} ({time_unit})")
        print(f"  Min value:    {min_val:>{w}.4f} at {min_ts:>{w}.4f} ({time_unit})")
        print(f"  Max value:    {max_val:>{w}.4f} at {max_ts:>{w}.4f} ({time_unit})")
        print(f"  Integral:     {integral:>{w}.4f}")
        print(f"  Average value:{avg_val:>{w}.4f}")
    else:
        print("  Empty DataInstance.")


def pretty_print_single_run_info(
    data: SingleRunData,
    time_unit: str = "s",
) -> None:
    """
    Print overall information about the SingleRunData.

    Parameters
    ----------
    data : SingleRunData
        Data structure containing CSV file data
    time_unit : str, optional
        Unit for time values: "ms" or "s". Default is "s"

    Returns
    -------
    None
    """
    print("Parser Info:")
    start_time = float(data.data_start_time)
    end_time = float(data.data_end_time)
    if time_unit == "s":
        start_time = start_time / 1e3
        end_time = end_time / 1e3
    print(f"  Time range: {start_time} to {end_time} ({time_unit})")
    print(f"  Total CAN IDs:   {len(data.data_instance_map)}")
    print(f"  Total CAN Names: {len(data.can_id_map)}")
    print(f"  Total Data Points: {data.total_data_points}")


def pretty_print_single_run_variables(
    data: SingleRunData,
    search: Optional[str] = None,
    strict_search: bool = False,
    sort_by: str = "name",
) -> None:
    """
    Print a list of all available variables in the dataset.

    Parameters
    ----------
    data : SingleRunData
        Data structure containing CSV file data
    search : Optional[str], optional
        Search term to filter variables. Multiple terms separated by spaces. Default is None
    strict_search : bool, optional
        If True, all search terms must be present.
        If False, any search term present is enough. Default is False
    sort_by : str, optional
        How to sort the variables list:
        - "name": Sort alphabetically by variable name
        - "canid": Sort by CAN ID
        Default is "name"
    """
    # Parse variable names into (inside, outside) pairs for sorting
    variable_pairs = []
    if search is not None:
        search_list = search.lower().split(" ")
    for canid in data.var_name_map:
        full_name = data.var_name_map[canid]
        # Strict search: all terms must be present
        if strict_search and search is not None:
            if not all(term in full_name.lower() for term in search_list):
                continue
        # Non-strict search: any term present is enough
        if not strict_search and search is not None:
            if not any(term in full_name.lower() for term in search_list):
                continue
        s = full_name.strip()
        left = s.rfind("(")
        # well-formed "( … )" at the end?
        if left != -1 and s.endswith(")"):
            description = s[:left].rstrip()
            short_can_name = s[left + 1 : -1].strip()  # drop '(' and ')'
            if short_can_name:  # both parts exist
                # pass search filter, append
                variable_pairs.append((canid, short_can_name, description))
                continue
        # fallback: single column
        variable_pairs.append((canid, s, ""))

    # Sort by name or ID
    if sort_by == "name":
        # sort by outside first, then inside, then canid
        sorted_pairs = sorted(variable_pairs, key=lambda t: (t[1], t[2], t[0]))
    else:
        # sort by canid only
        sorted_pairs = sorted(variable_pairs, key=lambda t: t[0])

    # First print total count, indicating search methods and outputs
    if search is None:
        print(f"Total CAN Variables: {len(sorted_pairs)}")
    elif strict_search:
        print(f"Total CAN Variables matching ALL terms '{search}': {len(sorted_pairs)}")
    else:
        print(f"Total CAN Variables matching ANY terms '{search}': {len(sorted_pairs)}")

    # if sorted_pairs is empty, print no matching found
    if len(sorted_pairs) == 0:
        print("No matching CAN variables found.")
        return

    # Print in Three columns
    col1_width = max(len(str(id)) for id, _, _ in sorted_pairs)
    col2_width = max(len(name) for _, name, _ in sorted_pairs)
    if sort_by == "name":
        print(f"{'CAN Name':<{col2_width}}  {'CAN ID':<{col1_width}}  Description")
        print("-" * (col1_width + col2_width + 15))
        for canid, short_name, description in sorted_pairs:
            print(f"{short_name:<{col2_width}}  {canid:<{col1_width}}  {description}")
    else:
        print(f"{'CAN ID':<{col1_width}}  {'CAN Name':<{col2_width}}  Description")
        print("-" * (col1_width + col2_width + 15))
        for canid, short_name, description in sorted_pairs:
            print(f"{canid:<{col1_width}}  {short_name:<{col2_width}}  {description}")
    # end with line
    print("-" * (col1_width + col2_width + 15))
