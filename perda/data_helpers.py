from typing import Optional, Union
from .csv_parser import SingleRunData
from .data_instance import DataInstance


def print_info(data: SingleRunData, input_canid_name: Optional[Union[str, int, DataInstance]] = None, time_unit: str = "s"):
    """
    Print information about the data, a specific variable, or a CAN ID.

    data: SingleRunData instance
    input_canid_name (Union[None, str, int], optional): What to get info about:
        - None: Print overall dataset information
        - str: Print information about a specific variable name
        - int: Print information about a specific CAN ID
        Defaults to None.
    time_unit (str, optional): Unit for time values ("ms" or "s").
        Defaults to "s".
    """
    if input_canid_name is None:
        print("Parser Info:")
        start_time = data.data_start_time
        end_time = data.data_end_time
        if time_unit == "s":
            start_time = float(start_time) / 1e3
            end_time = float(end_time) / 1e3
        print(f"  Time range: {start_time} to {end_time} ({time_unit})")
        print(f"  Total CAN IDs:   {len(data.tv_map)}")
        print(f"  Total CAN Names: {len(data.name_map)}")
        print(f"  Total Data Points: {data.total_data_points}")
    else:
        if isinstance(input_canid_name, DataInstance):
            input_canid_name.print_info(time_unit=time_unit)
        else:
            di = data[input_canid_name]
            di.print_info(time_unit=time_unit)


def print_variables(data: SingleRunData, search: str = None, strict_search: bool = False, sort_by: str = "name"):
    """
    Print a list of all available variables in the dataset.

    data: SingleRunData instance
    search (str, optional): Search term to filter variables
    strict_search (bool, optional): If True, all search terms must be present.
        If False, any search term present is enough. Defaults to False.
    sort_by (str, optional): How to sort the variables list:
        - "name": Sort alphabetically by variable name
        - "canid": Sort by CAN ID
        Defaults to "name".
    """
    # Parse variable names into (inside, outside) pairs for sorting
    variable_pairs = []
    if search is not None:
        search_list = search.lower().split(" ")
    for canid in data.id_map:
        full_name = data.id_map[canid]
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
        print(
            f"Total CAN Variables matching ALL terms '{search}': {len(sorted_pairs)}"
        )
    else:
        print(
            f"Total CAN Variables matching ANY terms '{search}': {len(sorted_pairs)}"
        )

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
            print(
                f"{short_name:<{col2_width}}  {canid:<{col1_width}}  {description}"
            )
    else:
        print(f"{'CAN ID':<{col1_width}}  {'CAN Name':<{col2_width}}  Description")
        print("-" * (col1_width + col2_width + 15))
        for canid, short_name, description in sorted_pairs:
            print(
                f"{canid:<{col1_width}}  {short_name:<{col2_width}}  {description}"
            )
    # end with line
    print("-" * (col1_width + col2_width + 15))