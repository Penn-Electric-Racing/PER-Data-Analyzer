from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .data_instance import DataInstance
from .single_run_data import SingleRunData
from ..utils.types import Timescale


def _resolve_parse_unit(
    header_line: str,
    parse_unit: Timescale | str | None,
) -> Timescale:
    if isinstance(parse_unit, str):
        parse_unit = parse_unit.strip().lower()
        if parse_unit not in (Timescale.MS.value, Timescale.US.value):
            raise ValueError(
                f"parse_unit must be 'ms' or 'us', got {parse_unit}"
            )
        parse_unit = Timescale(parse_unit)

    if parse_unit is not None and parse_unit not in (Timescale.MS, Timescale.US):
        raise ValueError(
            f"parse_unit must be Timescale.MS or Timescale.US, got {parse_unit}"
        )

    if parse_unit is not None:
        return parse_unit

    return Timescale.US if header_line.rstrip().endswith("v2.0") else Timescale.MS


def parse_csv(
    file_path: str,
    ts_offset: int = 0,
    parsing_errors_limit: int = 100,
    parse_unit: Timescale | str | None = None,
) -> SingleRunData:
    """
    Parse CSV file and return SingleRunData model.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to parse
    parsing_errors_limit : int, optional
        Maximum number of parsing errors before stopping. -1 for no limit. Default is 100
    parse_unit : Timescale | str | None, optional
        Logging timestamp unit. If None, auto-detects using header suffix "v2.0" (us) or defaults to ms.

    Returns
    -------
    SingleRunData
        Parsed data structure containing all variables
    """
    # Maps variable ID to variable name
    id_to_cpp_name: dict[int, str] = {}
    id_to_descript: dict[int, str] = {}

    # Temporary data structure with separate lists for timestamps and values
    temp_time_value_list_map: defaultdict[
        int, tuple[list[int] | np.ndarray, list[float] | np.ndarray]
    ] = defaultdict(lambda: ([], []))

    with open(file_path, "r") as log:
        # Parse and print first line (header)
        header_line = next(log, "")
        parse_unit = _resolve_parse_unit(header_line, parse_unit)
        print(f"Header: {header_line.rstrip()}")
        print(f"Timestamp unit: {parse_unit.value}")

        # Block 1: Variable ID/Name pairs
        pbar = tqdm(desc="Reading variable ID mappings", unit=" lines", initial=2)
        line = next(log, None)
        while line is not None and line.startswith("Value "):
            pbar.update(1)

            # Remove "Value " prefix, separate into variable name and ID
            identifier = line[6:].strip().split(": ")

            try:
                var_id = int(identifier[1])
                name_part = identifier[0]

                # Check format: Value Desc (cpp.name): id | Value cpp.name: id
                if "(" in name_part and ")" in name_part:
                    open_idx = name_part.rfind("(")
                    close_idx = name_part.rfind(")")
                    if open_idx < close_idx:
                        cpp_name = name_part[open_idx + 1 : close_idx].strip()
                        descript = name_part[:open_idx].strip()
                    else:
                        cpp_name = name_part.strip()
                        descript = ""
                else:
                    cpp_name = name_part.strip()
                    descript = ""
                if not cpp_name:
                    raise ValueError(f"Empty cpp_name in mapping line: {line.strip()}")

                # Store variable ID to name mapping
                if var_id in id_to_cpp_name:
                    print(
                        f"Warning: Duplicate variable ID {var_id} at line {pbar.n}. Overwriting previous name."
                    )
                id_to_cpp_name[var_id] = cpp_name
                id_to_descript[var_id] = descript

            except Exception as e:
                print(f"Error parsing variable ID/Name pair at line {pbar.n}: {e}")

            line = next(log, None)
        pbar.close()

        # Block 2: Data lines
        pbar = tqdm(desc="Reading data", unit=" lines", initial=0)
        data_start_time: int | None = None
        data_end_time: int = 0
        parsing_errors = 0
        while line is not None:
            pbar.update(1)
            data = line.strip().split(",")
            try:
                var_id = int(data[1])
                timestamp = int(data[0]) + ts_offset
                val = float(data[2])

                if data_start_time is None:
                    data_start_time = timestamp
                data_end_time = timestamp

                # Append timestamp and value to temporary lists
                temp_time_value_list_map[var_id][0].append(timestamp)
                temp_time_value_list_map[var_id][1].append(val)

            except Exception as e:
                print(f"Error parsing data line {pbar.n}: {e}")
                parsing_errors += 1

                if parsing_errors_limit > 0 and parsing_errors >= parsing_errors_limit:
                    raise Exception("Too many data parsing errors encountered.")

            line = next(log, None)
        if data_start_time is None:
            data_start_time = 0
        total_data_points = pbar.n

        pbar.close()

        # Block 3: Sort timestamps
        pbar = tqdm(
            temp_time_value_list_map.items(),
            desc="Sorting timestamps",
            unit=" vars",
            total=len(temp_time_value_list_map),
        )
        for var_id, (timestamps_list, values_list) in pbar:
            timestamps_np = np.asarray(timestamps_list, dtype=np.int64)
            values_np = np.asarray(values_list, dtype=np.float64)

            if timestamps_np.size >= 2:
                # Use stable sort
                order = np.argsort(timestamps_np, kind="stable")
                timestamps_np = timestamps_np[order]
                values_np = values_np[order]

            # Now temp_time_value_list_map should all have ndarray
            temp_time_value_list_map[var_id] = (timestamps_np, values_np)
        pbar.close()

    # Format data as DataInstances
    id_to_instance: dict[int, DataInstance] = {}
    cpp_name_to_id: dict[str, int] = {}
    for var_id in tqdm(id_to_cpp_name, desc="Creating DataInstances"):
        name = id_to_cpp_name[var_id]
        descript = id_to_descript[var_id]
        cpp_name_to_id[name] = var_id
        timestamps_list, values_list = temp_time_value_list_map[var_id]
        id_to_instance[var_id] = DataInstance(
            timestamp_np=np.asarray(timestamps_list),
            value_np=np.asarray(values_list),
            label=descript,
            var_id=var_id,
            cpp_name=name,
        )

    # Create and return SingleRunData model
    print(f"CSV parsing complete with {parsing_errors} parsing errors.")
    return SingleRunData(
        id_to_instance=id_to_instance,
        cpp_name_to_id=cpp_name_to_id,
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        total_data_points=total_data_points,
        data_start_time=data_start_time,
        data_end_time=data_end_time,
        timestamp_unit=parse_unit,
    )
