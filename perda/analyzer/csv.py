from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .data_instance import DataInstance
from .single_run_data import SingleRunData


def parse_csv(file_path: str, parsing_errors_limit: int = 100) -> SingleRunData:
    """
    Parse CSV file and return SingleRunData model.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to parse
    parsing_errors_limit : int, optional
        Maximum number of parsing errors before stopping. -1 for no limit. Default is 100

    Returns
    -------
    SingleRunData
        Parsed data structure containing all variables
    """
    # Maps variable ID to variable name
    id_to_cpp_name: dict[int, str] = {}
    id_to_descript: dict[int, str] = {}

    # Temporary data structure with separate lists for timestamps and values
    temp_time_value_list_map: defaultdict[int, tuple[list, list]] = defaultdict(
        lambda: ([], [])
    )

    with open(file_path, "r") as log:
        # Skip and print first line (header)
        print(f"Header: {next(log)}")

        # Block 1: Variable ID/Name pairs
        pbar = tqdm(desc="Reading variable ID mappings", unit=" lines", initial=2)
        line = next(log, None)
        while line is not None and line.startswith("Value "):
            pbar.update(1)

            # Remove "Value " prefix, separate into variable name and ID
            identifier = line[6:].strip().split(": ")

            try:
                var_id = int(identifier[1])

                name_list = identifier[0].split()
                cpp_name = name_list[-1][1:-1]  # drop surrounding brackets
                descript = " ".join(name_list[:-1])

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
        data_start_time = 0
        parsing_errors = 0
        while line is not None:
            pbar.update(1)
            data = line.strip().split(",")
            try:
                var_id = int(data[1])
                timestamp = int(data[0])
                val = float(data[2])

                if not data_start_time:
                    data_start_time = timestamp

                # Append timestamp and value to temporary lists
                temp_time_value_list_map[var_id][0].append(timestamp)
                temp_time_value_list_map[var_id][1].append(val)

            except Exception as e:
                print(f"Error parsing data line {pbar.n}: {e}")
                parsing_errors += 1

                if parsing_errors_limit > 0 and parsing_errors >= parsing_errors_limit:
                    raise Exception("Too many data parsing errors encountered.")

            line = next(log, None)
        data_end_time = timestamp
        total_data_points = pbar.n

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
            timestamp_np=np.array(timestamps_list),
            value_np=np.array(values_list),
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
    )
