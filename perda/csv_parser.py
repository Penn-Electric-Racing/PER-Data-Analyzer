from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .analyzer.models import DataInstance, SingleRunData


class CSVParser:
    """
    Callable CSV parser that returns SingleRunData model.

    Parses data from CSV files into structured SingleRunData objects.
    """

    def __call__(
        self, file_path: str, parsing_errors_limit: int = 100
    ) -> SingleRunData:
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

        Raises
        ------
        Exception
            If too many parsing errors are encountered
        """
        # Maps variable ID to variable name
        var_name_map: dict[int, str] = {}
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
                var_id_name = line[6:].strip().split(": ")
                try:
                    var_id = int(var_id_name[1])
                    name = var_id_name[0]

                    # Store variable ID to name mapping
                    if var_id in var_name_map:
                        print(
                            f"Warning: Duplicate variable ID {var_id} at line {pbar.n}. Overwriting previous name."
                        )
                    var_name_map[var_id] = name
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
                    id = int(data[1])
                    timestamp = int(data[0])
                    val = float(data[2])

                    if not data_start_time:
                        data_start_time = timestamp

                    # Append timestamp and value to temporary lists
                    temp_time_value_list_map[id][0].append(timestamp)
                    temp_time_value_list_map[id][1].append(val)

                except Exception as e:
                    print(f"Error parsing data line {pbar.n}: {e}")
                    parsing_errors += 1

                    if (
                        parsing_errors_limit > 0
                        and parsing_errors >= parsing_errors_limit
                    ):
                        raise Exception("Too many data parsing errors encountered.")

                line = next(log, None)
            data_end_time = timestamp
            total_data_points = pbar.n

            pbar.close()

        # Format data as DataInstances
        data_instance_map: dict[int, DataInstance] = {}
        var_id_map: dict[str, int] = {}
        for var_id in tqdm(var_name_map, desc="Creating DataInstances"):
            name = var_name_map[var_id]
            var_id_map[name] = var_id
            timestamps_list, values_list = temp_time_value_list_map[var_id]
            data_instance_map[var_id] = DataInstance(
                timestamp_np=np.array(timestamps_list),
                value_np=np.array(values_list),
                label=name,
                var_id=var_id,
            )

        # Create and return SingleRunData model
        print(f"CSV parsing complete with {parsing_errors} parsing errors.")
        return SingleRunData(
            data_instance_map=data_instance_map,
            var_name_map=var_name_map,
            var_id_map=var_id_map,
            total_data_points=total_data_points,
            data_start_time=data_start_time,
            data_end_time=data_end_time,
        )
