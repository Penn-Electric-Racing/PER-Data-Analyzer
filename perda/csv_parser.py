import numpy as np
from tqdm import tqdm

from .data_instance import DataInstance


class CSVParser:
    def __init__(self):
        self.__tv_map = {}
        self.__ID_map = {}
        self.__name_map = {}
        self.__total_data_points = 0
        self.__data_start_time = None
        self.__data_end_time = None
        self.__file_read = False

    def read_csv(self, file_path, bad_data_limit=100):
        """
        Main function to parse csv file from path
        file_path: path of file we want to parse
        bad_data_limit: number of bad data before stopping (-1 for no limit)
        """
        # Reset previous data
        if self.__file_read:
            self.__tv_map = {}
            self.__ID_map = {}
            self.__name_map = {}
            self.__total_data_points = 0
            self.__data_start_time = None
            self.__data_end_time = None
            print("Resetting previous data.")
        self.__file_read = True

        # Initialize bad data count
        bad_data = 0

        with open(file_path, "r") as log:
            # Skip and print first line (header)
            header = next(log)
            print(f"Header: {header}")
            # Start at second line for displaying error line
            line_num = 1
            for line in tqdm(log, desc="Reading CSV", unit=" lines", initial=1):
                line_num += 1
                # Stop read if too many bad lines
                if bad_data_limit > 0 and bad_data >= bad_data_limit:
                    raise Exception("Too many bad data lines encountered.")
                # Read canid and name before data
                if line.startswith("Value "):
                    canid_name = line[6:].strip().split(": ")
                    try:
                        canID = int(canid_name[1])
                        name = canid_name[0]
                        # Store everything in map
                        if canID in self.__ID_map:
                            print(
                                f"Warning: Duplicate CAN ID {canID} at line {line_num}. Overwriting previous name."
                            )
                        self.__ID_map[canID] = name
                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        bad_data += 1
                        continue
                # Read data lines
                else:
                    data = line.strip().split(",")
                    try:
                        id = int(data[1])
                        timestamp = int(data[0])
                        val = float(data[2])

                        # Record start time
                        if self.__data_start_time is None:
                            self.__data_start_time = timestamp

                        if id not in self.__tv_map:
                            # Use can ID as key since there is somehow duplicate names sometimes :(
                            self.__tv_map[id] = []
                        self.__tv_map[id].append([timestamp, val])
                        self.__total_data_points += 1

                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        bad_data += 1
                        continue

        # Convert lists to DataInstance
        for canid in tqdm(self.__ID_map, desc="Creating DataInstances"):
            name = self.__ID_map[canid]
            self.__name_map[name] = canid
            if canid not in self.__tv_map:
                self.__tv_map[canid] = DataInstance(
                    np.array([]), np.array([]), label=name, canid=canid
                )
            else:
                data_array = self.__tv_map[canid]
                data_array = np.array(data_array)
                timestamps = data_array[:, 0]
                values = data_array[:, 1]
                self.__tv_map[canid] = DataInstance(
                    timestamps, values, label=name, canid=canid
                )

        # Record end time as last timestamp
        self.__data_end_time = timestamp
        if bad_data != 0:
            print(f"Csv parsing complete with: {bad_data} bad lines.")
        else:
            print("Csv parsing complete.")

    def get_data(self, input_canid_name):
        """
        Get DataInstance by canid or name.
        Raises AttributeError if no csv read or cannot find input.
        If input is DataInstance, return itself (used for plotting).
        Very very very very useful function :)) core of perda prob
        """
        # Dummy return for plotting function for convenience
        if isinstance(input_canid_name, DataInstance):
            return input_canid_name
        if not self.__file_read:
            raise AttributeError("No csv read. Call .read_csv() before getting data.")
        # If input is canid
        if isinstance(input_canid_name, int):
            if input_canid_name not in self.__tv_map:
                raise AttributeError("Aborted: Cannot Find Input ID")
            canid = input_canid_name
        # If input is can name
        elif isinstance(input_canid_name, str):
            canid = None
            for long_name in self.__name_map:
                if CSVParser.name_matches(input_canid_name, long_name):
                    canid = self.__name_map[long_name]
                    break
            if canid is None:
                raise AttributeError("Aborted: Cannot Find Input Name")
        else:
            raise ValueError("Input must be a string, int.")
        # Return DataInstance
        return self.__tv_map[canid]

    def name_matches(short_name, full_name):
        return f"({short_name})" in full_name or f"{short_name}" in full_name

    def print_info(self, input_canid_name=None, time_unit: str = "s"):
        """
        If input_canid_name is None, print parser info.
        If input_canid_name is DataInstance, print its info.
        If input_canid_name is canid or name, print its DataInstance info.
        """
        if input_canid_name is None:
            if not self.__file_read:
                raise AttributeError(
                    "No csv read. Call .read_csv() before printing info."
                )
            print("Parser Info:")
            start_time = self.__data_start_time
            end_time = self.__data_end_time
            if time_unit == "s":
                start_time = float(start_time) / 1e3
                end_time = float(end_time) / 1e3
            print(f"  Time range: {start_time} to {end_time} ({time_unit})")
            print(f"  Total CAN IDs:   {len(self.__tv_map)}")
            print(f"  Total CAN Names: {len(self.__name_map)}")
            print(f"  Total Data Points: {self.__total_data_points}")
        else:
            if isinstance(input_canid_name, DataInstance):
                input_canid_name.print_info(time_unit=time_unit)
            else:
                di = self.get_data(input_canid_name)
                di.print_info(time_unit=time_unit)

    def print_variables(
        self, search: str = None, strict_search: bool = False, sort_by="name"
    ):
        """
        Print all available CAN IDs and names.
        If search is given, only print ones containing the search string.
        """
        if not self.__file_read:
            raise AttributeError(
                "No csv read. Call .read_csv() before printing variables."
            )
        # Parse variable names into (inside, outside) pairs for sorting
        variable_pairs = []
        if search is not None:
            search_list = search.lower().split(" ")
        for canid in self.__ID_map:
            full_name = self.__ID_map[canid]
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
            sorted_pairs = np.array(
                sorted(variable_pairs, key=lambda t: (t[1], t[2], t[0])), dtype=object
            )
        else:
            # sort by canid only
            sorted_pairs = np.array(
                sorted(variable_pairs, key=lambda t: t[0]), dtype=object
            )
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
