import numpy as np
from tqdm import tqdm

from .datainstance import DataInstance


class newparser:
    def __init__(self):
        self.__tv_map = {}
        self.__ID_map = {}
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
                        name = self.__ID_map[id]
                        timestamp = int(data[0])
                        val = float(data[2])

                        # Record start time
                        if self.__data_start_time is None:
                            self.__data_start_time = timestamp

                        if name not in self.__tv_map:
                            # self.__tv_map[name] = DataInstance()
                            self.__tv_map[name] = []
                        self.__tv_map[name].append([timestamp, val])

                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        bad_data += 1
                        continue

        # Convert lists to DataInstance
        for name in tqdm(self.__tv_map, desc="Creating DataInstances"):
            data_array = np.array(self.__tv_map[name])
            timestamps = data_array[:, 0]
            values = data_array[:, 1]
            self.__tv_map[name] = DataInstance(timestamps, values, label=name)

        # temp data check, in idmap not in tvmap
        for id in self.__ID_map:
            name = self.__ID_map[id]
            if name not in self.__tv_map:
                print(f"Warning: ID {id} with name '{name}' has no data.")

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
        If input is already DataInstance, return it directly.
        Very very very very useful function :)) core of perda prob
        """
        if isinstance(input_canid_name, DataInstance):
            return input_canid_name
        if not self.__file_read:
            raise AttributeError("No csv read. Call .read_csv() before getting data.")
        # If input is canid
        if isinstance(input_canid_name, int):
            if input_canid_name not in self.__ID_map:
                raise AttributeError("Aborted: Cannot Find Input ID")
            name = self.__ID_map[input_canid_name]
        # If input is can name
        elif isinstance(input_canid_name, str):
            name = None
            for var_name in self.__tv_map.keys():
                if newparser.name_matches(input_canid_name, var_name):
                    name = var_name
                    break
            if name is None:
                raise AttributeError("Aborted: Cannot Find Input Name")
        else:
            raise ValueError("Input must be a string, int, or DataInstance.")
        # Return DataInstance
        return self.__tv_map[name]

    def name_matches(short_name, full_name):
        return f"({short_name})" in full_name or f"{short_name}" in full_name
