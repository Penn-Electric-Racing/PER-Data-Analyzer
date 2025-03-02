import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class analyzer:
    def __init__(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__file_read = False
        self.__data_start_time = None
        self.__data_end_time = None

        self.__plot_same_graph = True
        self.__plot_start_time = 0
        self.__plot_end_time = -1
        self.__plot_unit = "s"

        print("Analyzer Created")

    def reset(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__file_read = False
        self.__data_start_time = None
        self.__data_end_time = None

        self.__plot_same_graph = True
        self.__plot_start_time = 0
        self.__plot_end_time = -1
        self.__plot_unit = "s"

        print("Reset Analyzer")

    def read_csv(self, path: str):
        if self.__file_read:
            print("Call .reset() before reading new csv")
            return
        # Reset but do not print
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None

        self.__plot_same_graph = True
        self.__plot_start_time = 0
        self.__plot_end_time = -1
        self.__plot_unit = "s"

        time_stamp = 0
        start_time_read = False
        high_voltage = False
        any_implausibility = False

        with open(path, 'r') as log:
            header = next(log)
            print(f"Reading file: {header}")
            line_num = 2
            with tqdm(desc="Processing CSV", unit="lines", initial = 1) as pbar:
                for line in log:
                    if (line.startswith("Value")):
                        canID_name_value = line[6:].strip().split(": ")
                        try:
                            self.__ID_map[int(canID_name_value[1])] = canID_name_value[0]
                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    else:
                        data = line.strip().split(",")
                        try:
                            id = int(data[1])
                            name = self.__ID_map[id]
                            val = float(data[2])
                            raw_time = int(data[0])

                            if not start_time_read and self.__name_matches("sdl.startTime", name):
                                self.__data_start_time = val
                                start_time_read = True
                            elif self.__name_matches("sdl.currentTime", name):
                                time_stamp = val / 1e3
                            elif self.__name_matches("ams.airsState", name):
                                if high_voltage and val == 0:
                                    high_voltage = False
                                    self.__high_voltage_changes.append((time_stamp, high_voltage))
                                elif not high_voltage and val == 4:
                                    high_voltage = True
                                    self.__high_voltage_changes.append((time_stamp, high_voltage))
                            elif self.__name_matches("pcm.pedals.implausibility.anyImplausibility", name):
                                any_implausibility = val

                            if name not in self.__value_map:
                                self.__value_map[name] = []
                            
                            self.__value_map[name].append([time_stamp, val, high_voltage, any_implausibility, raw_time])

                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    line_num += 1
                    pbar.update(1)
        self.__data_end_time = raw_time
        self.__file_read = True

    def set_plot(self, start_time = 0, end_time = -1, same_graph = True, unit = "s"):
        self.__plot_start_time = start_time
        self.__plot_end_time = end_time
        self.__plot_same_graph = same_graph
        if unit != "s" and unit != "ms":
            print("Invalid unit input, please use \"s\" or \"ms\". Default to s.")
            self.__plot_unit = "s"
        else:
            self.__plot_unit = unit

    def plot(self, variables: list):
        if not self.__file_read:
            print("No csv read. Call .read_csv() before plotting.")
            return
        for var in variables:
            if type(var) is list:
                self.plot(var, True)
                continue
            elif type(var) is str:
                short_name = var
                vals = self.get_np_array(short_name)
            elif isinstance(var, np.ndarray):
                if var.ndim != 2 or var.shape[1] < 2:
                    print("Invalid Input Array For Plotting")
                    continue
                vals = var
            else:
                print("Invalid Input For Plotting")
                continue

            last_time_stamp = self.__plot_end_time
            first_time_stamp = self.__plot_start_time

            if self.__plot_unit == "s":
                last_time_stamp *= 1e3
                first_time_stamp *= 1e3
            
            if self.__plot_end_time == -1:
                last_time_stamp = vals[-1,0]
            
            filtered_vals = vals[(vals[:, 0] >= first_time_stamp) & (vals[:, 0] <= last_time_stamp )]

            if self.__plot_unit == "s":
                t = filtered_vals[::, 0]/1e3
                plt.xlabel("Timestamp (s)")
            else:
                t = filtered_vals[::, 0]
                plt.xlabel("Timestamp (ms)")

            for hvT, hvOn in self.__high_voltage_changes:
                color = "red" if hvOn else "green"
                if (hvT >= first_time_stamp and hvT <= last_time_stamp):
                    if self.__plot_unit == "s":
                        hvT = hvT/1e3
                    plt.axvline(x=hvT, color=color)

            y = filtered_vals[::, 1]
            plt.plot(t, y, label=short_name)
            plt.legend()

            if not self.__plot_same_graph:
                plt.suptitle(short_name)
                plt.show()
                print("\n\n")
        
        if self.__plot_same_graph:
            plt.show()
            print("\n\n")

    def get_np_array(self, short_name: str):
        if not self.__file_read:
            print("No csv read. Call .read_csv() before finding variable.")
            return None
        full_name = None
        for var_name in self.__value_map.keys():
            if self.__name_matches(short_name, var_name):
                full_name = var_name
                break

        if full_name is None:
            print("Error: could not find data for " + short_name)
            return None

        return np.array(self.__value_map[full_name])
    
    def get_compute_arrays(self, op_list: list[str], match_type: str = "extend", start_time = 0, end_time = -1, unit = "s"):
        var_arrs = []
        operations = []
        max_start = start_time
        min_end = end_time
        if min_end == -1:
            min_end = self.__data_end_time
        if unit == "s":
            max_start = max_start * 1e3
            if min_end != -1:
                min_end = min_end * 1e3
        max_start = max(max_start, 0)
        min_end = min(min_end, self.__data_end_time)
        
        is_var = True
        for ops in op_list:
            if is_var:
                var_np = self.get_np_array(ops)
                if var_np is None:
                    print("Abroated: Missing Information")
                    return None
                max_start = max(var_np[0,0], max_start)
                min_end = min(var_np[-1,0], min_end)
                var_arrs.append(var_np)
            else:
                if ops != "+" and ops != "-" and ops != "*" and ops != "/":
                    print("Abroated: Invalid Operations Format")
                    return None
                operations.append(ops)
            is_var = not is_var
        filtered_arrs = []
        for np_arr in var_arrs:
            filtered_np = np_arr[(np_arr[:, 0] >= max_start) & (np_arr[:, 0] <= min_end)]
            filtered_arrs.append(filtered_np)
        
        sorted_arr = self.__align_nparr(filtered_arrs)

        
    def __align_nparr(self, np_list: list[np.ndarray], match_type = "connect"):
        num_arr = len(np_list)
        sorted_arr = self.__merge_and_sort_with_source(np_list)
        final_sorted_arr = []
        curr_idx = -1
        this_tsp = 0
        last_tsp = 0
        for arr in sorted_arr:
            this_tsp = arr[0]
            arr_input_num = arr[-1]
            if this_tsp != last_tsp:
                final_sorted_arr.append([None] * (num_arr+1))
                curr_idx += 1
                final_sorted_arr[curr_idx][0] = this_tsp
            final_sorted_arr[curr_idx][arr_input_num+1] = arr[1:-1]
            last_tsp = this_tsp
        
        for cnum in range(num_arr):
            for fin_arr in final_sorted_arr:
                if fin_arr[cnum+1] is None:
                    pass



        
        return final_sorted_arr
    
    def __merge_and_sort_with_source(self, arrays: list[np.ndarray]):
        """
        Merge a list of NumPy arrays (each with a timestamp in the first column) into one array.
        An extra column is added to mark the source (input order) of each row. The merged array is
        then sorted first by the timestamp and, for rows with identical timestamps, by the input sequence.
        
        :param arrays: List of NumPy arrays, each assumed to have a timestamp in column 0.
        :return: A single merged and sorted NumPy array with an extra column indicating the source.
        """
        new_arrays = []
        
        # Add an extra column to each array that indicates the input order (source index)
        for source_index, arr in enumerate(arrays):
            # Create a column with the source index repeated for each row in the array
            source_col = np.full((arr.shape[0], 1), source_index)
            # Append the source column to the right of the original array
            new_arr = np.hstack((arr, source_col))
            new_arrays.append(new_arr)
        
        # Merge all arrays vertically
        merged_array = np.vstack(new_arrays)
        
        # Sort the merged array using lexsort.
        # np.lexsort takes a tuple of keys, where the last key is the primary key.
        # Here, we want to sort primarily by the timestamp (column 0) and then by source index (last column).
        sorted_indices = np.lexsort((merged_array[:, -1], merged_array[:, 0]))
        sorted_array = merged_array[sorted_indices]
        
        return sorted_array

    def __fill_value(self, fst, fst_tsp, sec, sec_tsp, tsp):
        if sec_tsp == fst_tsp:
            raise ValueError("First and second timestamps must be different.")
        
        # Compute slope of the line
        slope = (sec - fst) / (sec_tsp - fst_tsp)
        
        # Compute the interpolated value at tsp
        return fst + slope * (tsp - fst_tsp)
        
    def nearest_effective(values):
        """
        Given a list of values (which may contain None), returns two lists:
        - before: the nearest effective (non-None) value from the left (including current)
        - after: the nearest effective (non-None) value from the right (including current)
        """
        n = len(values)
        before = [None] * n
        after = [None] * n

        # Forward pass to fill "before"
        last_effective = None
        for i, val in enumerate(values):
            if val is not None:
                last_effective = val
            before[i] = last_effective

        # Backward pass to fill "after"
        next_effective = None
        for i in range(n-1, -1, -1):
            if values[i] is not None:
                next_effective = values[i]
            after[i] = next_effective

        return before, after
    
    def __name_matches(self, short_name, full_name):
        return f'({short_name})' in full_name