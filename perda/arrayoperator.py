import numpy as np

from .csvparser import csvparser
from . import helper

class arrayoperator:
    def __init__(self):
        self.__csvparser = csvparser()
        self.__file_read = False

    def reset(self):
        self.__csvparser = csvparser()
        self.__file_read = False

    def set_csvparser(self, cp: csvparser):
        self.__csvparser = cp
        self.__file_read = True
    
    def get_filtered_nparr(self, arr, start_time = 0, end_time = -1, unit = "s"):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before taking integral.")

        if type(arr) is str:
            short_name = arr
            vals = self.__csvparser.get_np_array(short_name)
            if vals is None:
                raise AttributeError("Aborted: Cannot Find Input Name")
        elif isinstance(arr, np.ndarray):
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise AttributeError("Invalid Input Array For Integral")
            short_name = "Input Array"
            vals = arr
        else:
            raise AttributeError("Invalid Input For Integral")

        max_start = float(start_time)
        min_end = float(end_time)
        if min_end == -1:
            min_end = self.__csvparser.get_data_end_time()
        if unit == "s":
            max_start = max_start * 1e3
            if min_end != -1:
                min_end = min_end * 1e3
        max_start = max(max_start, vals[0,0])
        min_end = min(vals[-1,0],min(min_end, self.__csvparser.get_data_end_time()))

        filtered_arr = vals[(vals[:, 0] >= max_start) & (vals[:, 0] <= min_end)]

        if len(filtered_arr) == 0:
            raise ValueError("No data within time stamp.")

        return filtered_arr
    
    def get_integral_avg(self, filtered_arr):

        time_diff = filtered_arr[-1,0] - filtered_arr[0,0]
        timestamps = filtered_arr[:, 0]
        values = filtered_arr[:, 1]

        integral = np.trapz(values, timestamps)
        avg = integral/time_diff

        return integral, avg
    
    def get_max_min(self, filtered_arr):
        timestamps = filtered_arr[:, 0]
        values = filtered_arr[:, 1]

        min_index = np.argmin(values)  # Index of min value
        max_index = np.argmax(values)  # Index of max value

        min_value = values[min_index]
        max_value = values[max_index]

        min_timestamp = timestamps[min_index]
        max_timestamp = timestamps[max_index]

        return max_value, max_timestamp, min_value, min_timestamp
    
    def align_arrays(self, arr_list: list, match_type: str = "connect", start_time = 0, end_time = -1, unit = "s"):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        var_arrs = []
        max_start = start_time
        min_end = end_time
        if min_end == -1:
            min_end = self.__csvparser.get_data_end_time()
        if unit == "s":
            max_start = max_start * 1e3
            if min_end != -1:
                min_end = min_end * 1e3
        max_start = max(max_start, 0)
        min_end = min(min_end, self.__csvparser.get_data_end_time())

        for arr in arr_list:
            if type(arr) is str:
                var_np = self.__csvparser.get_np_array(arr)
                if var_np is None:
                    raise AttributeError("Abroated: Missing Information: Cannot Find Input Name")
            elif isinstance(arr, np.ndarray):
                var_np = arr
            else:
                raise TypeError("Abroated: Invalid Input Type")
            max_start = max(var_np[0,0], max_start)
            min_end = min(var_np[-1,0], min_end)
            var_arrs.append(var_np)

        filtered_arrs = []
        for np_arr in var_arrs:
            filtered_np = np_arr[(np_arr[:, 0] >= max_start) & (np_arr[:, 0] <= min_end)]
            filtered_arrs.append(filtered_np)

        arr_num = len(filtered_arrs)
        sorted_arr, sorted_hvimp = helper.align_nparr(filtered_arrs)
        filled_data = []
        for i in range(arr_num):
            # Stack the timestamp column with the current value column
            arr = np.column_stack((sorted_arr[:, 0], sorted_arr[:, i+1]))
            filled_arr = helper.fill_missing_values(arr, match_type)
            combined_filled = np.hstack((filled_arr, sorted_hvimp))
            filled_data.append(combined_filled)
        
        return filled_data

    def get_compute_arrays(self, op_list: list[str], match_type: str = "connect", start_time = 0, end_time = -1, unit = "s"):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        var_arrs = []
        operations = []
        max_start = start_time
        min_end = end_time
        if min_end == -1:
            min_end = self.__csvparser.get_data_end_time()
        if unit == "s":
            max_start = max_start * 1e3
            if min_end != -1:
                min_end = min_end * 1e3
        max_start = max(max_start, 0)
        min_end = min(min_end, self.__csvparser.get_data_end_time())
        
        is_var = True
        for ops in op_list:
            if is_var:
                var_np = self.__csvparser.get_np_array(ops)
                if var_np is None:
                    raise AttributeError("Abroated: Missing Information: Cannot Find Input Name")
                max_start = max(var_np[0,0], max_start)
                min_end = min(var_np[-1,0], min_end)
                var_arrs.append(var_np)
            else:
                if ops != "+" and ops != "-" and ops != "*" and ops != "/":
                    raise AttributeError("Abroated: Invalid Operations Format")
                operations.append(ops)
            is_var = not is_var
        filtered_arrs = []
        for np_arr in var_arrs:
            filtered_np = np_arr[(np_arr[:, 0] >= max_start) & (np_arr[:, 0] <= min_end)]
            filtered_arrs.append(filtered_np)
        
        arr_num = len(filtered_arrs)
        sorted_arr, sorted_hvimp = helper.align_nparr(filtered_arrs)
        filled_data = []
        for i in range(arr_num):
            # Stack the timestamp column with the current value column
            arr = np.column_stack((sorted_arr[:, 0], sorted_arr[:, i+1]))
            filled_data.append(helper.fill_missing_values(arr, match_type))
        final_arr = []
        for i in range(len(sorted_arr)):
            this_timestamp = filled_data[0][i][0]
            calculated_val = filled_data[0][i][1]
            this_hv = sorted_hvimp[i][0]
            this_imp = sorted_hvimp[i][1]
            for j in range(arr_num-1):
                if operations[j] == "+":
                    calculated_val += filled_data[j+1][i][1]
                elif operations[j] == "-":
                    calculated_val -= filled_data[j+1][i][1]
                elif operations[j] == "*":
                    calculated_val *= filled_data[j+1][i][1]
                else:
                    if filled_data[j+1][i][1] == 0:
                        raise ZeroDivisionError("Cannot Divide By 0")
                    calculated_val /= filled_data[j+1][i][1]
            final_arr.append([this_timestamp, calculated_val, this_hv, this_imp])
        
        return np.array(final_arr)