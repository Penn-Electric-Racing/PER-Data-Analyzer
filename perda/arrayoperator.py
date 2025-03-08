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

    def get_csvparser(self, cp: csvparser):
        self.__csvparser = cp
        self.__file_read = True

    def get_compute_arrays(self, op_list: list[str], match_type: str = "extend", start_time = 0, end_time = -1, unit = "s"):
        if not self.__file_read:
            print("No csv read. Call .get_csvparser() before plotting.")
            return
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
                        print("Cannot Divide By 0")
                        return None
                    calculated_val /= filled_data[j+1][i][1]
            final_arr.append([this_timestamp, calculated_val, this_hv, this_imp])
        
        return np.array(final_arr)