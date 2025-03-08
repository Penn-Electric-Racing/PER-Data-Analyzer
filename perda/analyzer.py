from .csvparser import csvparser
from .dataplotter import dataplotter
from .arrayoperator import arrayoperator

class analyzer:
    def __init__(self):
        self.__csvparser = csvparser()
        self.__dataplotter = dataplotter()
        self.__operator = arrayoperator()
        self.__file_read = False

        print("Analyzer Created")

    def reset(self):
        self.__csvparser.reset()
        self.__dataplotter.reset()
        self.__operator.reset()
        self.__file_read = False

        print("Reset Analyzer")

    def read_csv(self, path: str):
        if self.__file_read:
            print("Call .reset() before reading new csv")
            return
        self.__csvparser.read_csv(path)
        self.__dataplotter.get_csvparser(self.__csvparser)
        self.__operator.get_csvparser(self.__csvparser)
        self.__file_read = True

    def set_plot(self, start_time = 0, end_time = -1, same_graph = True, unit = "s"):
        self.__dataplotter.set_plot(start_time, end_time, same_graph, unit)

    def plot(self, variables: list):
        self.__dataplotter.plot(variables)

    def get_nparray(self, short_name: str):
        return self.__csvparser.get_np_array(short_name)
    
    def get_filtered_nparray(self, arr, start_time = 0, end_time = -1, unit = "s"):
        return self.__operator.get_filtered_nparr(arr, start_time, end_time, unit)

    def get_compute_arrays(self, op_list: list[str], match_type: str = "extend", start_time = 0, end_time = -1, unit = "s"):
        return self.__operator.get_compute_arrays(op_list, match_type, start_time, end_time, unit)
    
    def get_integral(self, arr, start_time = 0, end_time = -1, unit = "s"):
        filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, unit)
        integral, _ = self.__operator.get_integral_avg(filtered_arr)
        if unit == "s":
            integral /= 1e3
        return integral
    
    def get_average(self, arr, start_time = 0, end_time = -1, unit = "s"):
        filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, unit)
        _, average = self.__operator.get_integral_avg(filtered_arr)
        return average
    
    def analyze_data(self, arr_list: list, start_time = 0, end_time = -1, unit = "s"):
        arr_num = 0
        for arr in arr_list:
            if type(arr) is str:
                var_name = arr
            else:
                var_name = f"Input Data {arr_num}"
            
            filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, unit)
            data_start_time = filtered_arr[0,0]
            data_end_time = filtered_arr[-1,0]
            duration = data_end_time - data_start_time
            integral, average = self.__operator.get_integral_avg(filtered_arr)
            max_value, max_timestamp, min_value, min_timestamp = self.__operator.get_max_min(filtered_arr)
            decimals = 3

            if unit == "s":
                max_timestamp /= 1e3
                min_timestamp /= 1e3
                data_start_time /= 1e3
                data_end_time /= 1e3
                duration /= 1e3
                integral /= 1e3
                decimals = 6

            print(f"Statistics for **{var_name}**:")
            print(f"Start: {format(data_start_time,f",.{decimals}f")}" + unit + f" | End: {format(data_end_time,f",.{decimals}f")}" + unit + f" | Duration: {format(duration,f",.{decimals}f")}" + unit)
            print(f"Max Value: {max_value} ({format(max_timestamp,f",.{decimals}f")}" + unit + ")")
            print(f"Min Value: {min_value} ({format(min_timestamp,f",.{decimals}f")}" + unit + ")")
            print(f"Average: {average}")
            print(f"Integral: {integral}")
            print("\n")

