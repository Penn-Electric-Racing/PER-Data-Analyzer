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

    def read_csv(self, path):
        if self.__file_read:
            print("Call .reset() before reading new csv")
            return
        # Reset but do not print
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
        if unit != "s" and unit != "ms" and unit != "us":
            print("Invalid unit input, please use \"s\", \"ms\", or \"us\". Default to s.")
            self.__plot_unit = "s"
        else:
            self.__plot_unit = unit

    def plot(self, variables):
        if not self.__file_read:
            print("No csv read. Call .read_csv() before plotting.")
            return
        for var in variables:
            if type(var) is list:
                self.plot(var, True)
                continue

            short_name = var
            vals = self.get_np_array(short_name)
            if vals is None:
                continue

            last_time_stamp = self.__plot_end_time
            first_time_stamp = self.__plot_start_time

            if self.__plot_unit == "s":
                last_time_stamp *= 1e3
                first_time_stamp *= 1e3
            elif self.__plot_unit == "us":
                last_time_stamp /= 1e3
                first_time_stamp /= 1e3
            
            if self.__plot_end_time == -1:
                last_time_stamp = vals[-1,0]
            
            filtered_vals = vals[(vals[:, 0] >= first_time_stamp) & (vals[:, 0] <= last_time_stamp )]

            if self.__plot_unit == "s":
                t = filtered_vals[::, 0]/1e3
                plt.xlabel("Timestamp (s)")
            elif self.__plot_unit == "ms":
                t = filtered_vals[::, 0]
                plt.xlabel("Timestamp (ms)")
            else:
                t = filtered_vals[::, 0]*1e3
                plt.xlabel("Timestamp (us)")

            for hvT, hvOn in self.__high_voltage_changes:
                color = "red" if hvOn else "green"
                if (hvT >= first_time_stamp and hvT <= last_time_stamp):
                    if self.__plot_unit == "s":
                        hvT = hvT/1e3
                    elif self.__plot_unit == "us":
                        hvT = hvT* 1e3
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

    def get_np_array(self, short_name):
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
    
    def __name_matches(self, short_name, full_name):
        return f'({short_name})' in full_name