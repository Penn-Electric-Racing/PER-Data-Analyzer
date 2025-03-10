import numpy as np
import matplotlib.pyplot as plt

from .csvparser import csvparser

class dataplotter:
    def __init__(self):
        self.__csvparser = csvparser()
        self.__file_read = False

        self.__plot_same_graph = True
        self.__plot_start_time = 0
        self.__plot_end_time = -1
        self.__plot_unit = "s"

    def reset(self):
        self.__csvparser = csvparser()
        self.__file_read = False

        self.__plot_same_graph = True
        self.__plot_start_time = 0
        self.__plot_end_time = -1
        self.__plot_unit = "s"

    def get_csvparser(self, cp: csvparser):
        self.__csvparser = cp
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
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        for var in variables:
            if type(var) is list:
                self.plot(var, True)
                continue
            elif type(var) is str:
                short_name = var
                vals = self.__csvparser.get_np_array(short_name)
            elif isinstance(var, np.ndarray):
                if var.ndim != 2 or var.shape[1] < 2:
                    print("Invalid Input Array For Plotting")
                    continue
                short_name = "Input Array"
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

            for hvT, hvOn in self.__csvparser.get_HV_changes():
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