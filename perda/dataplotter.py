import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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

    def set_csvparser(self, cp: csvparser):
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

    def plot_norm(self, variables: list, labels: bool = True, label_list: list[str] = [], h_lines: list = [], v_lines: list = []):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        for h in h_lines:
            plt.axhline(y=h, color='black', linestyle='--', zorder=1)
        for v in v_lines:
            plt.axvline(x=v, color='black', linestyle='--', zorder=1)
        
        for index, var in enumerate(variables):
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
            this_label = ""
            if index >= len(label_list):
                this_label = short_name
            else:
                this_label = label_list[index]

            plt.plot(t, y, label=this_label)
            if labels:
                plt.legend()

            if not self.__plot_same_graph:
                plt.suptitle(short_name)
                plt.show()
                print("\n\n")
                for h in h_lines:
                    plt.axhline(y=h, color='black', linestyle='--')
                
                for v in v_lines:
                    plt.axvline(x=v, color='black', linestyle='--')
        
        if self.__plot_same_graph:
            plt.show()
            print("\n\n")
        
    def plot_dual(self, variables: list, h_lines: list = [], v_lines: list = []):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        num_vars = len(variables)
        grouped_vars = [variables[i:i+2] for i in range(0, num_vars, 2)]
        xlabel = "Timestamp (s)" if self.__plot_unit == "s" else "Timestamp (ms)"
        for group in grouped_vars:
            fig, ax1 = plt.subplots()
            ax2 = None

            for h in h_lines:
                plt.axhline(y=h, color='black', linestyle='--', zorder=1)
            
            for v in v_lines:
                plt.axvline(x=v, color='black', linestyle='--', zorder=1)
    
            for i, var in enumerate(group):
                if type(var) is list:
                    self.plot(var)
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
                if filtered_vals.size == 0:
                    print(f"Warning: No data available for {short_name} in the given time range.")
                    continue

                if self.__plot_unit == "s":
                    t = filtered_vals[::, 0]/1e3
                else:
                    t = filtered_vals[::, 0]

                for hvT, hvOn in self.__csvparser.get_HV_changes():
                    color = "red" if hvOn else "green"
                    if (hvT >= first_time_stamp and hvT <= last_time_stamp):
                        if self.__plot_unit == "s":
                            hvT = hvT/1e3
                        plt.axvline(x=hvT, color=color)

                y = filtered_vals[::, 1]

                if self.__plot_same_graph:
                    if i == 0:
                        ax1.plot(t, y, label=short_name, color='b')
                        ax1.set_ylabel(short_name, color='b')
                        ax1.tick_params(axis='y', labelcolor='b')
                    else:
                        ax2 = ax1.twinx()
                        ax2.plot(t, y, label=short_name, color='r') #, linestyle="--"
                        ax2.set_ylabel(short_name, color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                else:
                    fig, ax = plt.subplots()
                    ax.plot(t, y, label=short_name, color="b")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(short_name, color="b")
                    ax.tick_params(axis="y", labelcolor="b")
                    ax.set_title(short_name)
                    ax.legend(loc="upper left")
                    print(f"graph for {var}")
                    plt.show()
                    print("\n\n")
            
            if self.__plot_same_graph:
                valid_names = []
                for var in group:
                    if isinstance(var, str):  
                        valid_names.append(var)
                    elif isinstance(var, np.ndarray):  
                        valid_names.append("Input Array")  
                if len(valid_names) >= 1:
                    plt.xlabel(xlabel)
                    title = " and ".join(valid_names)
                    plt.title(f"Dual Graph for {title}" if len(valid_names) == 2 else f"Single Y-Axis Graph for {title}")
                    fig.tight_layout()
                    print(f"graph for {title}")
                    plt.show()
                    print("\n\n")
       
    def plot_dual_group(self, l_v: list, r_v: list, left_title: str = "Left", right_title: str = "Right", middle_title: str = "title", labels = False, left_spacing = -1, right_spacing = -1, h_lines: list = [], v_lines: list = []):
        if not self.__file_read:
            raise AttributeError("No csv read. Call .get_csvparser() before plotting.")
        left_len = len(l_v)
        both_l = l_v + r_v
        xlabel = "Timestamp (s)" if self.__plot_unit == "s" else "Timestamp (ms)"
        fig, ax1 = plt.subplots()
        ax2 = None
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_index = 0


        for h in h_lines:
            plt.axhline(y=h, color='black', linestyle='--', zorder=1)
        
        for v in v_lines:
            plt.axvline(x=v, color='black', linestyle='--', zorder=1)

        for index, var in enumerate(both_l):
            if type(var) is str:
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
            if filtered_vals.size == 0:
                print(f"Warning: No data available for {short_name} in the given time range.")
                continue

            if self.__plot_unit == "s":
                t = filtered_vals[::, 0]/1e3
            else:
                t = filtered_vals[::, 0]

            for hvT, hvOn in self.__csvparser.get_HV_changes():
                color = "red" if hvOn else "green"
                if (hvT >= first_time_stamp and hvT <= last_time_stamp):
                    if self.__plot_unit == "s":
                        hvT = hvT/1e3
                    plt.axvline(x=hvT, color=color)

            y = filtered_vals[::, 1]

            if index < left_len:
                ax1.plot(t, y, label=short_name, color = colors[index % len(colors)])
                ax1.set_ylabel(left_title)
                ax1.tick_params(axis='y')
                if left_spacing != -1:
                    ax1.yaxis.set_major_locator(MultipleLocator(left_spacing))
            else:
                if ax2 is None:
                    ax2 = ax1.twinx()
                ax2.plot(t, y, label=short_name, color = colors[index % len(colors)]) #, linestyle="--"
                ax2.set_ylabel(right_title)
                ax2.tick_params(axis='y')
                if right_spacing != -1:
                    ax2.yaxis.set_major_locator(MultipleLocator(right_spacing))
        plt.xlabel(xlabel)
        plt.title(f"Join Graph for {middle_title}")
        if labels:
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        fig.tight_layout()
        plt.show()
    
    def plot0to60(self, numWheels):
        self.set_plot()
        short_names = ["pcm.wheelSpeeds.frontLeft", "pcm.wheelSpeeds.frontRight", 
                       "pcm.wheelSpeeds.backLeft", "pcm.wheelSpeeds.backRight"]

        didNotReach60 = 0
    
        for i in range(numWheels):
            minTime = (float('inf'), -1, -1, -1) # duration, startTime, endTime, endSpeed
            vals = self.__csvparser.get_np_array(short_names[i])

            if vals.shape[1] < 2:  # Ensure vals has at least 2 columns (time, speed)
                print(f"Skipping {short_names[i]} due to unexpected format")
                continue

            lastTime, lastSpeed, startTime, endTime = -1, -1, -1, -1

            reached60 = False
            for row in vals:
                t, y = row[:2] 
                t /= 1000
                # wheel speed sensor is NaN if 0
                if np.isnan(y) or y <= 0:
                    startTime = t
                elif y >= 60 and lastSpeed < 60:
                    endTime = lastTime + (60 - lastSpeed) * (t - lastTime) / (y - lastSpeed)

                    duration = endTime - startTime

                    if duration < minTime[0]:
                        minTime = (duration, startTime, endTime, y)
                    
                    reached60 = True
                    break

                lastTime = t
                lastSpeed = y

            if not reached60:
                print(f"{short_names[i]} Did not reach 60")
                didNotReach60 += 1
                continue

            t = vals[:, 0]
            y = vals[:, 1]

            mask = (t >= (minTime[1] * 1000)) & (t <= (minTime[2] * 1000))
            t = t[mask]/1e3
            y = y[mask]

            plt.figure(figsize=(8, 5))
            plt.plot(t, y, label = short_names[i])
            plt.axhline(y=60, color='red', linestyle="--",)
            plt.xlabel("Timestamp (s)")
            plt.ylabel("Speed (MPH)")
            plt.title("0-60 MPH in " + str(minTime[0]) + "s")
            plt.legend()
            plt.show()
            
        if (didNotReach60 == numWheels):
            print(f"No wheels reached 60mph")
            