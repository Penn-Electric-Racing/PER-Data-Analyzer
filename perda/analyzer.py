import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class analyzer:
    def __init__(self):
        self.value_map = {}
        self.ID_map = {}
        self.high_voltage_changes = []
        self.file_read = False
        self.start_time = None
        self.end_time = None
        print("Analyzer Initialized")

    def reset(self):
        self.value_map = {}
        self.ID_map = {}
        self.high_voltage_changes = []
        self.file_read = False
        self.start_time = None
        self.end_time = None
        print("Reset Analyzer")

    def read_csv(self, path):
        if self.file_read:
            print("Call .reset() before reading new csv")
            return
        
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
                            self.ID_map[int(canID_name_value[1])] = canID_name_value[0]
                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    else:
                        data = line.strip().split(",")
                        try:
                            id = int(data[1])
                            name = self.ID_map[id]
                            val = float(data[2])
                            raw_time = int(data[0])

                            if not start_time_read and self.name_matches("sdl.startTime", name):
                                self.start_time = val
                                start_time_read = True
                            elif self.name_matches("sdl.currentTime", name):
                                time_stamp = val / 1e6
                            elif self.name_matches("ams.airsState", name):
                                if high_voltage and val == 0:
                                    high_voltage = False
                                    self.high_voltage_changes.append((time_stamp, high_voltage))
                                elif not high_voltage and val == 4:
                                    high_voltage = True
                                    self.high_voltage_changes.append((time_stamp, high_voltage))
                            elif self.name_matches("pcm.pedals.implausibility.anyImplausibility", name):
                                any_implausibility = val

                            if name not in self.value_map:
                                self.value_map[name] = []
                            
                            self.value_map[name].append([time_stamp, val, high_voltage, any_implausibility, raw_time])

                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    line_num += 1
                    pbar.update(1)
        self.end_time = raw_time
        self.file_read = True

    def plot(self, variables, same_graph, start_time = 0, end_time = -1, unit_ms = False):
        if not self.file_read:
            print("No csv read. Call .read_csv() before plotting.")
            return
        for var in variables:
            if type(var) is list:
                self.plot(var, True)
                continue

            short_name = var[0] if type(var) is tuple else var

            vals = self.find_variable(short_name, self.value_map)

            if vals is None:
                continue

            # if not unit_ms:
            #     t = vals[::, 0]/1e6
            #     plt.xlabel("Timestamp (s)")
            # else:
            #     t = vals[::, 0]
            #     plt.xlabel("Timestamp (ms)")

            t = vals[::, 0]
            y = vals[::, 1]

            # if type(var) is tuple:
            #     f = var[1]
            #     y = [f(n) for n in y]

            plt.plot(t, y, label=short_name)
            plt.xlabel("Timestamp (s)")
            plt.legend()

            for hvT, hvOn in self.high_voltage_changes:
                color = "red" if hvOn else "green"
                plt.axvline(x=hvT, color=color)

            if not same_graph:
                plt.suptitle(short_name)
                plt.show()
                print("\n\n")

        if same_graph:
            plt.show()
            print("\n\n")

    def find_variable(self, short_name, value_map):
        full_name = None
        for var_name in value_map.keys():
            if self.name_matches(short_name, var_name):
                full_name = var_name
                break

        if full_name is None:
            print("Error: could not find data for " + short_name)
            return None

        return np.array(value_map[full_name])
    
    def name_matches(self, short_name, full_name):
        return f'({short_name})' in full_name