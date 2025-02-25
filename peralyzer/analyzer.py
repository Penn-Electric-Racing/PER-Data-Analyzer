import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class analyzer:
    def __init__(self):
        self.value_map = {}
        self.ID_map = {}
        self.hv_changes = []
        print("Analyzer initialized!")

    def read_csv(self, path):
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
                            time = int(data[0])
                            if name not in self.value_map:
                                self.value_map[name] = []
                            self.value_map[name].append([time,val])
                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    line_num += 1
                    pbar.update(1)

    def find_variable(self, short_name, value_map):
        full_name = None
        for var_name in value_map.keys():
            if f'({short_name})' in var_name:
                full_name = var_name
                break

        if full_name is None:
            print("Error: could not find data for " + short_name)
            return None

        return np.array(value_map[full_name])
    
    def plot(self, variables, same_graph=False):
        for var in variables:
            if type(var) is list:
                self.plot(var, True)
                continue

            short_name = var[0] if type(var) is tuple else var

            vals = self.find_variable(short_name, self.value_map)

            if vals is None:
                continue

            t = vals[::, 0]
            y = vals[::, 1]

            if type(var) is tuple:
                f = var[1]
                y = [f(n) for n in y]

            plt.plot(t, y, label=short_name)
            plt.xlabel("Timestamp (s)")
            plt.legend()

            if not same_graph:
                plt.suptitle(short_name)
                plt.show()
                print("\n\n")

        if same_graph:
            plt.show()
            print("\n\n")

    def analyze(self, data):
        return f"Analyzing {data}"
