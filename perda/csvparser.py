import numpy as np
from tqdm import tqdm

from . import helper

class csvparser:
    def __init__(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None
        self.__file_read = False
        self.__dic_count = {}

    def reset(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None
        self.__file_read = False
        self.__dic_count = {}
    
    def read_csv(self, path: str, input_align_name = "default"):
        if self.__file_read:
            print("Call .reset() before reading new csv")
            return
        # Reset but do not print
        self.__value_map = {}
        self.__ID_map = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None

        time_stamp = -1
        start_time_read = False
        high_voltage = False
        any_implausibility = False

        align_id = None
        align_name = input_align_name

        if input_align_name == "default":

            count_dic = {}

            with open(path, 'r') as pre_log:
                header = next(pre_log)
                line_num = 2
                with tqdm(desc="Finding Highest Freq Value", unit="lines", initial = 1) as pbar:
                    for pre_line in pre_log:
                        if (pre_line.startswith("Value")):
                            canID_name_value = pre_line[6:].strip().split(": ")
                            try:
                                this_id = int(canID_name_value[1])
                                self.__ID_map[this_id] = canID_name_value[0]
                            except Exception as e:
                                print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                                break
                        else:
                            data = pre_line.strip().split(",")
                            try:
                                id = int(data[1])

                                if id not in count_dic:
                                    count_dic[id] = 0
                                
                                count_dic[id] += 1

                            except Exception as e:
                                print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                                break
                        line_num += 1
                        pbar.update(1)
            align_id = max(count_dic, key=count_dic.get)
            num_data = count_dic[align_id]
            max_name = self.__ID_map[align_id]
            align_name = max_name
            self.__dic_count = count_dic
            print(f"Timestamp aligned to {max_name} | canID: {align_id} | Number of data: {num_data}\n")

        with open(path, 'r') as log:
            header = next(log)
            print(f"Reading file: {header}")
            line_num = 2
            with tqdm(desc="Processing CSV", unit="lines", initial = 1) as pbar:
                for line in log:
                    if (line.startswith("Value")):
                        if input_align_name == "default":
                            continue
                        canID_name_value = line[6:].strip().split(": ")
                        try:
                            this_id = int(canID_name_value[1])
                            self.__ID_map[this_id] = canID_name_value[0]
                            if helper.name_matches(align_name, canID_name_value[0]):
                                align_id = this_id
                        except Exception as e:
                            print(f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}")
                            break
                    else:
                        if align_id is None:
                            print("Cannot Find Align Variable Name.")
                            return None
                        data = line.strip().split(",")
                        try:
                            id = int(data[1])
                            name = self.__ID_map[id]
                            val = float(data[2])
                            raw_time = int(data[0])

                            if not start_time_read:
                                self.__data_start_time = raw_time
                                time_stamp = self.__data_start_time
                                start_time_read = True
                            elif (helper.name_matches(align_name, name) or id == align_id):
                                # time_stamp = val / 1e3
                                time_stamp = raw_time
                            elif helper.name_matches("ams.airsState", name):
                                if high_voltage and val == 0:
                                    high_voltage = False
                                    self.__high_voltage_changes.append((time_stamp, high_voltage))
                                elif not high_voltage and val == 4:
                                    high_voltage = True
                                    self.__high_voltage_changes.append((time_stamp, high_voltage))
                            elif helper.name_matches("pcm.pedals.implausibility.anyImplausibility", name):
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
    
    def get_np_array(self, short_name: str):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return None
        full_name = None
        for var_name in self.__value_map.keys():
            if helper.name_matches(short_name, var_name):
                full_name = var_name
                break

        if full_name is None:
            print("Error: could not find data for " + short_name)
            return None
        return np.array(self.__value_map[full_name])
    
    def get_value_map(self):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        return self.__value_map
    
    def get_ID_map(self):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        return self.__ID_map
    
    def get_can_id(self, name: str):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        canID = next((k for k, v in self.__ID_map.items() if helper.name_matches(name, v)), None)
        if canID is None:
            print("No Can ID Found")
        return canID
    
    def get_HV_changes(self):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        return self.__high_voltage_changes
    
    def get_data_start_time(self):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        return self.__data_start_time
    
    def get_data_end_time(self):
        if not self.__file_read:
            print("Empty parser, read csv before calling.")
            return
        return self.__data_end_time
    
    def is_empty(self):
        return not self.__file_read
