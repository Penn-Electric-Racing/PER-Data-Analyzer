import numpy as np
from tqdm import tqdm

from . import helper


class csvparser:
    def __init__(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__already_filtered = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None
        self.__file_read = False

    def reset(self):
        self.__value_map = {}
        self.__ID_map = {}
        self.__already_filtered = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None
        self.__file_read = False

    def read_csv(self, path: str, input_align_name="default"):
        if self.__file_read:
            raise AttributeError("Call .reset() before reading new csv")
        # Reset but do not print
        self.__value_map = {}
        self.__ID_map = {}
        self.__already_filtered = {}
        self.__high_voltage_changes = []
        self.__data_start_time = None
        self.__data_end_time = None

        time_stamp = -1
        start_time_read = False
        high_voltage = False
        any_implausibility = False

        align_id = None
        align_name = input_align_name

        bad_data = 0

        if input_align_name == "max_freq":

            count_dic = {}

            with open(path, "r") as pre_log:
                header = next(pre_log)
                line_num = 2
                with tqdm(
                    desc="Finding Highest Freq Value", unit="lines", initial=1
                ) as pbar:
                    for pre_line in pre_log:
                        if bad_data >= 100:
                            raise Exception("Bad data exceeds 100, aborted.")
                        if pre_line.startswith("Value"):
                            canID_name_value = pre_line[6:].strip().split(": ")
                            try:
                                this_id = int(canID_name_value[1])
                                self.__ID_map[this_id] = canID_name_value[0]
                            except Exception as e:
                                print(
                                    f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}"
                                )
                                bad_data += 1
                                continue
                        else:
                            data = pre_line.strip().split(",")
                            try:
                                id = int(data[1])

                                if id not in count_dic:
                                    count_dic[id] = 0

                                count_dic[id] += 1

                            except Exception as e:
                                print(
                                    f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}"
                                )
                                bad_data += 1
                                continue
                        line_num += 1
                        pbar.update(1)
            align_id = max(count_dic, key=count_dic.get)
            num_data = count_dic[align_id]
            max_name = self.__ID_map[align_id]
            align_name = max_name
            print(
                f"Timestamp aligned to {max_name} | canID: {align_id} | Number of data: {num_data}\n"
            )

        with open(path, "r") as log:
            header = next(log)
            print(f"Reading file: {header}")
            line_num = 2
            with tqdm(desc="Processing CSV", unit="lines", initial=1) as pbar:
                for line in log:
                    if bad_data >= 100:
                        raise Exception("Bad data exceeds 100, aborted.")
                    if line.startswith("Value"):
                        if input_align_name == "max_freq":
                            continue
                        canID_name_value = line[6:].strip().split(": ")
                        try:
                            this_id = int(canID_name_value[1])
                            self.__ID_map[this_id] = canID_name_value[0]
                            if helper.name_matches(align_name, canID_name_value[0]):
                                align_id = this_id
                        except Exception as e:
                            print(
                                f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}"
                            )
                            bad_data += 1
                            continue
                    else:
                        if align_name != "default" and align_id is None:
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
                            elif (
                                helper.name_matches(align_name, name) or id == align_id
                            ):
                                # time_stamp = val / 1e3
                                time_stamp = raw_time
                            elif helper.name_matches("ams.airsState", name):
                                if high_voltage and val == 0:
                                    high_voltage = False
                                    self.__high_voltage_changes.append(
                                        (time_stamp, high_voltage)
                                    )
                                elif not high_voltage and val == 4:
                                    high_voltage = True
                                    self.__high_voltage_changes.append(
                                        (time_stamp, high_voltage)
                                    )
                            elif helper.name_matches(
                                "pcm.pedals.implausibility.anyImplausibility", name
                            ):
                                any_implausibility = val

                            if name not in self.__value_map:
                                self.__value_map[name] = []

                            if align_name == "default":
                                self.__value_map[name].append(
                                    [raw_time, val, high_voltage, any_implausibility]
                                )
                            else:
                                self.__value_map[name].append(
                                    [time_stamp, val, high_voltage, any_implausibility]
                                )

                        except Exception as e:
                            print(
                                f"Error parsing ID | Line number {line_num} | Line: {line} | Error: {e}"
                            )
                            bad_data += 1
                            continue
                    line_num += 1
                    pbar.update(1)
        self.__data_end_time = raw_time
        self.__file_read = True
        print("Csv parsing complete.")
        if bad_data != 0:
            print(f"Number of bad data: {bad_data}")

    # might need in the future
    def filter_data_replace(self, value_map: dict):
        window_size = 5
        for key in value_map:
            data = np.array(value_map[key])
            if data.shape[0] < window_size:
                continue

            time_col = data[:, 0]
            values_col = data[:, 1]
            cleaned_values = values_col.copy()

            for i in range(len(values_col) - window_size + 1):
                window = values_col[i : i + window_size]
                Q1 = np.percentile(window, 25)
                Q3 = np.percentile(window, 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                mid_idx = i + window_size // 2
                if (
                    values_col[mid_idx] < lower_bound
                    or values_col[mid_idx] > upper_bound
                ):
                    median_value = np.median(window)
                    if lower_bound <= median_value <= upper_bound:
                        cleaned_values[mid_idx] = median_value

            value_map[key] = np.column_stack(
                (time_col, cleaned_values, data[:, 2:])
            ).tolist()

    def filter_data_drop(self, key: str):
        if self.__already_filtered.get(key, False):
            return self.__value_map[key]
        outlier_count = 0
        with tqdm(
            desc=f"Cleaning outliers for {key}. ", unit=" windows", initial=1
        ) as pbar:
            data = np.array(self.__value_map[key])
            N = data.shape[0]
            if N < 5:
                return self.__value_map[key]

            window_size = max(5, int(np.sqrt(N)))
            if window_size % 2 == 0:
                window_size += 1

            values_col = data[:, 1]

            valid_indices = []
            firstCol = True

            for i in range(N - window_size + 1):
                window = values_col[i : i + window_size]
                Q1 = np.percentile(window, 25)
                Q3 = np.percentile(window, 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if firstCol:
                    for idx in range(window_size // 2):
                        if lower_bound <= values_col[idx] <= upper_bound:
                            valid_indices.append(idx)
                        else:
                            outlier_count += 1
                    firstCol = False
                mid_idx = i + (window_size // 2)
                if lower_bound <= values_col[mid_idx] <= upper_bound:
                    valid_indices.append(mid_idx)
                if i == N - window_size:
                    for idx in range(i + (window_size // 2), N):
                        if lower_bound <= values_col[idx] <= upper_bound:
                            valid_indices.append(idx)
                        else:
                            outlier_count += 1
                pbar.update(1)

            # Convert valid indices into a filtered dataset
            filtered_data = data[valid_indices]

            # Store the cleaned data back in value_map as a list of lists
            self.__value_map[key] = filtered_data.tolist()
            self.__already_filtered[key] = True
        print(f"{outlier_count} outliers found for {key}.")
        return self.__value_map[key]

    def filter_all_data(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        print(
            "Warning: This is an expensive and time consuming operation. Do you wish to continue?"
        )
        while True:
            inp = input("Y/N: ")
            if inp == "Y" or inp == "y":
                break
            elif inp == "N" or inp == "n":
                return
            else:
                print("Invalid input.")
        for eachKey in self.__value_map:
            self.filter_data_drop(eachKey)

    def get_np_array(self, short_name: str, filter=False):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        full_name = None
        for var_name in self.__value_map.keys():
            if helper.name_matches(short_name, var_name):
                full_name = var_name
                break

        if full_name is None:
            raise AttributeError("Error: could not find data for " + short_name)
        if filter:
            return np.array(self.filter_data_drop(full_name))
        return np.array(self.__value_map[full_name])

    def get_input_can_variables(self):
        """
        Return a NumPy array of tuples:
            (outside_text, inside_text)
        • If a value looks like 'foo (bar)', it becomes ('foo', 'bar')
        • Otherwise it becomes ('foo', '')
        The array is lexicographically sorted by the two fields.
        """
        pairs = []

        for raw in self.__ID_map.values():
            s = raw.strip()
            left = s.rfind("(")
            # well-formed "( … )" at the end?
            if left != -1 and s.endswith(")"):
                outside = s[:left].rstrip()
                inside = s[left + 1 : -1].strip()  # drop '(' and ')'
                if inside:  # both parts exist
                    pairs.append((inside, outside))
                    continue
            # fallback: single column
            pairs.append((s, ""))

        # sort by outside first, then inside
        sorted_pairs = np.array(sorted(pairs, key=lambda t: (t[0], t[1])), dtype=object)
        return sorted_pairs

    def get_value_map(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        print("Not all data in the value map might have outliers filtered.")
        return self.__value_map

    def get_ID_map(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        return self.__ID_map

    def get_can_id(self, name: str):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        canID = next(
            (k for k, v in self.__ID_map.items() if helper.name_matches(name, v)), None
        )
        if canID is None:
            raise AttributeError("No Can ID Found")
        return canID

    def get_HV_changes(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        return self.__high_voltage_changes

    def get_data_start_time(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        return self.__data_start_time

    def get_data_end_time(self):
        if not self.__file_read:
            raise AttributeError("Empty parser, read csv before calling.")
        return self.__data_end_time

    def is_empty(self):
        return not self.__file_read
