from .newparser import newparser
from .newplotter import plot


class newanalyzer:
    def __init__(self):
        """
        todo
        """
        self.__parser = newparser()
        self.__file_read = False
        print("Analyzer Created")

    def read_csv(self, path: str, bad_data_limit=100):
        """
        todo
        """
        if self.__file_read:
            self.__parser = newparser()
            print("Resetting previous data.")
        self.__file_read = True
        self.__parser.read_csv(path, bad_data_limit)

    def get_data(self, variable: str):
        """
        todo
        """
        return self.__parser.get_data(variable)

    # def print_variables(self):
    #     input_variables = self.__csvparser.get_input_can_variables()
    #     col1_width = max(len(id) for id, _ in input_variables)
    #     print(f"{'CAN Variables':<{col1_width}}  Info")
    #     print("-" * (col1_width + 7))  # optional separator
    #     for id, discript in input_variables:
    #         print(f"{id:<{col1_width}}  {discript}")

    def plot(
        self,
        left_input,
        right_input=None,
        start_time=0,
        end_time=-1,
        time_unit="ms",
        label=True,
        left_spacing=-1,
        right_spacing=-1,
        left_title="",
        right_title="",
        top_title="",
    ):
        """
        todo
        """
        if time_unit == "s":
            start_time *= 1e3
            end_time *= 1e3
        plot(
            self.__parser,
            left_input,
            right_input=right_input,
            start_time=start_time,
            end_time=end_time,
            unit=time_unit,
            label=label,
            left_spacing=left_spacing,
            right_spacing=right_spacing,
            left_title=left_title,
            right_title=right_title,
            top_title=top_title,
        )

    # def align_array(
    #     self,
    #     align_list: list,
    #     match_type: str = "connect",
    #     start_time=0,
    #     end_time=-1,
    #     time_unit="s",
    # ):
    #     """Align multiple arrays based on timestamps.

    #     Args:
    #         align_list (list): List of variable names or arrays to align
    #         match_type (str, optional):  The method used for filling missing values.
    #                 "connect" - linear interpolation between nearest valid neighbors (default).
    #                 "extend_forward" - use the last valid value.
    #                 "extend_back" - use the next valid value.
    #         start_time (int, optional): Start time for alignment. Defaults to 0.
    #         end_time (int, optional): End time for alignment. -1 for end of data. Defaults to -1.
    #         time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".

    #     Returns:
    #         list: List of aligned arrays
    #     """
    #     return self.__operator.align_arrays(
    #         align_list, match_type, start_time, end_time, time_unit
    #     )
