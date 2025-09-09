from .csvparser import csvparser
from .dataplotter import plot


class analyzer:
    def __init__(self):
        """
        todo
        """
        self.__parser = csvparser()
        self.__file_read = False
        print("Analyzer Created")

    def read_csv(self, path: str, bad_data_limit=100):
        """
        todo
        """
        if self.__file_read:
            self.__parser = csvparser()
            print("Resetting previous data.")
        self.__file_read = True
        self.__parser.read_csv(path, bad_data_limit)

    def get_data(self, variable: str):
        """
        todo
        """
        return self.__parser.get_data(variable)

    def print_info(self, input=None, time_unit: str = "s"):
        """
        Generic info print function.
        Input can be None (for overall info), str (for variable name), or int (for CAN ID).
        """
        self.__parser.print_info(input, time_unit=time_unit)

    def print_variables(self, sort_by="name"):
        """
        Function to show all possible variables in the data.
        """
        self.__parser.print_variables(sort_by=sort_by)

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
        figsize=(8, 5),
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
            figsize=figsize,
        )
