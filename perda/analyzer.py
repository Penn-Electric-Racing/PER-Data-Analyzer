from .csvparser import csvparser
from .datainstance import DataInstance
from .dataplotter import plot


class analyzer:
    def __init__(self):
        """
        Initialize a new analyzer instance.
        Creates new csvparser and initialize.
        """
        self.__parser = csvparser()
        self.__file_read = False
        print("Analyzer Created")

    def read_csv(self, path: str, bad_data_limit: int = 100) -> None:
        """
        Read and parse a CSV file, information stored inside csvparser.

        path: The file path to the CSV file.
        bad_data_limit: Number of allowed bad data rows before raising error.

        Reset csvparser if already have data.
        """
        if self.__file_read:
            self.__parser = csvparser()
            print("Resetting previous data.")
        self.__file_read = True
        self.__parser.read_csv(path, bad_data_limit)

    def get_data(self, variable: str) -> "DataInstance":
        """
        Get DataInstance by canid or name.
        Raises AttributeError if no csv read or cannot find input.
        Very very very very useful function :)) core of perda prob

        variable: name or canid of the variable to get.
        """
        return self.__parser.get_data(variable)

    def print_info(self, input=None, time_unit: str = "s") -> None:
        """
        Print information about the data, a specific variable, or a CAN ID.

        input (Union[None, str, int], optional): What to get info about:
            - None: Print overall dataset information
            - str: Print information about a specific variable name
            - int: Print information about a specific CAN ID
            Defaults to None.
        time_unit (str, optional): Unit for time values ("ms" or "s").
            Defaults to "s".
        """
        self.__parser.print_info(input, time_unit=time_unit)

    def print_variables(
        self, search: str = None, strict_search: bool = False, sort_by: str = "name"
    ) -> None:
        """
        Print a list of all available variables in the dataset.

        sort_by (str, optional): How to sort the variables list:
            - "name": Sort alphabetically by variable name
            - "canid": Sort by CAN ID
            Defaults to "name".
        """
        self.__parser.print_variables(
            search, strict_search=strict_search, sort_by=sort_by
        )

    def plot(
        self,
        left_input,
        right_input=None,
        start_time=0,
        end_time=-1,
        time_unit="s",
        label=True,
        left_spacing=-1,
        right_spacing=-1,
        left_title="",
        right_title="",
        top_title="",
        figsize=(8, 5),
    ) -> None:
        """
        Main plot function, has all functionalities

        left_input (Union[str, List[str], DataInstance, List[DataInstance]]):
            Variable(s) to plot on the left y-axis. Can be variable name(s) or DataInstance(s).
        right_input (Union[str, List[str], DataInstance, List[DataInstance]], optional):
            Variable(s) to plot on the right y-axis. Can be variable name(s) or DataInstance(s).
            If None, only uses left y-axis. Defaults to None.
        start_time (int/float, optional): Start time for plotting. Defaults to 0.
        end_time (int/float, optional): End time for plotting. -1 means until the end.
            Defaults to -1.
        time_unit (str, optional): Unit for time axis, either "ms" or "s".
            Affects both display and input interpretation. Defaults to "ms".
        label (bool, optional): Whether to show plot legends. Defaults to True.
        left_spacing (float, optional): Spacing between major ticks on left y-axis.
            -1 means auto spacing. Defaults to -1.
        right_spacing (float, optional): Spacing between major ticks on right y-axis.
            -1 means auto spacing. Defaults to -1.
        left_title (str, optional): Label for left y-axis. Defaults to "".
        right_title (str, optional): Label for right y-axis. Defaults to "".
        top_title (str, optional): Title for the entire plot. Defaults to "".
        figsize (tuple, optional): Figure size in inches (width, height).
            Defaults to (8, 5).

        Note:
            When using time_unit="s", start_time and end_time are interpreted as seconds
            and converted to milliseconds internally.
        """
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
