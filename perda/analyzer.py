from typing import Union, List, Optional, Tuple
from .csv_parser import CSVParser, SingleRunData
from .data_instance import DataInstance
from .data_plotter import plot
from . import data_helpers


class Analyzer:
    def __init__(self):
        """
        Initialize a new analyzer instance.
        """
        self.data: SingleRunData = None
        print("Analyzer Created")

    def read_csv(self, path: str, bad_data_limit: int = 100) -> None:
        """
        Read and parse a CSV file, storing the result in self.data.

        path: The file path to the CSV file.
        bad_data_limit: Number of allowed bad data rows before raising error.
        """
        if self.data is not None:
            print("Resetting previous data.")
        parser = CSVParser()
        self.data = parser(path, bad_data_limit)

    def get_data(self, variable: str) -> "DataInstance":
        """
        Get DataInstance by canid or name.
        Raises AttributeError if no csv read or cannot find input.
        Very very very very useful function :)) core of perda prob

        variable: name or canid of the variable to get.
        """
        if self.data is None:
            raise AttributeError("No csv read. Call .read_csv() before getting data.")
        return self.data[variable]



    def plot(
        self,
        left_input: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
        right_input: Optional[Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]] = None,
        start_time: Union[int, float] = 0,
        end_time: Union[int, float] = -1,
        time_unit: str = "s",
        label: bool = True,
        left_spacing: Union[int, float] = -1,
        right_spacing: Union[int, float] = -1,
        left_title: str = "",
        right_title: str = "",
        top_title: str = "",
        figsize: Tuple[Union[int, float], Union[int, float]] = (8, 5),
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
        if self.data is None:
            raise AttributeError("No csv read. Call .read_csv() before plotting.")
        plot(
            self.data,
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
