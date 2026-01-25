from typing import List, Optional, Tuple, Union

from ..plotting.data_plotter import plot
from .models import DataInstance, SingleRunData


class Analyzer:
    def __init__(self, data: SingleRunData) -> None:
        """
        Initialize a new analyzer instance.

        Parameters
        ----------
        data : SingleRunData
            Parsed CSV data containing CAN bus variables
        """
        self.data: SingleRunData = data

    def get_data(self, variable: str) -> DataInstance:
        """
        Get DataInstance by CAN ID or name.

        Parameters
        ----------
        variable : str
            Name or CAN ID of the variable to get

        Returns
        -------
        DataInstance
            DataInstance object containing timestamp and value arrays
        """
        if self.data is None:
            raise AttributeError("No csv read. Call .read_csv() before getting data.")
        return self.data[variable]

    def plot(
        self,
        left_input: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
        right_input: Optional[
            Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
        ] = None,
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
        Display up to two variables from the parsed data on a plot.

        Parameters
        ----------
        left_input : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
            Variable(s) to plot on the left y-axis. Can be variable name(s), CAN ID(s), or DataInstance(s)
        right_input : Optional[Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]], optional
            Variable(s) to plot on the right y-axis. Can be variable name(s), CAN ID(s), or DataInstance(s).
            If None, only uses left y-axis. Default is None
        start_time : Union[int, float], optional
            Start time for plotting. Default is 0
        end_time : Union[int, float], optional
            End time for plotting. -1 means until the end. Default is -1
        time_unit : str, optional
            Unit for time axis, either "ms" or "s".
            Affects both display and input interpretation. Default is "s"
        label : bool, optional
            Whether to show plot legends. Default is True
        left_spacing : Union[int, float], optional
            Spacing between major ticks on left y-axis.
            -1 means auto spacing. Default is -1
        right_spacing : Union[int, float], optional
            Spacing between major ticks on right y-axis.
            -1 means auto spacing. Default is -1
        left_title : str, optional
            Label for left y-axis. Default is ""
        right_title : str, optional
            Label for right y-axis. Default is ""
        top_title : str, optional
            Title for the entire plot. Default is ""
        figsize : Tuple[Union[int, float], Union[int, float]], optional
            Figure size in inches (width, height). Default is (8, 5)

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If no CSV has been read before calling this method

        Notes
        -----
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
