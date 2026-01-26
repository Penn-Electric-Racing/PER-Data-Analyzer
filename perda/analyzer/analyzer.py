from typing import List, Union

from ..plotting import plot_dual_axis, plot_single_axis
from ..plotting.plotting_constants import *
from .models import DataInstance, SingleRunData

from plotly import graph_objects as go


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

    def plot(
        self,
        var_1: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
        var_2: (
            Union[str, int, DataInstance, List[Union[str, int, DataInstance]]] | None
        ) = None,
        title: str | None = None,
        y_label_1: str | None = None,
        y_label_2: str | None = None,
        show_legend: bool = True,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    ) -> go.Figure:
        """
        Display variables from the parsed data on an interactive Plotly plot.

        Parameters
        ----------
        var_1 : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
            Variable(s) to plot on the left y-axis. Can be variable name(s), CAN ID(s), or DataInstance(s)
        var_2 : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]] | None, optional
            Optional variable(s) to plot on the right y-axis. Can be variable name(s), CAN ID(s), or DataInstance(s).
        title : str | None, optional
        y_label_1 : str | None, optional
            Label for left y-axis (or only y-axis if no right input).
        y_label_2 : str | None, optional
            Label for right y-axis.
        show_legend : bool, optional
            Whether to show plot legends. Default is True
        font_config : FontConfig, optional
            Font configuration for plot elements. Default is DEFAULT_FONT_CONFIG
        layout_config : LayoutConfig, optional
            Layout configuration for plot dimensions. Default is DEFAULT_LAYOUT_CONFIG

        Notes
        -----
        This method uses Plotly for interactive plotting with zoom, pan, and hover capabilities.
        Time filtering can be done interactively in the plot.
        """
        # Normalize left input to List[DataInstance]
        var_1 = self._normalize_input(var_1)

        if var_2 is not None:
            # Normalize right input to List[DataInstance]
            var_2 = self._normalize_input(var_2)

            return plot_dual_axis(
                left_data_instances=var_1,
                right_data_instances=var_2,
                title=title,
                left_y_axis_title=y_label_1,
                right_y_axis_title=y_label_2,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
            )
        else:
            return plot_single_axis(
                data_instances=var_1,
                title=title,
                y_axis_title=y_label_1,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
            )

    def _normalize_input(
        self,
        input_data: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
    ) -> List[DataInstance]:
        """
        Normalize various input types to a list of DataInstances.
        """
        if isinstance(input_data, list):
            return [self.data[item] for item in input_data]
        else:
            return [self.data[input_data]]
