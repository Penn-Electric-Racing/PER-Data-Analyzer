import io
import sys
from typing import List, Union

from plotly import graph_objects as go

from ..plotting.data_instance_plotter import *
from ..plotting.plotting_constants import *
from ..utils.concat import concat_single_run_data
from ..utils.data_summary import single_run_summary
from ..utils.diff import diff
from ..utils.frequency_analysis import analyze_frequency as _analyze_frequency
from ..utils.search import search
from ..utils.types import Timescale
from .csv import *
from .data_instance import DataInstance
from .single_run_data import SingleRunData


class Analyzer:
    def __init__(
        self,
        filepath: str,
        ts_offset: int = 0,
        parsing_errors_limit: int = 100,
        parse_unit: Timescale | str | None = None,
    ) -> None:
        """
        Initialize a new analyzer instance.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing CAN bus variables
        ts_offset : int, optional
            Timestamp offset to apply to all data points. Default is 0
        parsing_errors_limit : int, optional
            Maximum number of parsing errors before stopping. Default is 100
        parse_unit : Timescale | str | None, optional
            Logging timestamp unit for parsing. If None, auto-detect from header:
            lines ending with "v2.0" use us, otherwise ms.
        """
        self.data: SingleRunData = parse_csv(
            filepath,
            ts_offset,
            parsing_errors_limit=parsing_errors_limit,
            parse_unit=parse_unit,
        )

    @staticmethod
    def concat(
        first: "Analyzer",
        second: "Analyzer",
        gap: int = 1,
    ) -> "Analyzer":
        """
        Concatenate two Analyzers sequentially in time.

        Variables are matched by cpp_name. Unmatched variables are kept with
        data from only the run that has them. If the two runs use different
        timestamp units the ms run is upscaled to us.

        Parameters
        ----------
        first : Analyzer
            First analyzer (earlier in time)
        second : Analyzer
            Second analyzer (appended after first)
        gap : int
            Gap in timestamp units between the two runs. Default is 1.

        Returns
        -------
        Analyzer
            New Analyzer containing the concatenated data

        Examples
        --------
        >>> merged = Analyzer.concat(aly1, aly2)
        >>> merged.plot("ams.pack.voltage")
        """
        merged = object.__new__(Analyzer)
        merged.data = concat_single_run_data(first.data, second.data, gap=gap)
        return merged

    def __str__(self) -> str:
        old_stdout = sys.stdout

        buffer = io.StringIO()
        sys.stdout = buffer

        single_run_summary(self.data)

        output = buffer.getvalue()
        buffer.close()

        sys.stdout = old_stdout

        return output

    def search(self, query: str) -> None:
        """
        Natural language search for available variables in the parsed data.

        Parameters
        ----------
        query : str
            Search query
        """
        search(self.data, query)

    def plot(
        self,
        var_1: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
        var_2: (
            Union[str, int, DataInstance, List[Union[str, int, DataInstance]]] | None
        ) = None,
        ts_start: float | None = None,
        ts_end: float | None = None,
        title: str | None = None,
        y_label_1: str | None = None,
        y_label_2: str | None = None,
        show_legend: bool = True,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
        vline_config: VLineConfig = DEFAULT_VLINE_CONFIG,
    ) -> go.Figure:
        """
        Display variables from the parsed data on an interactive Plotly plot.

        Concat boundaries (if any) are automatically shown as vertical lines.

        Parameters
        ----------
        var_1 : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]]
            Variable(s) to plot on the left y-axis. Can be variable name(s), variable ID(s), or DataInstance(s)
        var_2 : Union[str, int, DataInstance, List[Union[str, int, DataInstance]]] | None, optional
            Optional variable(s) to plot on the right y-axis. Can be variable name(s), variable ID(s), or DataInstance(s).
        ts_start : float | None, optional
            Start of the time window in seconds. Data points before this time are excluded. Default is None (no lower bound).
        ts_end : float | None, optional
            End of the time window in seconds. Data points after this time are excluded. Default is None (no upper bound).
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
        vline_config : VLineConfig, optional
            Visual configuration for concat boundary lines. Default is DEFAULT_VLINE_CONFIG.
        """
        # Normalize left input to List[DataInstance]
        var_1_norm = self._normalize_input(var_1)

        # Divisor to convert raw timestamps to seconds (used for vlines and time range filter)
        unit = self.data.timestamp_unit
        divisor = 1e6 if unit == Timescale.US else 1e3 if unit == Timescale.MS else 1.0

        # Convert concat boundaries to seconds for the plotter
        vlines: List[float] | None = None
        if self.data.concat_boundaries:
            vlines = [b / divisor for b in self.data.concat_boundaries]

        # Apply time range filter if specified (convert seconds → raw units for trim)
        if ts_start is not None or ts_end is not None:
            start_raw = ts_start * divisor if ts_start is not None else None
            end_raw = ts_end * divisor if ts_end is not None else None
            var_1_norm = [di.trim(start_raw, end_raw) for di in var_1_norm]

        if var_2 is not None:
            # Normalize right input to List[DataInstance]
            var_2_norm = self._normalize_input(var_2)
            if ts_start is not None or ts_end is not None:
                var_2_norm = [di.trim(start_raw, end_raw) for di in var_2_norm]

            return plot_dual_axis(
                left_data_instances=var_1_norm,
                right_data_instances=var_2_norm,
                title=title,
                left_y_axis_title=y_label_1,
                right_y_axis_title=y_label_2,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
                timestamp_unit=self.data.timestamp_unit,
                vlines=vlines,
                vline_config=vline_config,
            )
        else:
            return plot_single_axis(
                data_instances=var_1_norm,
                title=title,
                y_axis_title=y_label_1,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
                timestamp_unit=self.data.timestamp_unit,
                vlines=vlines,
                vline_config=vline_config,
            )

    def diff(
        self,
        server_data: SingleRunData,
        timestamp_tolerance_ms: int = 2,
        diff_rtol: float = 1e-3,
        diff_atol: float = 1e-3,
    ) -> go.Figure:
        """
        Compute the differences between the current data (assumed to be from RPI) and server data.

        Parameters
        ----------
        server_data : SingleRunData
            The server data to compare against.
        timestamp_tolerance_ms : int, optional
            Timestamp tolerance used to match points between streams.
        diff_rtol : float, optional
            Relative tolerance for value comparison (numpy.isclose).
        diff_atol : float, optional
            Absolute tolerance for value comparison (numpy.isclose).
        """
        return diff(
            self.data,
            server_data,
            timestamp_tolerance_ms=timestamp_tolerance_ms,
            diff_rtol=diff_rtol,
            diff_atol=diff_atol,
        )

    def analyze_frequency(
        self,
        var: Union[str, int],
        expected_frequency_hz: float | None = None,
        gap_threshold_multiplier: float = 2.0,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
        plot_config: ScatterHistogramPlotConfig = DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG,
    ) -> go.Figure:
        """
        Analyse the sampling frequency of a variable and return a diagnostic figure.

        Prints a summary to stdout and returns a Plotly figure with two subplots:
        instantaneous frequency over time and an inter-sample interval histogram.

        Parameters
        ----------
        var : Union[str, int]
            Variable name or ID to look up in the parsed data.
        expected_frequency_hz : float | None, optional
            Nominal expected sampling frequency in Hz for error and gap diagnostics.
            Default is None.
        gap_threshold_multiplier : float, optional
            Intervals exceeding this multiple of the expected (or median) interval
            are flagged as gaps. Default is 2.0.
        font_config : FontConfig, optional
            Font sizes for plot elements. Default is DEFAULT_FONT_CONFIG.
        layout_config : LayoutConfig, optional
            Plot dimensions and margins. Default is DEFAULT_LAYOUT_CONFIG.

        Returns
        -------
        go.Figure
            Plotly figure with frequency diagnostics.

        Examples
        --------
        >>> fig = aly.analyze_frequency("ams.stack.thermistors.temperature[38]", expected_frequency_hz=100)
        >>> fig.show()
        """
        di = self.data[var]
        return _analyze_frequency(
            di,
            expected_frequency_hz=expected_frequency_hz,
            source_time_unit=self.data.timestamp_unit,
            gap_threshold_multiplier=gap_threshold_multiplier,
            font_config=font_config,
            layout_config=layout_config,
            plot_config=plot_config,
        )

    def _normalize_input(
        self,
        input_data: Union[str, int, DataInstance, List[Union[str, int, DataInstance]]],
    ) -> List[DataInstance]:
        """
        Normalize various input types to a list of DataInstances.
        """
        if isinstance(input_data, DataInstance):
            return [input_data]
        elif isinstance(input_data, list):
            return [
                item if isinstance(item, DataInstance) else self.data[item]
                for item in input_data
            ]
        else:
            return [self.data[input_data]]
