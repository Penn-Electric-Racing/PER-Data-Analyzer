import io
import sys
from typing import List, Union

from plotly import graph_objects as go

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..plotting.data_instance_plotter import *
from ..plotting.plotting_constants import *
from ..units import Timescale, mph_seconds_to_meters
from ..utils.accel_calculator import AccelSegmentResult, compute_accel_results
from ..utils.accel_calculator import detect_accel_event as _detect_accel_event
from ..utils.data_summary import single_run_summary
from ..utils.diff import diff
from ..utils.frequency_analysis import analyze_frequency as _analyze_frequency
from ..utils.integrate import smoothed_filtered_integration
from ..utils.search import search
from .csv import *


class Analyzer:
    def __init__(
        self,
        filepath: str,
        ts_offset: int = 0,
        parsing_errors_limit: int = 100,
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
        )

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

    def detect_accel_event(
        self,
        torque_var: Union[str, int],
        speed_var: Union[str, int],
        torque_threshold: float = 100,
        speed_threshold: float = 0.5,
    ) -> DataInstance:
        """Detect acceleration events based on torque and speed thresholds.

        An event is active when torque exceeds `torque_threshold` and speed exceeds
        `speed_threshold`, and ends when speed drops back to or below `speed_threshold`.

        Parameters
        ----------
        torque_var : Union[str, int]
            Variable name or ID for motor torque.
        speed_var : Union[str, int]
            Variable name or ID for wheel speed. The output is aligned to these timestamps.
        torque_threshold : float, optional
            Minimum torque (in Nm) required to trigger an acceleration event. Default is 100.
        speed_threshold : float, optional
            Speed value used as the trigger floor and reset condition. Default is 0.5.

        Returns
        -------
        DataInstance
            Binary signal (0.0 or 1.0) on `speed_var` timestamps, labeled "Accel Event",
            where 1.0 indicates an active acceleration event.
        """
        return _detect_accel_event(
            self.data[torque_var],
            self.data[speed_var],
            torque_threshold=torque_threshold,
            speed_threshold=speed_threshold,
        )

    def get_accel_triggers(
        self,
        torque_var: Union[str, int] = "pcm.moc.motor.requestedTorque",
        speed_var: Union[str, int] = "pcm.wheelSpeeds.frontRight",
        target_dist: float = 75,
        timescale: float = 1000000,
        torque_threshold: float = 100,
        speed_threshold: float = 0.5,
        filter_window_size: int = 10,
        n_sigmas: float = 3,
        smoothing_window_len: int = 11,
        smoothing_poly_order: int = 2,
    ) -> list:
        """Find all acceleration events in a run and compute time-to-distance for each.

        Detects acceleration events, integrates wheel speed to compute cumulative distance,
        and records the elapsed time from each event start until `target_dist` meters are covered.

        Parameters
        ----------
        torque_var : Union[str, int], optional
            Variable name or ID for motor torque. Default is "pcm.moc.motor.requestedTorque".
        speed_var : Union[str, int], optional
            Variable name or ID for wheel speed. Default is "pcm.wheelSpeeds.frontRight".
        target_dist : float, optional
            Target distance in meters to measure time to. Default is 75.
        timescale : float, optional
            Factor to convert raw timestamps to seconds (e.g. 1000000 for microseconds). Default is 1000000.
        torque_threshold : float, optional
            Minimum torque (in Nm) to trigger an acceleration event. Default is 100.
        speed_threshold : float, optional
            Speed value used as the trigger floor and reset condition. Default is 0.5.
        filter_window_size : int, optional
            Window size for outlier filtering in cumulative integration. Default is 10.
        n_sigmas : float, optional
            Number of standard deviations for outlier detection in cumulative integration. Default is 3.
        smoothing_window_len : int, optional
            Window length for Savitzky-Golay smoothing in cumulative integration. Default is 11.
        smoothing_poly_order : int, optional
            Polynomial order for Savitzky-Golay smoothing in cumulative integration. Default is 2.

        Returns
        -------
        list[AccelSegmentResult]
            One result per qualifying segment.
        """
        speed_obj = (
            self.data["pcm.wheelSpeeds.frontRight"]
            + self.data["pcm.wheelSpeeds.frontLeft"]
        ) / 2.0
        signal_obj = self.detect_accel_event(
            torque_var=torque_var,
            speed_var=speed_var,
            torque_threshold=torque_threshold,
            speed_threshold=speed_threshold,
        )

        time_arr, _, distance = smoothed_filtered_integration(
            speed_obj,
            source_time_unit=self.data.timestamp_unit,
            target_time_unit=Timescale.S,
            filter_window_size=filter_window_size,
            n_sigmas=n_sigmas,
            smoothing_window_len=smoothing_window_len,
            smoothing_poly_order=smoothing_poly_order,
        )
        distance_obj = DataInstance(
            timestamp_np=time_arr,
            value_np=mph_seconds_to_meters(distance),
            label="Distance",
        )

        results = compute_accel_results(
            signal_obj,
            distance_obj,
            target_dist=target_dist,
            source_time_unit=self.data.timestamp_unit,
            target_time_unit=Timescale.S,
        )
        for e in results:
            print(e)

        return results

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
