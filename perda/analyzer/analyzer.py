import io
import sys
from typing import List, Union

from plotly import graph_objects as go

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..plotting.data_instance_plotter import *
from ..plotting.plotting_constants import *
from ..plotting.subplots import data_instance_subplots
from ..units import Timescale, _from_seconds, _to_seconds
from ..utils.accel_calculator import *
from ..utils.data_summary import single_run_summary
from ..utils.diff import diff
from ..utils.frequency_analysis import analyze_frequency as _analyze_frequency
from ..utils.integrate import smoothed_filtered_integration
from ..utils.preprocessing import Preprocessing, apply_preprocessing
from ..utils.search import SearchResult, search
from .csv import *


class Analyzer:
    """Primary class for loading and analyzing car log data.

    After loading, all variables live in ``analyzer.data`` (a
    :class:`~perda.core_data_structures.SingleRunData`), which supports
    dictionary-like access returning :class:`~perda.core_data_structures.DataInstance`
    objects.

    Access by name or variable ID
    ------------------------------
    >>> di = aly.data["pcm.wheelSpeeds.frontRight"]   # by cpp_name
    >>> di = aly.data[42]                             # by variable ID

    Check membership
    ----------------
    >>> "pcm.wheelSpeeds.frontRight" in aly.data       # True / False

    Read raw arrays
    ---------------
    >>> di.timestamp_np   # NDArray[int64] — timestamps in the log's native unit
    >>> di.value_np       # NDArray[float64] — sample values

    Arithmetic between variables
    ----------------------------
    >>> avg_speed = (aly.data["pcm.wheelSpeeds.frontRight"] + aly.data["pcm.wheelSpeeds.frontLeft"]) / 2.0

    Trim to a time window (timestamps in the log's native unit)
    ------------------------------------------------------------
    >>> di_trimmed = aly.data["pcm.wheelSpeeds.frontRight"].trim(ts_start=10_000, ts_end=30_000)

    Find variables when you don't know the exact name
    --------------------------------------------------
    >>> results = aly.search("front wheel speed")   # prints + returns list[SearchResult]
    >>> di = aly.data[results[0].cpp_name]

    Enumerate all variables with summary stats
    ------------------------------------------
    >>> summaries = aly.variable_summary()          # list[VariableSummary], sorted by name
    >>> [v.cpp_name for v in summaries]
    """

    def __init__(
        self,
        filepath: str,
        ts_offset: int = 0,
        parsing_errors_limit: int = 100,
        corrections: list[Preprocessing] | None = None,
    ) -> None:
        """
        Initialize a new analyzer instance.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing CAN bus variables.
        ts_offset : int, optional
            Timestamp offset to apply to all data points. Default is 0.
        parsing_errors_limit : int, optional
            Maximum number of parsing errors before stopping. Default is 100.
        corrections : list[Preprocessing] | None, optional
            Ordered list of post-parse corrections to apply. Each correction
            is skipped with a warning if required variables are absent.
            Default is None (no corrections).

        Examples
        --------
        >>> from perda.utils import Preprocessing
        >>> aly = Analyzer("path/to/log.csv", corrections=[Preprocessing.NED_VELOCITY])
        >>> print(aly)  # lists all available variables
        """
        self.data: SingleRunData = parse_csv(
            filepath,
            ts_offset,
            parsing_errors_limit=parsing_errors_limit,
        )
        if corrections:
            self.data = apply_preprocessing(self.data, corrections)

    def __str__(self) -> str:
        """Return a summary of all variables in the loaded run data."""
        old_stdout = sys.stdout

        buffer = io.StringIO()
        sys.stdout = buffer

        single_run_summary(self.data)

        output = buffer.getvalue()
        buffer.close()

        sys.stdout = old_stdout

        return output

    def search(self, query: str, top_n: int = 10) -> list[SearchResult]:
        """
        Natural language search for available variables in the parsed data.

        Prints matching results to stdout and returns them for programmatic use.

        Parameters
        ----------
        query : str
            Free-text search query (e.g. "front wheel speed").
        top_n : int
            Maximum number of results to return and display (default 10).

        Returns
        -------
        list[SearchResult]
            Top matches in descending relevance order (at most ``top_n`` entries).
            Each entry has ``rank``, ``score``, ``var_id``, ``cpp_name``,
            and ``descript``.

        Examples
        --------
        >>> results = aly.search("front wheel speed")
        >>> results = aly.search("front wheel speed", top_n=5)
        >>> names = [r.cpp_name for r in results]
        """
        return search(self.data, query, top_n)

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
            Font configuration for plot elements.
        layout_config : LayoutConfig, optional
            Layout configuration for plot dimensions.
        vline_config : VLineConfig, optional
            Visual configuration for concat boundary lines.

        Examples
        --------
        >>> fig = aly.plot("pcm.wheelSpeeds.frontRight")
        >>> fig = aly.plot(["pcm.wheelSpeeds.frontRight", "pcm.wheelSpeeds.frontLeft"], title="Front Wheel Speeds")
        >>> fig = aly.plot("pcm.moc.motor.requestedTorque", "pcm.wheelSpeeds.frontRight", ts_start=10.0, ts_end=30.0)
        >>> # Plot a derived DataInstance (e.g. average of two signals)
        >>> avg_speed = (aly.data["pcm.wheelSpeeds.frontRight"] + aly.data["pcm.wheelSpeeds.frontLeft"]) / 2.0
        >>> fig = aly.plot(avg_speed)
        >>> fig.show()
        """
        # Normalize left input to List[DataInstance]
        var_1_norm = self._normalize_input(var_1)

        unit = self.data.timestamp_unit

        # Convert concat boundaries to seconds for the plotter
        vlines: List[float] | None = None
        if self.data.concat_boundaries:
            vlines = [_to_seconds(b, unit) for b in self.data.concat_boundaries]

        # Apply time range filter if specified (convert seconds → raw units for trim)
        if ts_start is not None or ts_end is not None:
            start_raw = _from_seconds(ts_start, unit) if ts_start is not None else None
            end_raw = _from_seconds(ts_end, unit) if ts_end is not None else None
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

    def subplots(
        self,
        rows: List[
            Union[
                str,
                int,
                DataInstance,
                List[Union[str, int, DataInstance]],
            ]
        ],
        title: str | None = None,
        row_y_labels: List[str | None] | None = None,
        ts_start: float | None = None,
        ts_end: float | None = None,
        show_legend: bool = True,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        subplot_config: SubplotConfig = DEFAULT_SUBPLOT_CONFIG,
    ) -> go.Figure:
        """
        Plot multiple variables as stacked subplots on a shared time axis.

        Each entry in ``rows`` becomes one subplot row. Pass a list of
        variables for a row to overlay multiple signals on the same panel,
        or a single variable for a dedicated panel.

        Parameters
        ----------
        rows : List[str | int | DataInstance | List[str | int | DataInstance]]
            One entry per subplot row (top to bottom). Each entry may be a
            single variable (name, ID, or DataInstance) or a list of variables
            to overlay on that row.
        title : str | None, optional
            Figure-level title. Default is None.
        row_y_labels : List[str | None] | None, optional
            Y-axis label for each row. ``None`` entries fall back to the
            DataInstance labels. Must match the length of ``rows`` when
            provided. Default is None.
        ts_start : float | None, optional
            Start of the time window in seconds. Data before this time is
            excluded from all rows. Default is None (no lower bound).
        ts_end : float | None, optional
            End of the time window in seconds. Data after this time is
            excluded from all rows. Default is None (no upper bound).
        show_legend : bool, optional
            Whether to show the figure legend. Default is True.
        font_config : FontConfig, optional
            Font sizes for plot elements.
        subplot_config : SubplotConfig, optional
            Row height, spacing, width, and style.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> fig = aly.subplots(["pcm.wheelSpeeds.frontRight", "pcm.moc.motor.requestedTorque"])
        >>> fig = aly.subplots(
        ...     rows=[
        ...         ["pcm.wheelSpeeds.frontRight", "pcm.wheelSpeeds.frontLeft"],
        ...         "pcm.moc.motor.requestedTorque",
        ...     ],
        ...     title="Run Overview",
        ...     row_y_labels=["Wheel Speed (mph)", "Torque (Nm)"],
        ...     ts_start=5.0,
        ...     ts_end=30.0,
        ... )
        >>> fig.show()
        """
        unit = self.data.timestamp_unit

        start_raw = _from_seconds(ts_start, unit) if ts_start is not None else None
        end_raw = _from_seconds(ts_end, unit) if ts_end is not None else None

        normalized_rows: List[List[DataInstance]] = []
        for row_entry in rows:
            row_dis = self._normalize_input(row_entry)
            if start_raw is not None or end_raw is not None:
                row_dis = [di.trim(start_raw, end_raw) for di in row_dis]
            normalized_rows.append(row_dis)

        return data_instance_subplots(
            rows=normalized_rows,
            title=title,
            row_y_labels=row_y_labels,
            show_legend=show_legend,
            font_config=font_config,
            subplot_config=subplot_config,
            timestamp_unit=unit,
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

        Examples
        --------
        >>> fig = aly.diff(server_data)
        >>> fig.show()
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
            Font sizes for plot elements.
        layout_config : LayoutConfig, optional
            Plot dimensions and margins.

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

    def get_accel_times(self) -> list[AccelSegmentResult]:
        """
        Intelligently detect and extract segments of the log where an
        acceleration run occurs, then compute acceleration times.

        Returns
        -------
        list[AccelSegmentResult]
            List of acceleration segment results.

        Examples
        --------
        >>> results = aly.get_accel_times()
        >>> for r in results:
        ...     print(r)
        """
        speed_obj = (
            self.data["pcm.wheelSpeeds.frontRight"]
            + self.data["pcm.wheelSpeeds.frontLeft"]
        ) / 2.0
        signal_obj = detect_accel_event(
            torque_obj=self.data["pcm.moc.motor.requestedTorque"], speed_obj=speed_obj
        )

        time_arr, _, distance = smoothed_filtered_integration(
            data=speed_obj, source_time_unit=self.data.timestamp_unit
        )
        distance_obj = DataInstance(
            timestamp_np=time_arr,
            value_np=distance,
            label="Distance",
        )

        return compute_accel_results(
            signal_obj=signal_obj,
            distance_obj=distance_obj,
            source_time_unit=self.data.timestamp_unit,
        )
