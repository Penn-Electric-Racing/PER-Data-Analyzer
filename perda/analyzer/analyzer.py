import io
import sys
from collections.abc import Sequence

from plotly import graph_objects as go

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..plotting.data_instance_plotter import plot_dual_axis, plot_single_axis
from ..plotting.defaults import (
    HISTOGRAM_BINS,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    SUBPLOT_VERTICAL_SPACING,
)
from ..plotting.subplots import data_instance_subplots
from ..units import from_seconds, to_seconds
from ..utils.accel_calculator import *
from ..utils.data_summary import single_run_summary
from ..utils.diff import DIFF_BUCKET_SIZE_S, diff
from ..utils.frequency_analysis import analyze_frequency as _analyze_frequency
from ..utils.integrate import smoothed_filtered_integration
from ..utils.preprocessing import PreprocessingStep, apply_preprocessing
from ..utils.search import SearchResult, search
from .csv import *

VariableInput = str | int | DataInstance
VariableInputs = VariableInput | Sequence[VariableInput]


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

    Summary stats for one variable
    ------------------------------
    >>> from perda.utils import data_instance_summary
    >>> summary = data_instance_summary(
    ...     aly.data["pcm.wheelSpeeds.frontRight"], source_time_unit=aly.data.timestamp_unit
    ... )
    >>> print(summary)          # formatted block
    >>> summary.max_value       # or read the fields directly
    """

    def __init__(
        self,
        filepath: str,
        ts_offset: int = 0,
        parsing_errors_limit: int = 100,
        verbose: int = 1,
        preprocessing: list[PreprocessingStep] | None = None,
        semantic_search: bool = False,
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
        preprocessing : list[PreprocessingStep] | None, optional
            Ordered list of post-parse preprocessing steps to apply. Each step
            is a ``SingleRunData -> SingleRunData`` callable. Steps are skipped
            with a warning if required variables are absent. Default is None.
        semantic_search : bool, optional
            Whether to vectorize variable descriptions so that ``search`` ranks by
            meaning rather than keywords. Requires the ``semantic`` extra and adds
            noticeable time to loading. Falls back to keyword search if the
            dependencies or model are unavailable. Default is False.

        Examples
        --------
        >>> from perda.utils.preprocessing import correct_motor_data, correct_steering_angle
        >>> aly = Analyzer("path/to/log.csv", preprocessing=[correct_motor_data])
        >>> aly = Analyzer("path/to/log.csv", preprocessing=[
        ...     correct_motor_data,
        ...     correct_steering_angle(calibration=((1.5, -90.0), (3.0, 0.0), (4.5, 90.0))),
        ... ])
        >>> print(aly)  # lists all available variables
        """
        self.data: SingleRunData = parse_csv(
            filepath,
            ts_offset,
            parsing_errors_limit=parsing_errors_limit,
            verbose=verbose,
            build_search_index=semantic_search,
        )
        if preprocessing:
            self.data = apply_preprocessing(self.data, preprocessing)

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

        Notes
        -----
        Ranks by meaning when the Analyzer was created with ``semantic_search=True``,
        and by fuzzy keyword matching otherwise.

        Examples
        --------
        >>> results = aly.search("front wheel speed")
        >>> results = aly.search("front wheel speed", top_n=5)
        >>> names = [r.cpp_name for r in results]
        """
        return search(self.data, query, top_n)

    def plot(
        self,
        var_1: VariableInputs,
        var_2: VariableInputs | None = None,
        ts_start: float | None = None,
        ts_end: float | None = None,
        title: str | None = None,
        y_label_1: str | None = None,
        y_label_2: str | None = None,
        show_legend: bool = True,
        max_points: int | None = None,
        log_x: bool = False,
        log_y: bool = False,
        width: int = PLOT_WIDTH,
        height: int = PLOT_HEIGHT,
    ) -> go.Figure:
        """
        Plot variables from the log against time on an interactive Plotly figure.

        Pass one variable or a list for a single y-axis. Also pass ``var_2`` to put a second group of variables on an independent right-hand y-axis. Concat boundaries (if any) are drawn as dashed vertical lines.

        Parameters
        ----------
        var_1 : str | int | DataInstance | Sequence[str | int | DataInstance]
            Variable(s) for the left (or only) y-axis. Each item can be a variable name (``"pcm.wheelSpeeds.frontRight"``), a variable ID (``42``), or a DataInstance (e.g. a derived signal).
        var_2 : str | int | DataInstance | Sequence[str | int | DataInstance] | None, optional
            Variable(s) for a second, right-hand y-axis, drawn dashed. Same accepted types as ``var_1``. Default is None (single y-axis).
        ts_start : float | None, optional
            Start of the time window in seconds on the log's clock. Earlier data is excluded. Default is None (from the start of the log).
        ts_end : float | None, optional
            End of the time window in seconds on the log's clock. Later data is excluded. Default is None (to the end of the log).
        title : str | None, optional
            Title shown above the plot. Default is None (no title).
        y_label_1 : str | None, optional
            Left (or only) y-axis label, e.g. ``"Speed (mph)"``. Default is None (no label).
        y_label_2 : str | None, optional
            Right y-axis label. Only used with ``var_2``. Default is None (no label).
        show_legend : bool, optional
            Whether to show the legend listing each variable. Default is True.
        max_points : int | None, optional
            Maximum number of points drawn per variable. Longer signals are evenly downsampled (every Nth point), which speeds up rendering of large logs. Default is None (draw every point).
        log_x : bool, optional
            Use a logarithmic time (x) axis. Default is False (linear).
        log_y : bool, optional
            Use logarithmic y-axes (both left and right when ``var_2`` is given). Default is False (linear).
        width : int, optional
            Figure width in pixels. Default is PLOT_WIDTH (1200).
        height : int, optional
            Figure height in pixels. Default is PLOT_HEIGHT (700).

        Returns
        -------
        go.Figure
            Plotly figure. Change anything not exposed here (fonts, colors, margins, axis ranges, ...) with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``.

        Notes
        -----
        Unlike the lower-level plotters, this reads the log's timestamp unit itself.
        ``ts_start`` and ``ts_end`` are seconds on the log's clock, i.e. raw timestamps
        converted to seconds, not seconds since the start of the log.

        Examples
        --------
        >>> fig = aly.plot("pcm.wheelSpeeds.frontRight")
        >>> fig = aly.plot(["pcm.wheelSpeeds.frontRight", "pcm.wheelSpeeds.frontLeft"], title="Front Wheel Speeds")
        >>> fig = aly.plot("pcm.moc.motor.requestedTorque", "pcm.wheelSpeeds.frontRight", ts_start=10.0, ts_end=30.0)
        >>> # Plot a derived DataInstance (e.g. average of two signals)
        >>> avg_speed = (aly.data["pcm.wheelSpeeds.frontRight"] + aly.data["pcm.wheelSpeeds.frontLeft"]) / 2.0
        >>> fig = aly.plot(avg_speed)
        >>> # Log-scaled y-axis, then a custom font size
        >>> fig = aly.plot("bms.pack.current", log_y=True)
        >>> fig.update_layout(font_size=16)
        """
        var_1_norm = self._normalize_input(var_1)
        var_2_norm = self._normalize_input(var_2) if var_2 is not None else None

        unit = self.data.timestamp_unit

        # Convert concat boundaries to seconds for the plotter
        vlines: list[float] | None = None
        if self.data.concat_boundaries:
            vlines = [to_seconds(float(b), unit) for b in self.data.concat_boundaries]

        # Apply time range filter if specified (convert seconds -> raw units for trim)
        if ts_start is not None or ts_end is not None:
            start_raw = from_seconds(ts_start, unit) if ts_start is not None else None
            end_raw = from_seconds(ts_end, unit) if ts_end is not None else None
            var_1_norm = [di.trim(start_raw, end_raw) for di in var_1_norm]
            if var_2_norm is not None:
                var_2_norm = [di.trim(start_raw, end_raw) for di in var_2_norm]

        if var_2_norm is not None:
            return plot_dual_axis(
                left_data_instances=var_1_norm,
                right_data_instances=var_2_norm,
                title=title,
                left_y_axis_title=y_label_1,
                right_y_axis_title=y_label_2,
                show_legend=show_legend,
                timestamp_unit=self.data.timestamp_unit,
                vlines=vlines,
                max_points=max_points,
                log_x=log_x,
                log_y=log_y,
                width=width,
                height=height,
            )
        else:
            return plot_single_axis(
                data_instances=var_1_norm,
                title=title,
                y_axis_title=y_label_1,
                show_legend=show_legend,
                timestamp_unit=self.data.timestamp_unit,
                vlines=vlines,
                max_points=max_points,
                log_x=log_x,
                log_y=log_y,
                width=width,
                height=height,
            )

    def subplots(
        self,
        rows: Sequence[VariableInputs],
        title: str | None = None,
        row_y_labels: list[str | None] | None = None,
        ts_start: float | None = None,
        ts_end: float | None = None,
        show_legend: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        width: int = PLOT_WIDTH,
        height: int | None = None,
        vertical_spacing: float = SUBPLOT_VERTICAL_SPACING,
    ) -> go.Figure:
        """
        Plot variables as vertically stacked subplots sharing one time axis.

        Each entry in ``rows`` becomes one subplot row. Pass a list of variables for a row to overlay them on the same panel, or a single variable for a dedicated panel. Zooming the time axis zooms every row together.

        Parameters
        ----------
        rows : Sequence[str | int | DataInstance | Sequence[str | int | DataInstance]]
            One entry per subplot row, top to bottom. Each entry is a single variable (name, ID, or DataInstance) or a list of variables to overlay on that row.
        title : str | None, optional
            Title shown above the whole figure. Default is None (no title).
        row_y_labels : list[str | None] | None, optional
            Y-axis label for each row, in the same order as ``rows``. A ``None`` entry falls back to the variable names in that row. Default is None (every row uses its variable names).
        ts_start : float | None, optional
            Start of the time window in seconds on the log's clock. Earlier data is excluded from all rows. Default is None (from the start of the log).
        ts_end : float | None, optional
            End of the time window in seconds on the log's clock. Later data is excluded from all rows. Default is None (to the end of the log).
        show_legend : bool, optional
            Whether to show the legend listing each variable. Default is True.
        log_x : bool, optional
            Use a logarithmic time (x) axis. Default is False (linear).
        log_y : bool, optional
            Use logarithmic y-axes on every row. Default is False (linear).
        width : int, optional
            Figure width in pixels. Default is PLOT_WIDTH (1200).
        height : int | None, optional
            Figure height in pixels. Default is None, which gives SUBPLOT_HEIGHT (350) pixels per row.
        vertical_spacing : float, optional
            Gap between rows as a fraction of the figure height (0 to 1). Default is SUBPLOT_VERTICAL_SPACING (0.12).

        Returns
        -------
        go.Figure
            Plotly figure. Change anything not exposed here with ``fig.update_layout``, ``fig.update_xaxes``, ``fig.update_yaxes`` or ``fig.update_traces`` before calling ``fig.show()``. Target one row with ``row=i, col=1``, e.g. ``fig.update_yaxes(type="log", row=2, col=1)``.

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
        """
        unit = self.data.timestamp_unit

        start_raw = from_seconds(ts_start, unit) if ts_start is not None else None
        end_raw = from_seconds(ts_end, unit) if ts_end is not None else None

        normalized_rows: list[list[DataInstance]] = []
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
            log_x=log_x,
            log_y=log_y,
            width=width,
            height=height,
            vertical_spacing=vertical_spacing,
            timestamp_unit=unit,
        )

    def diff(
        self,
        server_data: SingleRunData,
        timestamp_tolerance_s: float = 0.002,
        diff_rtol: float = 1e-3,
        diff_atol: float = 1e-3,
        bucket_size_s: float = DIFF_BUCKET_SIZE_S,
        log_y: bool = False,
        width: int = PLOT_WIDTH,
        height: int = PLOT_HEIGHT,
    ) -> go.Figure:
        """
        Compute the differences between the current data (assumed to be from RPI) and server data.

        Parameters
        ----------
        server_data : SingleRunData
            The server data to compare against.
        timestamp_tolerance_s : float, optional
            Timestamp tolerance in seconds used to match points between streams.
            Defaults to 0.002 (2 ms).
        diff_rtol : float, optional
            Relative tolerance for value comparison (numpy.isclose).
        diff_atol : float, optional
            Absolute tolerance for value comparison (numpy.isclose).
        bucket_size_s : float, optional
            Width of each time bucket (bar) in the plot, in seconds. Default is DIFF_BUCKET_SIZE_S (1.0).
        log_y : bool, optional
            Use a logarithmic count (y) axis. Default is False (linear).
        width : int, optional
            Figure width in pixels. Default is PLOT_WIDTH (1200).
        height : int, optional
            Figure height in pixels. Default is PLOT_HEIGHT (700).

        Returns
        -------
        go.Figure
            Bar chart of diff counts over time between the two streams. Customize further with ``fig.update_layout`` before calling ``fig.show()``.

        Examples
        --------
        >>> fig = aly.diff(server_data)
        """
        return diff(
            self.data,
            server_data,
            timestamp_tolerance_s=timestamp_tolerance_s,
            diff_rtol=diff_rtol,
            diff_atol=diff_atol,
            bucket_size_s=bucket_size_s,
            log_y=log_y,
            width=width,
            height=height,
        )

    def analyze_frequency(
        self,
        var: str | int,
        expected_frequency_hz: float | None = None,
        gap_threshold_multiplier: float = 2.0,
        histogram_bins: int = HISTOGRAM_BINS,
        width: int = PLOT_WIDTH,
        height: int = PLOT_HEIGHT,
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
        histogram_bins : int, optional
            Maximum number of bins in the frequency histogram. Default is HISTOGRAM_BINS (80).
        width : int, optional
            Figure width in pixels. Default is PLOT_WIDTH (1200).
        height : int, optional
            Figure height in pixels. Default is PLOT_HEIGHT (700).

        Returns
        -------
        go.Figure
            Plotly figure with frequency diagnostics. Customize further with ``fig.update_layout`` before calling ``fig.show()``.

        Examples
        --------
        >>> fig = aly.analyze_frequency("ams.stack.thermistors.temperature[38]", expected_frequency_hz=100)
        """
        di = self.data[var]
        return _analyze_frequency(
            di,
            expected_frequency_hz=expected_frequency_hz,
            source_time_unit=self.data.timestamp_unit,
            gap_threshold_multiplier=gap_threshold_multiplier,
            histogram_bins=histogram_bins,
            width=width,
            height=height,
        )

    def _normalize_input(
        self,
        input_data: VariableInputs,
    ) -> list[DataInstance]:
        """
        Normalize various input types to a list of DataInstances.
        """
        if isinstance(input_data, DataInstance):
            return [input_data]
        elif isinstance(input_data, (str, int)):
            return [self.data[input_data]]
        else:
            return [
                item if isinstance(item, DataInstance) else self.data[item]
                for item in input_data
            ]

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
