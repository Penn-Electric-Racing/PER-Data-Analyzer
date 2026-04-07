"""
perda.live.live_analyzer
------------------------
Live data analysis using the CDP IPC server, mirroring the Analyzer interface.

Example usage:
    # Connect to local server
    live = LiveAnalyzer.local()

    # Connect to the Penn Electric Racing dataserver
    live = LiveAnalyzer.dataserver()

    # Connect to an arbitrary IP (developers)
    live = LiveAnalyzer.remote("192.168.1.50")

    # Plot the last 30 seconds — no type needed
    live.plot("bms.board.glvTemp", time_secs=30)

    # Dual axis, multiple signals
    live.plot(["bms.board.glvTemp", "bms.board.hvTemp"], var_2="motor.temp")

    # Get/set still require a ValueType since the server type-checks these
    temp = live.get("bms.board.glvTemp", ValueType.FLOAT)
    live.set("motor.target", 100.0, ValueType.NUMERIC)
"""

from typing import List, Union

from plotly import graph_objects as go

from ..analyzer.data_instance import DataInstance
from ..plotting.data_instance_plotter import plot_dual_axis, plot_single_axis
from ..plotting.plotting_constants import (
    DEFAULT_FONT_CONFIG,
    DEFAULT_LAYOUT_CONFIG,
    DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG,
    FontConfig,
    LayoutConfig,
    ScatterHistogramPlotConfig,
)
from ..utils.frequency_analysis import analyze_frequency as _analyze_frequency
from ..utils.types import Timescale
from .cdp_client import CDPClient, CDPException, ValueType

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Range ops: bare string or list of strings — no ValueType needed
RangeSpec = Union[str, List[str]]

_DATASERVER_IP = "198.74.62.28"
_DEFAULT_PORT = 5001
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_RANGE_TIMEOUT = 2.0
_DEFAULT_TIME_SECS = 30
_TIMESTAMP_UNIT = Timescale.MS


def _to_list(specs: RangeSpec) -> List[str]:
    if isinstance(specs, list):
        return specs
    return [specs]


class LiveAnalyzer:
    """
    Live data analyzer that pulls time-series data from the CDP IPC server
    and exposes a plotting interface analogous to ``perda.analyzer.Analyzer``.

    Do not instantiate directly — use the named constructors:
    ``LiveAnalyzer.local()``, ``LiveAnalyzer.dataserver()``, or
    ``LiveAnalyzer.remote(host)``.
    """

    def __init__(self, host: str, port: int, timeout: float, range_timeout: float):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._range_timeout = range_timeout
        # Verify connectivity eagerly so callers get a clear error up front
        self._check_connection()

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def local(
        cls,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        range_timeout: float = _DEFAULT_RANGE_TIMEOUT,
    ) -> "LiveAnalyzer":
        """Connect to a CDP server running on this machine."""
        return cls("127.0.0.1", port, timeout, range_timeout)

    @classmethod
    def dataserver(
        cls,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        range_timeout: float = _DEFAULT_RANGE_TIMEOUT,
    ) -> "LiveAnalyzer":
        """Connect to the Penn Electric Racing dataserver."""
        return cls(_DATASERVER_IP, port, timeout, range_timeout)

    @classmethod
    def remote(
        cls,
        host: str,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        range_timeout: float = _DEFAULT_RANGE_TIMEOUT,
    ) -> "LiveAnalyzer":
        """Connect to an arbitrary CDP server (for developers)."""
        return cls(host, port, timeout, range_timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        var: str,
        time_secs: int = _DEFAULT_TIME_SECS,
    ) -> DataInstance:
        """
        Fetch a time-series range for a single signal and return a DataInstance.

        Parameters
        ----------
        var : str
            Access string for the signal (e.g. ``"bms.board.glvTemp"``).
        time_secs : int, optional
            How many seconds of history to request. Default is 30.

        Returns
        -------
        DataInstance
        """
        with self._client() as client:
            return client.get_range(var, time_secs)

    def get(
        self,
        access_string: str,
        value_type: ValueType,
    ) -> Union[float, bool, int]:
        """
        Get the latest scalar value for a signal.

        Parameters
        ----------
        access_string : str
            Access string for the signal.
        value_type : ValueType
            Expected type — required because the server validates this for
            point queries.

        Returns
        -------
        float | bool | int
        """
        with self._client() as client:
            return client.get(access_string, value_type)

    def set(
        self,
        access_string: str,
        value: Union[float, bool, int],
        value_type: ValueType,
    ) -> None:
        """
        Set a value on the CDP server.

        Parameters
        ----------
        access_string : str
            Access string for the signal.
        value : float | bool | int
            The value to set.
        value_type : ValueType
            Type of the value — required because the server validates this.
        """
        with self._client() as client:
            client.set(access_string, value, value_type)

    def plot(
        self,
        var_1: RangeSpec,
        var_2: RangeSpec | None = None,
        time_secs: int = _DEFAULT_TIME_SECS,
        title: str | None = None,
        y_label_1: str | None = None,
        y_label_2: str | None = None,
        show_legend: bool = True,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    ) -> go.Figure:
        """
        Fetch and plot one or two groups of signals.

        Parameters
        ----------
        var_1 : str | list[str]
            Signal(s) for the left y-axis.
        var_2 : str | list[str] | None, optional
            Signal(s) for the right y-axis.
        time_secs : int, optional
            Seconds of history to fetch for all signals. Default is 30.
        title : str | None, optional
        y_label_1 : str | None, optional
            Label for the left (or only) y-axis.
        y_label_2 : str | None, optional
            Label for the right y-axis.
        show_legend : bool, optional
            Whether to show plot legends. Default is True.
        font_config : FontConfig, optional
        layout_config : LayoutConfig, optional

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> live = LiveAnalyzer.local()

        >>> fig = live.plot("bms.board.glvTemp")
        >>> fig = live.plot("bms.board.glvTemp", var_2="motor.temp", time_secs=60)
        >>> fig = live.plot(["bms.board.glvTemp", "bms.board.hvTemp"])
        """
        left_dis = self._fetch_many(var_1, time_secs)

        if var_2 is not None:
            right_dis = self._fetch_many(var_2, time_secs)
            return plot_dual_axis(
                left_data_instances=left_dis,
                right_data_instances=right_dis,
                title=title,
                left_y_axis_title=y_label_1,
                right_y_axis_title=y_label_2,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
                timestamp_unit=_TIMESTAMP_UNIT,
            )
        else:
            return plot_single_axis(
                data_instances=left_dis,
                title=title,
                y_axis_title=y_label_1,
                show_legend=show_legend,
                font_config=font_config,
                layout_config=layout_config,
                timestamp_unit=_TIMESTAMP_UNIT,
            )

    def analyze_frequency(
        self,
        var: str,
        time_secs: int = _DEFAULT_TIME_SECS,
        expected_frequency_hz: float | None = None,
        gap_threshold_multiplier: float = 2.0,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
        plot_config: ScatterHistogramPlotConfig = DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG,
    ) -> go.Figure:
        """
        Fetch a signal and run a frequency diagnostic on it.

        Parameters
        ----------
        var : str
            Access string for the signal.
        time_secs : int, optional
            Seconds of history to fetch. Default is 30.
        expected_frequency_hz : float | None, optional
            Nominal expected sampling frequency in Hz. Default is None.
        gap_threshold_multiplier : float, optional
            Intervals exceeding this multiple of the expected (or median)
            interval are flagged as gaps. Default is 2.0.
        font_config : FontConfig, optional
        layout_config : LayoutConfig, optional
        plot_config : ScatterHistogramPlotConfig, optional

        Returns
        -------
        go.Figure
        """
        di = self.fetch(var, time_secs)
        return _analyze_frequency(
            di,
            expected_frequency_hz=expected_frequency_hz,
            source_time_unit=_TIMESTAMP_UNIT,
            gap_threshold_multiplier=gap_threshold_multiplier,
            font_config=font_config,
            layout_config=layout_config,
            plot_config=plot_config,
        )

    def __repr__(self) -> str:
        return f"LiveAnalyzer(host={self._host!r}, port={self._port})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> CDPClient:
        """Return a connected CDPClient ready to use as a context manager."""
        client = CDPClient(timeout=self._timeout, range_timeout=self._range_timeout)
        client.connect(self._host, self._port)
        return client

    def _fetch_many(self, specs: RangeSpec, time_secs: int) -> List[DataInstance]:
        """Fetch multiple signals in a single connection."""
        with self._client() as client:
            return [client.get_range(s, time_secs) for s in _to_list(specs)]

    def _check_connection(self) -> None:
        """Eagerly verify that the server is reachable."""
        try:
            with self._client():
                pass
        except CDPException as e:
            raise CDPException(
                f"Could not connect to CDP server at {self._host}:{self._port} — {e}"
            )
