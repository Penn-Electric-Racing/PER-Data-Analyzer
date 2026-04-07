"""
perda.live.live_analyzer
------------------------
Live data analysis using the CDP IPC server, mirroring the Analyzer interface.
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

PlotInput = Union[
    str,
    DataInstance,
    List[Union[str, DataInstance]],
]

_DATASERVER_IP = "198.74.62.28"
_DEFAULT_PORT = 5001
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_RANGE_TIMEOUT = 2.0
_DEFAULT_TIME_SECS = 30
_TIMESTAMP_UNIT = Timescale.MS


class LiveAnalyzer:
    """
    Live data analyzer that pulls time-series data from the CDP IPC server
    and exposes a plotting interface analogous to ``Analyzer``.
    """

    def __init__(self, host: str, port: int, timeout: float, range_timeout: float):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._range_timeout = range_timeout

        self._connected = False
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
        return cls("127.0.0.1", port, timeout, range_timeout)

    @classmethod
    def dataserver(
        cls,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        range_timeout: float = _DEFAULT_RANGE_TIMEOUT,
    ) -> "LiveAnalyzer":
        return cls(_DATASERVER_IP, port, timeout, range_timeout)

    @classmethod
    def remote(
        cls,
        host: str,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        range_timeout: float = _DEFAULT_RANGE_TIMEOUT,
    ) -> "LiveAnalyzer":
        return cls(host, port, timeout, range_timeout)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        try:
            with self._client():
                self._connected = True
        except CDPException as e:
            self._connected = False
            raise CDPException(
                f"Could not connect to CDP server at {self._host}:{self._port} — {e}"
            )

    def is_connected(self, refresh: bool = False) -> bool:
        if refresh:
            try:
                with self._client():
                    self._connected = True
            except CDPException:
                self._connected = False
        return self._connected

    def ensure_connected(self) -> None:
        if not self.is_connected(refresh=True):
            raise CDPException("Lost connection to CDP server")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        var: str,
        time_secs: int = _DEFAULT_TIME_SECS,
    ) -> DataInstance:
        self.ensure_connected()
        with self._client() as client:
            return client.get_range(var, time_secs)

    def get(
        self,
        access_string: str,
        value_type: ValueType,
    ) -> Union[float, bool, int]:
        self.ensure_connected()
        with self._client() as client:
            return client.get(access_string, value_type)

    def set(
        self,
        access_string: str,
        value: Union[float, bool, int],
        value_type: ValueType,
    ) -> None:
        self.ensure_connected()
        with self._client() as client:
            client.set(access_string, value, value_type)

    def plot(
        self,
        var_1: PlotInput,
        var_2: PlotInput | None = None,
        time_secs: int = _DEFAULT_TIME_SECS,
        title: str | None = None,
        y_label_1: str | None = None,
        y_label_2: str | None = None,
        show_legend: bool = True,
        font_config: FontConfig = DEFAULT_FONT_CONFIG,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
    ) -> go.Figure:
        self.ensure_connected()

        left_dis = self._normalize_input(var_1, time_secs)

        if var_2 is not None:
            right_dis = self._normalize_input(var_2, time_secs)

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
        client = CDPClient(timeout=self._timeout, range_timeout=self._range_timeout)
        client.connect(self._host, self._port)
        return client

    def _normalize_input(
        self,
        input_data: PlotInput,
        time_secs: int,
    ) -> List[DataInstance]:
        """
        Normalize input into a list of DataInstances.
        Fetch missing signals in a single connection.
        """
        if isinstance(input_data, DataInstance):
            return [input_data]

        items = input_data if isinstance(input_data, list) else [input_data]

        result: List[DataInstance] = []
        to_fetch: List[str] = []

        for item in items:
            if isinstance(item, DataInstance):
                result.append(item)
            else:
                to_fetch.append(item)

        if to_fetch:
            with self._client() as client:
                fetched = [client.get_range(s, time_secs) for s in to_fetch]
            result.extend(fetched)

        return result
