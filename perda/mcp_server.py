"""FastMCP server exposing PERDA analyzer capabilities as MCP tools."""

import contextlib
import io
import json
from typing import Any

import numpy as np

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional extra
    FastMCP = None  # type: ignore[assignment]
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None

from perda.analyzer import Analyzer
from perda.core_data_structures.data_instance import DataInstance


_analyzers: dict[str, Analyzer] = {}

if FastMCP is not None:
    mcp = FastMCP("PERDA")
else:  # pragma: no cover - depends on optional extra
    class _MissingFastMCP:
        """Fallback object used when FastMCP is not installed."""

        def tool(self) -> Any:
            """Return a no-op decorator for MCP tools.

            Returns
            -------
            Any
                Decorator that leaves the wrapped function unchanged.
            """

            def decorator(func: Any) -> Any:
                return func

            return decorator

        def run(self) -> None:
            """Raise a helpful error when attempting to start the server.

            Returns
            -------
            None
                This function returns nothing.
            """
            raise RuntimeError(
                "FastMCP is not installed. Install the MCP extra with "
                "`pip install perda[mcp]`."
            ) from FASTMCP_IMPORT_ERROR

    mcp = _MissingFastMCP()


def _get_analyzer(session_id: str) -> Analyzer:
    """Return an analyzer for a session ID.

    Parameters
    ----------
    session_id : str
        Client-provided session identifier.

    Returns
    -------
    Analyzer
        Loaded analyzer for the requested session.
    """
    if session_id not in _analyzers:
        raise KeyError(f"Session '{session_id}' does not exist.")
    return _analyzers[session_id]


def _get_data_instance(session_id: str, var_name: str) -> DataInstance:
    """Return a variable for a session.

    Parameters
    ----------
    session_id : str
        Client-provided session identifier.
    var_name : str
        Variable C++ name.

    Returns
    -------
    DataInstance
        Matching variable data instance.
    """
    analyzer = _get_analyzer(session_id)
    return analyzer.data[var_name]


def _format_error(exc: Exception) -> str:
    """Format an exception for MCP tool responses.

    Parameters
    ----------
    exc : Exception
        Exception to format.

    Returns
    -------
    str
        User-facing error string.
    """
    if isinstance(exc, KeyError) and exc.args:
        return f"Error: {exc.args[0]}"
    return f"Error: {exc}"


def _trim_data_instance(
    data_instance: DataInstance,
    ts_start: float | None = None,
    ts_end: float | None = None,
) -> DataInstance:
    """Trim a data instance to an optional timestamp range.

    Parameters
    ----------
    data_instance : DataInstance
        Variable data to trim.
    ts_start : float | None, optional
        Inclusive lower timestamp bound in the log's native timestamp units.
    ts_end : float | None, optional
        Inclusive upper timestamp bound in the log's native timestamp units.

    Returns
    -------
    DataInstance
        Original or trimmed data instance.
    """
    if ts_start is None and ts_end is None:
        return data_instance
    return data_instance.trim(ts_start=ts_start, ts_end=ts_end)


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    """Return evenly spaced indices for downsampling.

    Parameters
    ----------
    length : int
        Number of points in the source series.
    max_points : int
        Maximum number of points to keep.

    Returns
    -------
    np.ndarray
        Evenly spaced integer indices.
    """
    if max_points <= 0:
        raise ValueError("max_points must be greater than 0.")
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_points, dtype=np.int64)


def _json_default(value: Any) -> Any:
    """Convert repository types into JSON-serializable values.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-compatible value.
    """
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@mcp.tool()
def load_log(session_id: str, filepath: str) -> str:
    """Load a PER log file into a named session.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    filepath : str
        Path to a local CSV log file.

    Returns
    -------
    str
        Analyzer summary for the loaded log.
    """
    try:
        analyzer = Analyzer(filepath)
        _analyzers[session_id] = analyzer
        return str(analyzer)
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def unload_log(session_id: str) -> str:
    """Unload a previously loaded log session.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.

    Returns
    -------
    str
        Confirmation message.
    """
    if session_id not in _analyzers:
        return f"Session '{session_id}' not found."
    del _analyzers[session_id]
    return f"Unloaded session '{session_id}'."


@mcp.tool()
def list_sessions() -> str:
    """List active analyzer sessions.

    Returns
    -------
    str
        Newline-delimited session IDs.
    """
    if not _analyzers:
        return "No active sessions."
    return "\n".join(sorted(_analyzers))


@mcp.tool()
def get_summary(session_id: str) -> str:
    """Return the analyzer summary for a session.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.

    Returns
    -------
    str
        Human-readable variable summary.
    """
    try:
        return str(_get_analyzer(session_id))
    except KeyError as exc:
        return _format_error(exc)


@mcp.tool()
def search_variables(session_id: str, query: str) -> str:
    """Search variables in a loaded log using PERDA's search helper.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    query : str
        Natural language search query.

    Returns
    -------
    str
        Captured stdout from `Analyzer.search()`.
    """
    try:
        analyzer = _get_analyzer(session_id)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            analyzer.search(query)
        return buffer.getvalue().strip() or "No search output."
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def get_variable_stats(session_id: str, var_name: str) -> str:
    """Return summary statistics for a variable.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    var_name : str
        Variable C++ name.

    Returns
    -------
    str
        Formatted statistics string.
    """
    try:
        analyzer = _get_analyzer(session_id)
        data_instance = analyzer.data[var_name]
        values = data_instance.value_np
        timestamps = data_instance.timestamp_np
        if len(data_instance) == 0:
            min_value = None
            max_value = None
            mean_value = None
            std_value = None
            first_timestamp = None
            last_timestamp = None
        else:
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            first_timestamp = int(timestamps[0])
            last_timestamp = int(timestamps[-1])
        return "\n".join(
            [
                f"cpp_name: {data_instance.cpp_name}",
                f"label: {data_instance.label}",
                f"var_id: {data_instance.var_id}",
                f"n_samples: {len(data_instance)}",
                f"min: {min_value}",
                f"max: {max_value}",
                f"mean: {mean_value}",
                f"std: {std_value}",
                f"first_timestamp: {first_timestamp}",
                f"last_timestamp: {last_timestamp}",
                f"timestamp_unit: {analyzer.data.timestamp_unit.value}",
            ]
        )
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def get_variable_values(
    session_id: str,
    var_name: str,
    ts_start: float | None = None,
    ts_end: float | None = None,
    max_points: int = 1000,
) -> str:
    """Return variable values as JSON.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    var_name : str
        Variable C++ name.
    ts_start : float | None, optional
        Inclusive lower timestamp bound in the log's native timestamp units.
    ts_end : float | None, optional
        Inclusive upper timestamp bound in the log's native timestamp units.
    max_points : int, optional
        Maximum number of samples to return after even downsampling.

    Returns
    -------
    str
        JSON string containing timestamps and values.
    """
    try:
        data_instance = _trim_data_instance(
            _get_data_instance(session_id, var_name),
            ts_start=ts_start,
            ts_end=ts_end,
        )
        keep = _downsample_indices(len(data_instance), max_points)
        payload = {
            "timestamps": data_instance.timestamp_np[keep].tolist(),
            "values": data_instance.value_np[keep].tolist(),
        }
        return json.dumps(payload, default=_json_default)
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def plot_variable(
    session_id: str,
    var_1: str,
    var_2: str | None = None,
    ts_start: float | None = None,
    ts_end: float | None = None,
    title: str | None = None,
) -> str:
    """Create a Plotly figure for one or two variables.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    var_1 : str
        Variable plotted on the left axis.
    var_2 : str | None, optional
        Optional variable plotted on the right axis.
    ts_start : float | None, optional
        Inclusive start time in seconds.
    ts_end : float | None, optional
        Inclusive end time in seconds.
    title : str | None, optional
        Optional figure title.

    Returns
    -------
    str
        Plotly figure serialized with `Figure.to_json()`.
    """
    try:
        analyzer = _get_analyzer(session_id)
        fig = analyzer.plot(
            var_1=var_1,
            var_2=var_2,
            ts_start=ts_start,
            ts_end=ts_end,
            title=title,
        )
        return fig.to_json()
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def analyze_frequency(
    session_id: str,
    var_name: str,
    expected_hz: float | None = None,
) -> str:
    """Analyze the sampling frequency of a variable.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.
    var_name : str
        Variable C++ name.
    expected_hz : float | None, optional
        Expected sampling frequency in Hz.

    Returns
    -------
    str
        Plotly figure serialized with `Figure.to_json()`.
    """
    try:
        analyzer = _get_analyzer(session_id)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fig = analyzer.analyze_frequency(
                var=var_name, expected_frequency_hz=expected_hz
            )
        return fig.to_json()
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def get_accel_times(session_id: str) -> str:
    """Return PERDA acceleration segment results.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.

    Returns
    -------
    str
        Newline-delimited acceleration results.
    """
    try:
        results = _get_analyzer(session_id).get_accel_times()
        if not results:
            return "No acceleration segments found."
        return "\n".join(str(result) for result in results)
    except Exception as exc:
        return _format_error(exc)


@mcp.tool()
def get_log_metadata(session_id: str) -> str:
    """Return log metadata as JSON.

    Parameters
    ----------
    session_id : str
        Client-chosen session identifier.

    Returns
    -------
    str
        JSON string containing session metadata.
    """
    try:
        analyzer = _get_analyzer(session_id)
        data = analyzer.data
        payload = {
            "data_start_time": data.data_start_time,
            "data_end_time": data.data_end_time,
            "timestamp_unit": data.timestamp_unit,
            "total_data_points": data.total_data_points,
            "total_variables": len(data.id_to_instance),
            "concat_boundaries": data.concat_boundaries,
        }
        return json.dumps(payload, default=_json_default)
    except KeyError as exc:
        return _format_error(exc)


if __name__ == "__main__":
    mcp.run()
