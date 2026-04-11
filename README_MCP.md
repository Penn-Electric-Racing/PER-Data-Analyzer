# PERDA MCP Server

`perda/mcp_server.py` exposes PERDA Analyzer functionality as MCP tools through FastMCP. MCP clients such as Claude Desktop can load local PER CSV logs, inspect variables, run built-in analyses, and request Plotly figure JSON for client-side rendering.

## Install

```bash
pip install perda[mcp]
```

## Run

```bash
fastmcp run perda/mcp_server.py
```

You can also run the module directly:

```bash
python perda/mcp_server.py
```

## Tools

- `load_log(session_id, filepath)`: Load a local CSV log into a named session and return the analyzer summary.
- `unload_log(session_id)`: Remove a loaded session from server memory.
- `list_sessions()`: List all active session IDs.
- `get_summary(session_id)`: Return the full PERDA variable summary for a loaded session.
- `search_variables(session_id, query)`: Run PERDA's natural-language variable search and return its printed output.
- `get_variable_stats(session_id, var_name)`: Return basic statistics and metadata for one variable.
- `get_variable_values(session_id, var_name, ts_start, ts_end, max_points)`: Return timestamps and values as JSON, trimmed in native log timestamp units and evenly downsampled if needed.
- `plot_variable(session_id, var_1, var_2, ts_start, ts_end, title)`: Return a Plotly figure JSON string for one or two variables. `ts_start` and `ts_end` are in seconds, matching `Analyzer.plot()`.
- `analyze_frequency(session_id, var_name, expected_hz)`: Return a Plotly figure JSON string with frequency diagnostics for a variable.
- `get_accel_times(session_id)`: Return formatted PERDA acceleration test results.
- `get_log_metadata(session_id)`: Return log metadata as JSON, including time bounds, timestamp unit, variable count, data point count, and concat boundaries.

## Claude Desktop Config

Example configuration:

```json
{
  "mcpServers": {
    "perda": {
      "command": "fastmcp",
      "args": [
        "run",
        "/absolute/path/to/per-data-analyzer/perda/mcp_server.py"
      ]
    }
  }
}
```
