"""Time slice handler for MCP server."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser
from perda.utils import get_data_slice_by_timestamp

# CSV path from environment variable (relative to MCP server's cwd which is mcp/)
CSV_PATH = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

TOOL_DEFINITIONS = {
    "get_variable_time_slice": {
        "description": "Extract a subset of variable data within a specific time range. "
        "Useful for analyzing specific race phases, laps, or problem periods.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": "Full variable path (e.g., 'ams.pack.soc')"
                },
                "start_time": {
                    "type": "integer",
                    "description": "Start time in milliseconds"
                },
                "end_time": {
                    "type": "integer",
                    "description": "End time in milliseconds (-1 for end of data)"
                },
                "time_unit": {
                    "type": "string",
                    "description": "Time unit for display: 'ms' or 's' (optional, default: 's')",
                    "enum": ["ms", "s"]
                }
            },
            "required": ["variable", "start_time", "end_time"]
        }
    }
}


def handle_get_variable_time_slice(arguments: dict) -> list[Content]:
    """Get a time slice of variable data."""

    # Parse arguments
    variable = arguments.get("variable")
    start_time = int(arguments.get("start_time"))
    end_time = int(arguments.get("end_time"))
    time_unit = arguments.get("time_unit", "s")

    if not variable:
        return [TextContent(type="text", text="No variable provided")]

    if not os.path.exists(CSV_PATH):
        return [TextContent(type="text", text=f"CSV file not found: {CSV_PATH}")]

    # Parse the CSV (suppress output)
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(CSV_PATH)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Get the variable data
    try:
        data_instance = data[variable]
    except KeyError:
        return [TextContent(type="text", text=f"Variable '{variable}' not found in CSV")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error accessing variable: {e}")]

    # Get time slice
    try:
        sliced_data = get_data_slice_by_timestamp(data_instance, start_time, end_time)

        if len(sliced_data) == 0:
            return [TextContent(type="text", text=f"No data points found for '{variable}' in the specified time range")]

        # Calculate slice statistics
        min_value = float(sliced_data.value_np.min())
        max_value = float(sliced_data.value_np.max())
        avg_value = float(sliced_data.value_np.mean())
        num_points = len(sliced_data)

        actual_start = float(sliced_data.timestamp_np[0])
        actual_end = float(sliced_data.timestamp_np[-1])

        # Convert to display units
        if time_unit == "s":
            display_start = actual_start / 1000.0
            display_end = actual_end / 1000.0
            time_label = "seconds"
        else:
            display_start = actual_start
            display_end = actual_end
            time_label = "milliseconds"

        # Format output
        output = f"Time Slice for '{sliced_data.label}' (CAN ID: {sliced_data.canid})\n"
        output += f"=" * 50 + "\n\n"
        output += f"Requested Range:\n"
        output += f"  Start: {start_time / 1000.0 if time_unit == 's' else start_time:.3f} {time_label}\n"
        output += f"  End: {end_time / 1000.0 if time_unit == 's' else end_time:.3f} {time_label if end_time >= 0 else 'end of data'}\n\n"
        output += f"Actual Data Range:\n"
        output += f"  Start: {display_start:.3f} {time_label}\n"
        output += f"  End: {display_end:.3f} {time_label}\n"
        output += f"  Duration: {display_end - display_start:.3f} {time_label}\n\n"
        output += f"Statistics:\n"
        output += f"  Data Points: {num_points}\n"
        output += f"  Minimum: {min_value:.6f}\n"
        output += f"  Maximum: {max_value:.6f}\n"
        output += f"  Average: {avg_value:.6f}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting time slice: {e}")]


# Tool handlers mapping
TOOL_HANDLERS = {
    "get_variable_time_slice": handle_get_variable_time_slice,
}
