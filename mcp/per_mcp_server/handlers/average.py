"""Average handler for MCP server."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser
from perda.utils import average_over_time_range

# CSV path from environment variable (relative to MCP server's cwd which is mcp/)
CSV_PATH = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

TOOL_DEFINITIONS = {
    "average_variable_over_time": {
        "description": "Calculate the time-weighted average of a variable over a time range. "
        "More accurate than simple arithmetic mean for time-series data. "
        "Useful for average power draw, mean temperature, average speed over a stint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": "Full variable path (e.g., 'ams.pack.voltage')"
                },
                "start_time": {
                    "type": "integer",
                    "description": "Start time in milliseconds (optional, default: 0)"
                },
                "end_time": {
                    "type": "integer",
                    "description": "End time in milliseconds (optional, default: -1 for end of data)"
                },
                "time_unit": {
                    "type": "string",
                    "description": "Time unit for averaging: 'ms' or 's' (optional, default: 's')",
                    "enum": ["ms", "s"]
                }
            },
            "required": ["variable"]
        }
    }
}


def handle_average_variable_over_time(arguments: dict) -> list[Content]:
    """Calculate time-weighted average of a variable."""

    # Parse arguments
    variable = arguments.get("variable")
    start_time = int(arguments.get("start_time", 0))
    end_time = int(arguments.get("end_time", -1))
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

    # Calculate average
    try:
        average = average_over_time_range(data_instance, start_time, end_time, time_unit)

        # Get time range for display
        if end_time < 0:
            actual_end = data_instance.timestamp_np[-1] if len(data_instance) > 0 else 0
        else:
            actual_end = min(end_time, data_instance.timestamp_np[-1] if len(data_instance) > 0 else 0)

        actual_start = max(start_time, data_instance.timestamp_np[0] if len(data_instance) > 0 else 0)
        duration = actual_end - actual_start

        # Convert to display units
        if time_unit == "s":
            display_start = actual_start / 1000.0
            display_end = actual_end / 1000.0
            display_duration = duration / 1000.0
            time_label = "seconds"
        else:
            display_start = actual_start
            display_end = actual_end
            display_duration = duration
            time_label = "milliseconds"

        # Format output
        output = f"Time-Weighted Average of '{data_instance.label}' (CAN ID: {data_instance.canid})\n"
        output += f"=" * 50 + "\n\n"
        output += f"Time Range:\n"
        output += f"  Start: {display_start:.3f} {time_label}\n"
        output += f"  End: {display_end:.3f} {time_label}\n"
        output += f"  Duration: {display_duration:.3f} {time_label}\n\n"
        output += f"Average: {average:.6f}\n"
        output += f"\nNote: This is a time-weighted average, which accounts for\n"
        output += f"how long the variable stayed at each value."

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating average: {e}")]


# Tool handlers mapping
TOOL_HANDLERS = {
    "average_variable_over_time": handle_average_variable_over_time,
}
