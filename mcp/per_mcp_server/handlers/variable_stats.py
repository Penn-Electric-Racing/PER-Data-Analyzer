"""Variable statistics handler for MCP server."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser
from perda.utils import integrate_over_time_range, average_over_time_range

# CSV path (relative to MCP server's cwd which is mcp/)
CSV_PATH = "temp/16thMay13-52.csv"

TOOL_DEFINITIONS = {
    "get_variable_statistics": {
        "description": "Get comprehensive statistics for a CAN variable including min/max values, timestamps, average, integral, and data point count. "
        "Useful for understanding variable behavior and finding extremes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": "Full variable path (e.g., 'ams.pack.soc')"
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
                    "description": "Time unit: 'ms' or 's' (optional, default: 's')",
                    "enum": ["ms", "s"]
                }
            },
            "required": ["variable"]
        }
    }
}


def handle_get_variable_statistics(arguments: dict) -> list[Content]:
    """Get statistics for a CAN variable."""

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

    # Calculate statistics
    try:
        # Get time range to display
        if end_time < 0:
            actual_end_time = data_instance.timestamp_np[-1] if len(data_instance) > 0 else 0
        else:
            actual_end_time = min(end_time, data_instance.timestamp_np[-1] if len(data_instance) > 0 else 0)

        actual_start_time = max(start_time, data_instance.timestamp_np[0] if len(data_instance) > 0 else 0)

        # Convert to display units
        if time_unit == "s":
            display_start = actual_start_time / 1000.0
            display_end = actual_end_time / 1000.0
            time_label = "seconds"
        else:
            display_start = actual_start_time
            display_end = actual_end_time
            time_label = "milliseconds"

        # Calculate integral and average
        integral = integrate_over_time_range(data_instance, start_time, end_time, time_unit)
        average = average_over_time_range(data_instance, start_time, end_time, time_unit)

        # Find min/max in the time range
        mask = data_instance.timestamp_np >= actual_start_time
        if end_time >= 0:
            mask &= data_instance.timestamp_np < actual_end_time

        filtered_values = data_instance.value_np[mask]
        filtered_timestamps = data_instance.timestamp_np[mask]

        if len(filtered_values) == 0:
            return [TextContent(type="text", text=f"No data points found for '{variable}' in the specified time range")]

        min_value = float(filtered_values.min())
        max_value = float(filtered_values.max())
        min_idx = filtered_values.argmin()
        max_idx = filtered_values.argmax()
        min_time = float(filtered_timestamps[min_idx])
        max_time = float(filtered_timestamps[max_idx])

        # Convert timestamps to display units
        if time_unit == "s":
            min_time /= 1000.0
            max_time /= 1000.0

        # Format output
        output = f"Statistics for '{data_instance.label}' (CAN ID: {data_instance.canid})\n"
        output += f"Time range: {display_start:.3f} to {display_end:.3f} {time_label}\n"
        output += f"Data points: {len(filtered_values)}\n\n"
        output += f"Minimum: {min_value:.6f} at {min_time:.3f} {time_label}\n"
        output += f"Maximum: {max_value:.6f} at {max_time:.3f} {time_label}\n"
        output += f"Average: {average:.6f}\n"
        output += f"Integral: {integral:.6f}"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating statistics: {e}")]
