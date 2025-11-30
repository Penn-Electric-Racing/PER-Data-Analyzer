"""Dataset information handler for MCP server."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser

# CSV path from environment variable (relative to MCP server's cwd which is mcp/)
CSV_PATH = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

TOOL_DEFINITIONS = {
    "get_dataset_info": {
        "description": "Get overview information about the loaded CAN dataset including time range, number of variables, total data points, and CAN IDs. "
        "Useful for understanding dataset scope before detailed analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_unit": {
                    "type": "string",
                    "description": "Time unit for display: 'ms' or 's' (optional, default: 's')",
                    "enum": ["ms", "s"]
                }
            }
        }
    }
}


def handle_get_dataset_info(arguments: dict) -> list[Content]:
    """Get overview information about the dataset."""

    # Parse arguments
    time_unit = arguments.get("time_unit", "s")

    if not os.path.exists(CSV_PATH):
        return [TextContent(type="text", text=f"CSV file not found: {CSV_PATH}")]

    # Parse the CSV (suppress output)
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(CSV_PATH)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Calculate dataset info
    try:
        # Time range
        start_time = data.data_start_time
        end_time = data.data_end_time
        duration = end_time - start_time

        # Convert to display units
        if time_unit == "s":
            display_start = start_time / 1000.0
            display_end = end_time / 1000.0
            display_duration = duration / 1000.0
            time_label = "seconds"
        else:
            display_start = start_time
            display_end = end_time
            display_duration = duration
            time_label = "milliseconds"

        # Counts
        num_variables = len(data.id_map)
        num_can_ids = len(data.tv_map)
        total_points = data.total_data_points

        # Calculate average data rate
        avg_rate = total_points / display_duration if display_duration > 0 else 0

        # Format output
        output = f"Dataset Overview\n"
        output += f"=" * 50 + "\n\n"
        output += f"Time Range:\n"
        output += f"  Start: {display_start:.3f} {time_label}\n"
        output += f"  End: {display_end:.3f} {time_label}\n"
        output += f"  Duration: {display_duration:.3f} {time_label}\n\n"
        output += f"Variables:\n"
        output += f"  Total CAN IDs: {num_can_ids}\n"
        output += f"  Total Variables: {num_variables}\n\n"
        output += f"Data Points:\n"
        output += f"  Total: {total_points:,}\n"
        output += f"  Average Rate: {avg_rate:.1f} points/{time_label[:-1] if time_label.endswith('s') else time_label}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating dataset info: {e}")]


# Tool handlers mapping
TOOL_HANDLERS = {
    "get_dataset_info": handle_get_dataset_info,
}
