"""CAN variable query handlers using perda."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser
from perda.pretty_print_data import (
    pretty_print_data_instance_info,
    pretty_print_single_run_info,
)


# Tool definitions for query operations
TOOL_DEFINITIONS = {
    "get_can_variable_info": {
        "description": "Get detailed information and statistics for a specific CAN variable by its path (e.g., 'ams.pack.soc')",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable_path": {
                    "type": "string",
                    "description": "Full variable path (e.g., 'ams.pack.soc')"
                },
                "time_unit": {
                    "type": "string",
                    "description": "Time unit: 's' or 'ms' (default: 's')",
                    "enum": ["s", "ms"],
                    "default": "s"
                }
            },
            "required": ["variable_path"]
        }
    },
    "get_dataset_overview": {
        "description": "Get overview information about the loaded CAN dataset (time range, variable count, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_unit": {
                    "type": "string",
                    "description": "Time unit: 's' or 'ms' (default: 's')",
                    "enum": ["s", "ms"],
                    "default": "s"
                }
            }
        }
    }
}


def handle_get_can_variable_info(arguments: dict) -> list[Content]:
    """Get detailed info about a specific CAN variable."""
    variable_path = arguments.get("variable_path")
    time_unit = arguments.get("time_unit", "s")

    csv_path = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

    if not os.path.exists(csv_path):
        return [TextContent(type="text", text=f"CSV file not found: {csv_path}")]

    # Parse CSV
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(csv_path)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Get variable data instance
    try:
        data_instance = data[variable_path]
    except KeyError:
        return [TextContent(
            type="text",
            text=f"Variable '{variable_path}' not found in dataset. Use search_can_variables to find available variables."
        )]

    # Get info using perda's function
    output = StringIO()
    with redirect_stdout(output):
        pretty_print_data_instance_info(data_instance, time_unit=time_unit)

    return [TextContent(type="text", text=output.getvalue())]


def handle_get_dataset_overview(arguments: dict) -> list[Content]:
    """Get overview of the dataset."""
    time_unit = arguments.get("time_unit", "s")

    csv_path = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

    if not os.path.exists(csv_path):
        return [TextContent(type="text", text=f"CSV file not found: {csv_path}")]

    # Parse CSV
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(csv_path)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Get overview using perda's function
    output = StringIO()
    with redirect_stdout(output):
        pretty_print_single_run_info(data, time_unit=time_unit)

    return [TextContent(type="text", text=output.getvalue())]


# Tool handlers mapping
TOOL_HANDLERS = {
    "get_can_variable_info": handle_get_can_variable_info,
    "get_dataset_overview": handle_get_dataset_overview,
}
