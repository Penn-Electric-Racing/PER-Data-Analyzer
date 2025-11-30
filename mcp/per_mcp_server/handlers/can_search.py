"""CAN variable search handler using perda."""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from mcp.types import TextContent, Content

from perda.csv_parser import CSVParser
from perda.pretty_print_data import pretty_print_single_run_variables


# Tool definition for search
TOOL_DEFINITIONS = {
    "search_can_variables": {
        "description": "Search for CAN variables. Searches variable names and descriptions. Use space-separated terms for multiple keywords.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (space-separated keywords)"
                },
                "strict": {
                    "type": "boolean",
                    "description": "If true, ALL keywords must match. If false, ANY keyword matches. Default: false",
                    "default": False
                }
            },
            "required": ["query"]
        }
    },
    "list_all_can_variables": {
        "description": "List all CAN variables in the dataset",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
}


def handle_search_can_variables(arguments: dict) -> list[Content]:
    """Search for CAN variables using perda's search."""
    query = arguments.get("query", "")
    strict = arguments.get("strict", False)

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

    # Search using perda's function
    output = StringIO()
    with redirect_stdout(output):
        pretty_print_single_run_variables(
            data,
            search=query if query else None,
            strict_search=strict,
            sort_by="name"
        )

    result_text = output.getvalue()

    if not result_text or "No matching" in result_text:
        return [TextContent(
            type="text",
            text=f"No CAN variables found matching '{query}'"
        )]

    return [TextContent(type="text", text=result_text)]


def handle_list_all_can_variables(arguments: dict) -> list[Content]:
    """List all CAN variables in the dataset."""
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

    # List all variables
    output = StringIO()
    with redirect_stdout(output):
        pretty_print_single_run_variables(
            data,
            search=None,
            strict_search=False,
            sort_by="name"
        )

    return [TextContent(type="text", text=output.getvalue())]


# Tool handlers mapping
TOOL_HANDLERS = {
    "search_can_variables": handle_search_can_variables,
    "list_all_can_variables": handle_list_all_can_variables,
}
