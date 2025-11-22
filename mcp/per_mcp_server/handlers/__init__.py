"""Tool handlers for the PER MCP Server.

This module aggregates all tool definitions and handlers from individual handler modules.
Each handler module defines its own tool definitions and implementations, keeping related
functionality self-contained.
"""

from typing import Callable

from . import can_search, can_query


# Aggregate all tool definitions from handler modules
TOOL_DEFINITIONS = {
    **can_search.TOOL_DEFINITIONS,
    **can_query.TOOL_DEFINITIONS,
}


def get_tool_handlers(can_search_instance) -> dict[str, Callable]:
    """
    Get tool handlers with can_search bound.

    Args:
        can_search_instance: CANSearch instance to use for handlers

    Returns:
        Dictionary mapping tool names to handler functions
    """
    return {
        "search_can_variables": lambda args: can_search.handle_search_can_variables(can_search_instance, args),
        "get_can_variable": lambda args: can_query.handle_get_can_variable(can_search_instance, args),
        "list_device_variables": lambda args: can_query.handle_list_device_variables(can_search_instance, args),
        "list_can_devices": lambda args: can_query.handle_list_can_devices(can_search_instance, args),
    }


__all__ = ["TOOL_DEFINITIONS", "get_tool_handlers"]
