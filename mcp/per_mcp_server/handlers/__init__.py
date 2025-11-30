"""Tool handlers for the PER MCP Server.

This module aggregates all tool definitions and handlers from individual handler modules.
Each handler module defines its own tool definitions and implementations, keeping related
functionality self-contained.
"""

from typing import Callable

from . import (
    can_search,
    can_query,
    build_graph,
    variable_stats,
    dataset_info,
    time_slice,
    integrate,
    average,
    math
)


# Aggregate all tool definitions from handler modules
TOOL_DEFINITIONS = {
    **can_search.TOOL_DEFINITIONS,
    **can_query.TOOL_DEFINITIONS,
    **build_graph.TOOL_DEFINITIONS,
    **variable_stats.TOOL_DEFINITIONS,
    **dataset_info.TOOL_DEFINITIONS,
    **time_slice.TOOL_DEFINITIONS,
    **integrate.TOOL_DEFINITIONS,
    **average.TOOL_DEFINITIONS,
    **math.TOOL_DEFINITIONS,
}


# Aggregate all tool handlers from handler modules
TOOL_HANDLERS = {
    **can_search.TOOL_HANDLERS,
    **can_query.TOOL_HANDLERS,
    **build_graph.TOOL_HANDLERS,
    **variable_stats.TOOL_HANDLERS,
    **dataset_info.TOOL_HANDLERS,
    **time_slice.TOOL_HANDLERS,
    **integrate.TOOL_HANDLERS,
    **average.TOOL_HANDLERS,
    **math.TOOL_HANDLERS,
}


def get_tool_handlers() -> dict[str, Callable]:
    """
    Get all tool handlers.

    Returns:
        Dictionary mapping tool names to handler functions

    Note: This function is kept for backwards compatibility.
    Prefer using TOOL_HANDLERS directly for better performance.
    """
    return TOOL_HANDLERS


__all__ = ["TOOL_DEFINITIONS", "TOOL_HANDLERS", "get_tool_handlers"]
