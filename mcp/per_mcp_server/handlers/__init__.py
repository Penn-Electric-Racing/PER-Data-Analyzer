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
}


def get_tool_handlers() -> dict[str, Callable]:
    """
    Get all tool handlers.

    Returns:
        Dictionary mapping tool names to handler functions
    """
    return {
        # CAN search handlers (now use perda directly)
        "search_can_variables": can_search.handle_search_can_variables,
        "list_all_can_variables": can_search.handle_list_all_can_variables,
        # CAN query handlers (now use perda directly)
        "get_can_variable_info": can_query.handle_get_can_variable_info,
        "get_dataset_overview": can_query.handle_get_dataset_overview,
        # Graph handlers
        "build_graph_vs_time": build_graph.handle_build_graph_vs_time,
        "build_dual_axis_graph": build_graph.handle_build_dual_axis_graph,
        # Analysis handlers
        "get_variable_statistics": variable_stats.handle_get_variable_statistics,
        "get_dataset_info": dataset_info.handle_get_dataset_info,
        "get_variable_time_slice": time_slice.handle_get_variable_time_slice,
        "integrate_variable_over_time": integrate.handle_integrate_variable_over_time,
        "average_variable_over_time": average.handle_average_variable_over_time,
    }


__all__ = ["TOOL_DEFINITIONS", "get_tool_handlers"]
