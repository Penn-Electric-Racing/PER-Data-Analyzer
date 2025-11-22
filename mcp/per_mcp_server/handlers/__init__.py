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
        "build_graph_vs_time": lambda args: build_graph.handle_build_graph_vs_time(args),
        "build_dual_axis_graph": lambda args: build_graph.handle_build_dual_axis_graph(args),
        "get_variable_statistics": lambda args: variable_stats.handle_get_variable_statistics(args),
        "get_dataset_info": lambda args: dataset_info.handle_get_dataset_info(args),
        "get_variable_time_slice": lambda args: time_slice.handle_get_variable_time_slice(args),
        "integrate_variable_over_time": lambda args: integrate.handle_integrate_variable_over_time(args),
        "average_variable_over_time": lambda args: average.handle_average_variable_over_time(args),
    }


__all__ = ["TOOL_DEFINITIONS", "get_tool_handlers"]
