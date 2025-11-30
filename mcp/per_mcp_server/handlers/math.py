"""Math operations handler for MCP server.

Provides tools for performing arithmetic operations on two data arrays.
Operations create new computed variables that can be used in graphs and other tools.
"""

import numpy as np
from mcp.types import TextContent, Content

from ..utils.data_fetcher import get_variable_data
from ..utils.computed_variables import register_computed_variable

TOOL_DEFINITIONS = {
    "add_variables": {
        "description": "Add two variables element-wise across their full time range. "
        "Data is aligned by timestamp using interpolation. "
        "Creates a new computed variable that can be used in graphs and analysis. "
        "Useful for combining power sources, summing forces, or aggregating measurements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable1": {
                    "type": "string",
                    "description": "First variable name (CSV variable or computed variable)"
                },
                "variable2": {
                    "type": "string",
                    "description": "Second variable name (CSV variable or computed variable)"
                },
                "result_name": {
                    "type": "string",
                    "description": "Optional custom name for the result variable (auto-generated if not provided)"
                }
            },
            "required": ["variable1", "variable2"]
        }
    },
    "subtract_variables": {
        "description": "Subtract variable2 from variable1 element-wise across their full time range. "
        "Data is aligned by timestamp using interpolation. "
        "Creates a new computed variable that can be used in graphs and analysis. "
        "Useful for calculating differences, deltas, or comparing measurements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable1": {
                    "type": "string",
                    "description": "First variable name (minuend)"
                },
                "variable2": {
                    "type": "string",
                    "description": "Second variable name (subtrahend)"
                },
                "result_name": {
                    "type": "string",
                    "description": "Optional custom name for the result variable (auto-generated if not provided)"
                }
            },
            "required": ["variable1", "variable2"]
        }
    },
    "multiply_variables": {
        "description": "Multiply two variables element-wise across their full time range. "
        "Data is aligned by timestamp using interpolation. "
        "Creates a new computed variable that can be used in graphs and analysis. "
        "Useful for calculating power (voltage � current), force calculations, or scaling factors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable1": {
                    "type": "string",
                    "description": "First variable name"
                },
                "variable2": {
                    "type": "string",
                    "description": "Second variable name"
                },
                "result_name": {
                    "type": "string",
                    "description": "Optional custom name for the result variable (auto-generated if not provided)"
                }
            },
            "required": ["variable1", "variable2"]
        }
    },
    "divide_variables": {
        "description": "Divide variable1 by variable2 element-wise across their full time range. "
        "Data is aligned by timestamp using interpolation. "
        "Creates a new computed variable that can be used in graphs and analysis. "
        "Useful for calculating ratios, efficiency metrics, or normalized measurements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variable1": {
                    "type": "string",
                    "description": "Numerator variable name"
                },
                "variable2": {
                    "type": "string",
                    "description": "Denominator variable name"
                },
                "result_name": {
                    "type": "string",
                    "description": "Optional custom name for the result variable (auto-generated if not provided)"
                }
            },
            "required": ["variable1", "variable2"]
        }
    }
}


def align_data_arrays(data1, data2):
    """
    Align two data arrays by interpolating to common timestamps.

    Uses the union of both timestamp sets within their overlapping range.

    Args:
        data1: First DataInstance or ComputedVariable
        data2: Second DataInstance or ComputedVariable

    Returns:
        tuple: (aligned_timestamps, aligned_values1, aligned_values2)
    """
    # Get timestamp and value arrays
    ts1 = data1.timestamp_np
    ts2 = data2.timestamp_np
    val1 = data1.value_np
    val2 = data2.value_np

    # Find overlapping time range
    start_time = max(ts1[0], ts2[0])
    end_time = min(ts1[-1], ts2[-1])

    if start_time > end_time:
        raise ValueError("No overlapping time range between the two variables")

    # Create a union of timestamps within the overlap range
    ts1_in_range = ts1[(ts1 >= start_time) & (ts1 <= end_time)]
    ts2_in_range = ts2[(ts2 >= start_time) & (ts2 <= end_time)]
    common_timestamps = np.unique(np.concatenate([ts1_in_range, ts2_in_range]))

    # Interpolate both data series to common timestamps
    aligned_val1 = np.interp(common_timestamps, ts1, val1)
    aligned_val2 = np.interp(common_timestamps, ts2, val2)

    return common_timestamps, aligned_val1, aligned_val2


def perform_arithmetic_operation(arguments: dict, operation: str, op_symbol: str, op_func) -> list[Content]:
    """
    Generic handler for arithmetic operations on two variables.

    Args:
        arguments: Tool arguments
        operation: Name of the operation (e.g., "addition", "subtraction")
        op_symbol: Symbol for the operation (e.g., "+", "-")
        op_func: Function to perform the operation (e.g., np.add, np.subtract)

    Returns:
        List of Content with operation results and the new variable name
    """
    # Parse arguments
    variable1 = arguments.get("variable1")
    variable2 = arguments.get("variable2")
    result_name = arguments.get("result_name")

    if not variable1 or not variable2:
        return [TextContent(type="text", text="Both variable1 and variable2 are required")]

    # Get the variable data (full time range)
    try:
        data1 = get_variable_data(variable1)
        data2 = get_variable_data(variable2)
    except KeyError as e:
        return [TextContent(type="text", text=f"Variable not found: {e}")]
    except FileNotFoundError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading variables: {e}")]

    # Check that we have data
    if len(data1) == 0 or len(data2) == 0:
        return [TextContent(type="text", text="One or both variables have no data points")]

    # Get labels for display
    label1 = getattr(data1, 'label', variable1)
    label2 = getattr(data2, 'label', variable2)

    # Align the data arrays
    try:
        aligned_timestamps, aligned_val1, aligned_val2 = align_data_arrays(data1, data2)
    except Exception as e:
        return [TextContent(type="text", text=f"Error aligning data: {e}")]

    # Perform the arithmetic operation
    try:
        # Special handling for division to avoid divide by zero
        if operation == "division":
            # Replace zeros with NaN to avoid divide by zero errors
            aligned_val2_safe = np.where(aligned_val2 == 0, np.nan, aligned_val2)
            result = op_func(aligned_val1, aligned_val2_safe)
            has_div_by_zero = np.any(np.isnan(result))
        else:
            result = op_func(aligned_val1, aligned_val2)
            has_div_by_zero = False

        # Filter out NaN values for statistics and storage
        finite_mask = np.isfinite(result)
        result_timestamps = aligned_timestamps[finite_mask]
        result_values = result[finite_mask]

        if len(result_values) == 0:
            return [TextContent(type="text", text=f"No valid results after {operation} (possibly all divide by zero)")]

        # Calculate statistics
        min_value = float(result_values.min())
        max_value = float(result_values.max())
        avg_value = float(result_values.mean())
        std_value = float(result_values.std())
        num_points = len(result_values)

        # Create label for the computed variable
        result_label = f"{label1} {op_symbol} {label2}"

        # Register the computed variable
        metadata = {
            "operation": operation,
            "variable1": variable1,
            "variable2": variable2,
            "label1": label1,
            "label2": label2,
            "statistics": {
                "min": min_value,
                "max": max_value,
                "mean": avg_value,
                "std": std_value,
                "count": num_points
            }
        }

        if has_div_by_zero:
            num_nan = np.sum(np.isnan(result))
            metadata["divide_by_zero_count"] = int(num_nan)

        computed_name = register_computed_variable(
            timestamps=result_timestamps,
            values=result_values,
            label=result_label,
            metadata=metadata,
            name=result_name
        )

        # Format output
        time_start = float(result_timestamps[0]) / 1000.0  # Convert to seconds
        time_end = float(result_timestamps[-1]) / 1000.0
        duration = time_end - time_start

        output = f"Created computed variable: '{computed_name}'\n"
        output += f"=" * 60 + "\n\n"
        output += f"Operation: {label1} {op_symbol} {label2}\n"
        output += f"Label: {result_label}\n\n"
        output += f"Time Range:\n"
        output += f"  Start: {time_start:.3f} seconds\n"
        output += f"  End: {time_end:.3f} seconds\n"
        output += f"  Duration: {duration:.3f} seconds\n\n"
        output += f"Result Statistics:\n"
        output += f"  Data Points: {num_points:,}\n"
        output += f"  Minimum: {min_value:.6f}\n"
        output += f"  Maximum: {max_value:.6f}\n"
        output += f"  Average: {avg_value:.6f}\n"
        output += f"  Std Dev: {std_value:.6f}\n"

        if has_div_by_zero:
            num_nan = metadata["divide_by_zero_count"]
            output += f"\Warning: {num_nan:,} divide-by-zero occurrences (excluded from result)\n"

        output += f"\nUse this variable name in graphs and other tools: '{computed_name}'"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error performing {operation}: {e}")]


def handle_add_variables(arguments: dict) -> list[Content]:
    """Add two variables element-wise."""
    return perform_arithmetic_operation(arguments, "addition", "+", np.add)


def handle_subtract_variables(arguments: dict) -> list[Content]:
    """Subtract variable2 from variable1 element-wise."""
    return perform_arithmetic_operation(arguments, "subtraction", "-", np.subtract)


def handle_multiply_variables(arguments: dict) -> list[Content]:
    """Multiply two variables element-wise."""
    return perform_arithmetic_operation(arguments, "multiplication", "�", np.multiply)


def handle_divide_variables(arguments: dict) -> list[Content]:
    """Divide variable1 by variable2 element-wise."""
    return perform_arithmetic_operation(arguments, "division", "÷", np.divide)


# Tool handlers mapping
TOOL_HANDLERS = {
    "add_variables": handle_add_variables,
    "subtract_variables": handle_subtract_variables,
    "multiply_variables": handle_multiply_variables,
    "divide_variables": handle_divide_variables,
}
