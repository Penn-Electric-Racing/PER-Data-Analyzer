import os
import sys
import base64
from io import StringIO, BytesIO
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from mcp.types import TextContent, ImageContent, Content

from perda.csv_parser import CSVParser
from perda.utils import get_data_slice_by_timestamp

# Output directory for saved graphs (relative to MCP server's cwd which is mcp/)
GRAPH_OUTPUT_DIR = "temp"

TOOL_DEFINITIONS = {
    "build_graph_vs_time": {
        "description": "Build a graph of a CAN variable over time. Input should be a list of CAN variable names, and the time range to plot."
        " If no time range is provided, it will default to the full range of available data. Time range should be provided in milliseconds since beginning of the log.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Full variable path (e.g., 'pdu.sensors.batCurrent')"
                    },
                },
                "start_time": {
                    "type": "integer",
                    "description": "Start time in milliseconds since beginning of log (optional)"
                },
                "end_time": {
                    "type": "integer",
                    "description": "End time in milliseconds since beginning of log (optional)"
                }
            },
            "required": ["variables"]
        }
    },
    "build_dual_axis_graph": {
        "description": "Build a graph with two y-axes for comparing variables with different scales. "
        "Left and right variables are plotted on separate y-axes. Useful for comparing voltage vs current, temperature vs power, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "left_variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Variable paths for left y-axis (e.g., 'ams.pack.voltage')"
                    },
                },
                "right_variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Variable paths for right y-axis (e.g., 'ams.pack.current')"
                    },
                },
                "start_time": {
                    "type": "integer",
                    "description": "Start time in milliseconds (optional)"
                },
                "end_time": {
                    "type": "integer",
                    "description": "End time in milliseconds (optional)"
                }
            },
            "required": ["left_variables"]
        }
    }
}

def handle_build_graph_vs_time(arguments: dict) -> list[Content]:
    """Build a graph of CAN variables over time."""

    # Parse arguments
    variables = arguments.get("variables", [])
    start_time = int(arguments.get("start_time", 0))
    end_time = int(arguments.get("end_time", -1))

    if not variables:
        return [TextContent(type="text", text="No variables provided")]

    # Get CSV path from environment variable
    csv_path = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

    if not os.path.exists(csv_path):
        return [TextContent(type="text", text=f"CSV file not found: {csv_path}")]

    # Parse the CSV (suppress output to avoid polluting MCP stdout)
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(csv_path)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted_count = 0
    plotted_labels = []
    for i, var_name in enumerate(variables):
        try:
            # Get data instance for this variable
            di = data[var_name]

            # Apply time filtering
            filtered_di = get_data_slice_by_timestamp(di, start_time, end_time)

            if len(filtered_di) == 0:
                continue

            # Convert timestamps to seconds for better readability
            ts = filtered_di.timestamp_np.astype(np.float64) / 1e3
            val = filtered_di.value_np

            # Plot with label
            ax.plot(ts, val, label=filtered_di.label, color=colors[i % len(colors)])
            plotted_labels.append(filtered_di.label)
            plotted_count += 1

        except (KeyError, Exception):
            continue

    if plotted_count == 0:
        plt.close(fig)
        return [TextContent(type="text", text="No variables could be plotted")]

    # Configure plot with better labels
    ax.set_xlabel("Time (s)")

    # Create meaningful y-axis label
    if plotted_count == 1:
        ax.set_ylabel(plotted_labels[0])
    else:
        # For multiple variables, use a descriptive label
        ax.set_ylabel(f"Values ({plotted_count} variables)")

    # Title with variable names if there aren't too many
    if plotted_count <= 3:
        title = ", ".join(plotted_labels)
    else:
        title = f"{plotted_labels[0]}, {plotted_labels[1]}, and {plotted_count - 2} more"
    ax.set_title(title)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save plot to BytesIO buffer and encode as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    # Return image content with base64 data
    return [
        TextContent(
            type="text",
            text=f"Plotted {plotted_count} variable(s): {', '.join(plotted_labels)}"
        ),
        ImageContent(
            type="image",
            data=image_base64,
            mimeType="image/png"
        )
    ]


def handle_build_dual_axis_graph(arguments: dict) -> list[Content]:
    """Build a graph with dual y-axes for comparing variables with different scales."""

    # Parse arguments
    left_variables = arguments.get("left_variables", [])
    right_variables = arguments.get("right_variables", [])
    start_time = int(arguments.get("start_time", 0))
    end_time = int(arguments.get("end_time", -1))

    if not left_variables:
        return [TextContent(type="text", text="No left variables provided")]

    csv_path = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

    if not os.path.exists(csv_path):
        return [TextContent(type="text", text=f"CSV file not found: {csv_path}")]

    # Parse the CSV (suppress output)
    parser = CSVParser()
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            data = parser(csv_path)
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing CSV: {e}")]

    # Create the plot with dual axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    # Plot left axis variables
    left_plotted = 0
    left_labels = []
    for var_name in left_variables:
        try:
            di = data[var_name]
            filtered_di = get_data_slice_by_timestamp(di, start_time, end_time)

            if len(filtered_di) == 0:
                continue

            ts = filtered_di.timestamp_np.astype(np.float64) / 1e3
            val = filtered_di.value_np

            ax1.plot(ts, val, label=filtered_di.label, color=colors[color_idx % len(colors)])
            left_labels.append(filtered_di.label)
            left_plotted += 1
            color_idx += 1

        except (KeyError, Exception):
            continue

    if left_plotted == 0:
        plt.close(fig)
        return [TextContent(type="text", text="No left variables could be plotted")]

    # Configure left axis with meaningful label
    ax1.set_xlabel("Time (s)")
    if left_plotted == 1:
        ax1.set_ylabel(left_labels[0])
    else:
        ax1.set_ylabel(", ".join(left_labels) if left_plotted <= 2 else f"{left_labels[0]} and {left_plotted - 1} more")
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    # Plot right axis variables if provided
    right_plotted = 0
    right_labels = []
    if right_variables:
        ax2 = ax1.twinx()  # Create second y-axis

        for var_name in right_variables:
            try:
                di = data[var_name]
                filtered_di = get_data_slice_by_timestamp(di, start_time, end_time)

                if len(filtered_di) == 0:
                    continue

                ts = filtered_di.timestamp_np.astype(np.float64) / 1e3
                val = filtered_di.value_np

                ax2.plot(ts, val, label=filtered_di.label, linestyle='--',
                        color=colors[color_idx % len(colors)])
                right_labels.append(filtered_di.label)
                right_plotted += 1
                color_idx += 1

            except (KeyError, Exception):
                continue

        if right_plotted > 0:
            # Configure right axis with meaningful label
            if right_plotted == 1:
                ax2.set_ylabel(right_labels[0])
            else:
                ax2.set_ylabel(", ".join(right_labels) if right_plotted <= 2 else f"{right_labels[0]} and {right_plotted - 1} more")
            ax2.tick_params(axis='y')

    # Add legends and title
    fig.tight_layout(pad=2.0)
    if right_plotted > 0:
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax1.legend(loc="best")

    # Create descriptive title
    all_labels = left_labels + right_labels
    if len(all_labels) <= 3:
        title = " vs ".join(all_labels)
    else:
        title = f"{all_labels[0]} vs {all_labels[1]} and {len(all_labels) - 2} more"
    plt.title(title)

    # Save plot to BytesIO buffer and encode as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    # Return image content with base64 data
    total_plotted = left_plotted + right_plotted
    return [
        TextContent(
            type="text",
            text=f"Dual-axis graph: {left_plotted} left + {right_plotted} right = {total_plotted} total variable(s)"
        ),
        ImageContent(
            type="image",
            data=image_base64,
            mimeType="image/png"
        )
    ]
