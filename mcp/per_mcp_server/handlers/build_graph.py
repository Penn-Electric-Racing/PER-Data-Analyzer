from mcp.types import TextContent

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
            }
        }
    }
}

