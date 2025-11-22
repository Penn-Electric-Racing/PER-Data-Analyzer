"""CAN variable query handlers."""

from mcp.types import TextContent


# Tool definitions for query operations
TOOL_DEFINITIONS = {
    "get_can_variable": {
        "description": "Get a specific CAN variable by its full path (e.g., 'pdu.sensors.batCurrent')",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full variable path (e.g., 'pdu.sensors.batCurrent')"
                }
            },
            "required": ["path"]
        }
    },
    "list_device_variables": {
        "description": "List all variables for a specific device (e.g., 'pdu', 'pcm', 'ams')",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Device name or identifier"
                }
            },
            "required": ["device"]
        }
    },
    "list_can_devices": {
        "description": "List all CAN devices with variable counts",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
}


def handle_get_can_variable(can_search, arguments: dict) -> list[TextContent]:
    """Handle get_can_variable tool."""
    if not can_search:
        return [TextContent(type="text", text="CAN search not available")]

    path = arguments.get("path")
    var = can_search.get_by_path(path)

    if not var:
        return [TextContent(
            type="text",
            text=f"Variable not found: {path}"
        )]

    return [TextContent(type="text", text=var.format_result())]


def handle_list_device_variables(can_search, arguments: dict) -> list[TextContent]:
    """Handle list_device_variables tool."""
    if not can_search:
        return [TextContent(type="text", text="CAN search not available")]

    device = arguments.get("device")
    variables = can_search.get_by_device(device)

    if not variables:
        return [TextContent(
            type="text",
            text=f"No variables found for device: {device}"
        )]

    output = f"Found {len(variables)} variable(s) for device '{device}':\n\n"
    for var in variables[:50]:  # Limit to 50
        output += var.format_result() + "\n"

    if len(variables) > 50:
        output += f"\n... and {len(variables) - 50} more"

    return [TextContent(type="text", text=output)]


def handle_list_can_devices(can_search, arguments: dict) -> list[TextContent]:
    """Handle list_can_devices tool."""
    if not can_search:
        return [TextContent(type="text", text="CAN search not available")]

    devices = can_search.list_devices()

    output = "CAN Devices:\n\n"
    for device in devices:
        output += f"- **{device['name']}** (ID: {device['id']}): {device['count']} variables\n"

    return [TextContent(type="text", text=output)]
