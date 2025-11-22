"""CAN variable search handler."""

from mcp.types import TextContent


# Tool definition for search
TOOL_DEFINITIONS = {
    "search_can_variables": {
        "description": "Search for CAN variables using natural language. Examples: 'battery voltage', 'motor temperature', 'wheel speed'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                },
                "min_score": {
                    "type": "integer",
                    "description": "Minimum match score 0-100 (default: 60)",
                    "default": 60
                }
            },
            "required": ["query"]
        }
    }
}


def handle_search_can_variables(can_search, arguments: dict) -> list[TextContent]:
    """Handle search_can_variables tool."""
    if not can_search:
        return [TextContent(type="text", text="CAN search not available")]

    query = arguments.get("query")
    limit = arguments.get("limit", 10)
    min_score = arguments.get("min_score", 60)

    results = can_search.search(query, limit=limit, min_score=min_score)

    if not results:
        return [TextContent(
            type="text",
            text=f"No CAN variables found matching '{query}'"
        )]

    # Format results
    output = f"Found {len(results)} CAN variable(s) matching '{query}':\n\n"
    for result in results:
        output += f"**Match Score: {result['score']}/100**\n"
        output += result['formatted']
        output += "\n"

    return [TextContent(type="text", text=output)]
