"""
PER MCP Server

Provides CAN data analysis tools using perda library.
Tools directly parse CSV files and use perda's built-in search and analysis functions.
"""

import os
import logging
from typing import Any
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, Content
from mcp.server.stdio import stdio_server

from .handlers import TOOL_DEFINITIONS, TOOL_HANDLERS

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("per-telemetry")

# Tool handlers are loaded at import time from handlers module
logger.info(f"✓ MCP server ready with {len(TOOL_HANDLERS)} tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    tools = []

    for tool_name, tool_def in TOOL_DEFINITIONS.items():
        tools.append(Tool(
            name=tool_name,
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        ))

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[Content]:
    """Handle tool calls."""
    handler = TOOL_HANDLERS.get(name)

    if not handler:
        raise ValueError(f"Unknown tool: {name}")

    return handler(arguments)


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
