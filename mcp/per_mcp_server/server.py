"""
PER MCP Server

Provides fast, deterministic search for CAN variables from CanDefines.xml.
No AI/embeddings required - uses fuzzy string matching for natural language queries.
"""

import os
import logging
from typing import Any
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource, Content
from mcp.server.stdio import stdio_server

from .utils import CANSearch
from .handlers import TOOL_DEFINITIONS, get_tool_handlers

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("per-telemetry")

# Initialize CAN search engine
can_search = None
local_path = os.getenv('CAN_DEFINES_PATH')

# Try to initialize search engine
try:
    logger.info("Initializing CAN search engine...")
    can_search = CANSearch(
        can_defines_path=local_path,
        use_s3=True  # Will use S3 credentials from environment
    )
    
    if can_search.indexed:
        logger.info(f"✓ CAN search engine ready with {len(can_search.variables)} variables")
    else:
        logger.warning("CAN definitions not available. Search features will be disabled.")
        can_search = None
        
except Exception as e:
    logger.error(f"Failed to initialize CAN search: {e}")
    can_search = None

# Initialize tool handlers
tool_handlers = get_tool_handlers(can_search)


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = []

    if can_search and can_search.indexed:
        resources.extend([
            Resource(
                uri="can://definitions",
                name="CAN Definitions",
                mimeType="application/json",
                description="All CAN variable definitions from CanDefines.xml"
            ),
            Resource(
                uri="can://devices",
                name="CAN Devices",
                mimeType="application/json",
                description="List of all CAN devices"
            ),
        ])
    
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "can://definitions":
        if not can_search:
            return "CAN definitions not available"
        return str([var.to_dict() for var in can_search.variables])
    
    elif uri == "can://devices":
        if not can_search:
            return "CAN definitions not available"
        return str(can_search.list_devices())
    
    else:
        raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    tools = []

    # Add CAN search tools if available
    if can_search and can_search.indexed:
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
    handler = tool_handlers.get(name)

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
