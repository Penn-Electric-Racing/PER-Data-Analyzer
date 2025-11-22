#!/usr/bin/env python3
"""
Quick Gemini Client for PER MCP Server

This script connects Gemini to your MCP server so you can query CAN data using natural language.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as e:
    print("❌ Missing dependencies. Please install:")
    print("   pip install google-generativeai mcp")
    sys.exit(1)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not set in .env file")
    print("   Get your API key from: https://aistudio.google.com/app/apikey")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


def format_can_response(text: str) -> str:
    """
    Format CAN variable responses in a standardized way.
    Looks for CAN ID patterns and formats them consistently.
    """
    import re
    
    # Check if this looks like a CAN variable response
    if 'CAN ID:' not in text and 'can_id' not in text.lower():
        return text
    
    # Try to parse as JSON if it contains variable data
    try:
        # Look for JSON-like structures in the text
        if '{' in text and '"can_id"' in text:
            lines = text.split('\n')
            formatted_lines = []
            current_var = {}
            in_json = False
            
            for line in lines:
                # Try to extract variable info
                if '"name"' in line:
                    match = re.search(r'"name":\s*"([^"]+)"', line)
                    if match:
                        current_var['name'] = match.group(1)
                if '"path"' in line:
                    match = re.search(r'"path":\s*"([^"]+)"', line)
                    if match:
                        current_var['path'] = match.group(1)
                if '"can_id"' in line:
                    match = re.search(r'"can_id":\s*"([^"]+)"', line)
                    if match:
                        current_var['can_id'] = match.group(1)
                if '"type"' in line:
                    match = re.search(r'"type":\s*"([^"]+)"', line)
                    if match:
                        current_var['type'] = match.group(1)
                if '"units"' in line:
                    match = re.search(r'"units":\s*"([^"]+)"', line)
                    if match and match.group(1):
                        current_var['units'] = match.group(1)
                if '"description"' in line:
                    match = re.search(r'"description":\s*"([^"]+)"', line)
                    if match and match.group(1):
                        current_var['description'] = match.group(1)
                
                # Check if we've completed a variable
                if current_var.get('name') and current_var.get('can_id') and ('}' in line or len(current_var) >= 4):
                    formatted_lines.append(_format_single_variable(current_var))
                    current_var = {}
            
            if formatted_lines:
                return '\n'.join(formatted_lines)
    except:
        pass
    
    # If JSON parsing didn't work, just ensure consistent formatting
    # Replace various CAN ID formats with standard format
    text = re.sub(r'CAN\s*ID\s*:\s*(\d+)', r'[CAN:0x\1]', text, flags=re.IGNORECASE)
    
    return text


def _format_single_variable(var_data: dict) -> str:
    """Format a single CAN variable in standardized format."""
    name = var_data.get('name', 'Unknown')
    path = var_data.get('path', '')
    can_id = var_data.get('can_id', 'N/A')
    var_type = var_data.get('type', '')
    units = var_data.get('units', '')
    description = var_data.get('description', '')
    
    # Convert CAN ID to hex if it's a number
    try:
        can_id_int = int(can_id)
        can_id_hex = f"0x{can_id_int:03X}"
    except:
        can_id_hex = can_id
    
    # Build formatted output
    output = f"• {name}"
    if path:
        output += f" ({path})"
    output += f"\n  CAN ID: {can_id_hex}"
    if var_type:
        output += f" | Type: {var_type}"
        if units:
            output += f" [{units}]"
    if description:
        output += f"\n  {description}"
    
    return output


async def main():
    print("=" * 60)
    print("PER MCP Server - Gemini Client")
    print("=" * 60)
    print()
    
    # Server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "per_mcp_server"],
        cwd=str(Path(__file__).parent),
        env={
            "S3_BUCKET": os.getenv("S3_BUCKET", ""),
            "S3_KEY": os.getenv("S3_KEY", ""),
            "S3_SECRET": os.getenv("S3_SECRET", ""),
            "S3_ENDPOINT": os.getenv("S3_ENDPOINT", ""),
        }
    )
    
    print("Connecting to MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("✓ Connected to MCP server")
            print()
            
            # List available tools
            tools_response = await session.list_tools()
            available_tools = {tool.name: tool for tool in tools_response.tools}
            
            print(f"Available tools: {len(available_tools)}")
            for tool_name, tool in available_tools.items():
                print(f"  • {tool_name}: {tool.description}")
            print()
            print("=" * 60)
            print()
            
            # Initialize Gemini with function calling and system instructions
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                tools=[
                    genai.protos.Tool(
                        function_declarations=[
                            genai.protos.FunctionDeclaration(
                                name=tool.name,
                                description=tool.description,
                                parameters=genai.protos.Schema(
                                    type=genai.protos.Type.OBJECT,
                                    properties={
                                        k: genai.protos.Schema(
                                            type=genai.protos.Type.STRING,
                                            description=v.get('description', '')
                                        )
                                        for k, v in tool.inputSchema.get('properties', {}).items()
                                    },
                                    required=tool.inputSchema.get('required', [])
                                )
                            )
                            for tool in available_tools.values()
                        ]
                    )
                ],
                system_instruction="""You are a concise technical assistant for CAN bus data queries.

IMPORTANT RESPONSE GUIDELINES:
- Be direct and to the point
- Present results in a clean, structured format
- Only include the most relevant information
- Avoid unnecessary explanations or rambling

When presenting CAN variables, use this EXACT format for each variable:

• [Variable Name] (path.to.variable)
  CAN ID: 0x[HEX] | Type: [type] [units if applicable]
  [Brief description if relevant]

Example:
• AIR Close Time (ams.airCloseTime)
  CAN ID: 0x100 | Type: float
  
• DCDC Temperature (ams.dcdc.temp)
  CAN ID: 0x106 | Type: float [°C]
  Temperature of the DCDC converter

For multiple results:
- List up to 5 most relevant matches
- If more exist, mention "...and X more matches"
- Keep each entry on 2-3 lines maximum

Keep responses professional and brief."""
            )
            
            chat = model.start_chat()
            
            print("Chat with Gemini about your CAN data!")
            print("Examples:")
            print('  "Find battery voltage variables"')
            print('  "What temperature sensors are on the AMS?"')
            print('  "List all CAN devices"')
            print()
            print("Type 'exit' or 'quit' to end")
            print("-" * 60)
            print()
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        print("\nGoodbye! 👋")
                        break
                    
                    # Send to Gemini
                    response = chat.send_message(user_input)
                    
                    # Check if Gemini wants to call a function
                    while response.candidates[0].content.parts[0].function_call:
                        function_call = response.candidates[0].content.parts[0].function_call
                        tool_name = function_call.name
                        tool_args = dict(function_call.args)
                        
                        print(f"\n[Calling tool: {tool_name}]")
                        
                        # Call the MCP tool
                        result = await session.call_tool(tool_name, tool_args)
                        
                        # Extract text content
                        result_text = ""
                        for content in result.content:
                            if hasattr(content, 'text'):
                                result_text += content.text
                        
                        # Send result back to Gemini
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[
                                    genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"result": result_text}
                                        )
                                    )
                                ]
                            )
                        )
                    
                    # Display Gemini's response
                    formatted_response = format_can_response(response.text)
                    print(f"\nGemini: {formatted_response}")
                    print()
                
                except KeyboardInterrupt:
                    print("\n\nGoodbye! 👋")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
