#!/usr/bin/env python3
"""
Web Frontend for PER MCP Server

Flask-based web interface with chat UI and file upload for CAN data analysis.
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import google.generativeai as genai

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Setup
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure upload directory exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global cache for CAN variable definitions from XML
_can_defines_cache = None


def load_can_defines():
    """
    Parse GeneratedCanIds.xml and cache variable definitions.

    Returns condensed list of all CAN variables with descriptions for LLM context.
    """
    global _can_defines_cache

    # Return cached version if available
    if _can_defines_cache is not None:
        return _can_defines_cache

    # Check if CAN_DEFINES_PATH is configured
    can_defines_path = os.getenv("CAN_DEFINES_PATH")
    if not can_defines_path:
        logger.warning("CAN_DEFINES_PATH not configured - LLM won't have variable descriptions")
        _can_defines_cache = ""
        return ""

    if not os.path.exists(can_defines_path):
        logger.warning(f"CAN defines file not found: {can_defines_path}")
        _can_defines_cache = ""
        return ""

    try:
        from lxml import etree

        logger.info(f"Loading CAN definitions from: {can_defines_path}")
        tree = etree.parse(can_defines_path)
        root = tree.getroot()

        # Parse all variable definitions
        definitions = []
        for can_id_elem in root.findall('CanId'):
            for value_elem in can_id_elem.findall('Value'):
                path = value_elem.get('AccessString', '')
                name = value_elem.get('Name', '')
                desc = value_elem.get('Description', '')
                unit = value_elem.get('Unit', '')

                if path:  # Only include if we have a path
                    # Format: "path: Name - Description (unit)"
                    line = f"{path}: {name}"
                    if desc:
                        line += f" - {desc}"
                    if unit:
                        line += f" ({unit})"
                    definitions.append(line)

        # Cache the formatted list
        _can_defines_cache = "\n".join(definitions)
        logger.info(f"✓ Loaded {len(definitions)} CAN variable definitions")
        return _can_defines_cache

    except ImportError:
        logger.error("lxml not installed - cannot parse CAN defines XML")
        _can_defines_cache = ""
        return ""
    except Exception as e:
        logger.error(f"Failed to parse CAN defines: {e}", exc_info=True)
        _can_defines_cache = ""
        return ""


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Serve the main chat interface."""
    if 'messages' not in session:
        session['messages'] = [{'type': 'assistant', 'text': 'Welcome to CHUKKA - Car Hardware Unified Kinematics & Knowledge Analyzer! Upload a CSV file to get started, or ask questions about the default dataset.'}]
    if 'active_csv_path' not in session:
        session['active_csv_path'] = None
    if 'active_file' not in session:
        session['active_file'] = None

    return render_template('index.html',
                         messages=session['messages'],
                         active_file=session['active_file'])


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    if 'file' not in request.files:
        messages = session.get('messages', [])
        messages.append({'type': 'error', 'text': 'No file provided'})
        session['messages'] = messages
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        messages = session.get('messages', [])
        messages.append({'type': 'error', 'text': 'No file selected'})
        session['messages'] = messages
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        messages = session.get('messages', [])
        messages.append({'type': 'error', 'text': 'Only CSV files are allowed'})
        session['messages'] = messages
        return redirect(url_for('index'))

    # Save file with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = secure_filename(file.filename)
    filename = f"{timestamp}_{original_name}"
    filepath = app.config['UPLOAD_FOLDER'] / filename

    file.save(str(filepath))

    # Set as active CSV in session (relative path from mcp directory)
    relative_path = f"uploads/{filename}"
    session['active_csv_path'] = relative_path
    session['active_file'] = original_name

    logger.info(f"Uploaded and activated CSV: {filepath}")

    messages = session.get('messages', [])
    messages.append({
        'type': 'assistant',
        'text': f'Successfully uploaded {original_name}. You can now analyze this data!'
    })
    session['messages'] = messages

    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - communicate with MCP server."""
    data = request.get_json()
    message = data.get('message', '') if data else ''

    if not message:
        error_response = {'type': 'error', 'text': 'No message provided'}
        # Save to session
        messages = session.get('messages', [])
        messages.append(error_response)
        session['messages'] = messages
        return jsonify(error_response)

    # Add user message to session
    messages = session.get('messages', [])
    messages.append({'type': 'user', 'text': message})
    session['messages'] = messages

    # Run async MCP communication
    try:
        response = asyncio.run(send_to_mcp(message, session.get('active_csv_path')))
        # Save to session
        messages = session.get('messages', [])
        messages.append(response)
        session['messages'] = messages
        # Return just the response (not the user message)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error communicating with MCP: {e}", exc_info=True)
        error_response = {'type': 'error', 'text': f'Error: {str(e)}'}
        # Save to session
        messages = session.get('messages', [])
        messages.append(error_response)
        session['messages'] = messages
        return jsonify(error_response)


async def send_to_mcp(user_message: str, active_csv_path: str = None):
    """
    Send message to MCP server and get response.
    Uses Gemini to interpret user intent and call appropriate tools.
    """
    # MCP server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "per_mcp_server"],
        cwd=str(Path(__file__).parent),
        env={
            **os.environ.copy(),  # Include all current env vars
            "ACTIVE_CSV_PATH": active_csv_path or "temp/16thMay13-52.csv",
            "S3_BUCKET": os.getenv("S3_BUCKET", ""),
            "S3_KEY": os.getenv("S3_KEY", ""),
            "S3_SECRET": os.getenv("S3_SECRET", ""),
            "S3_ENDPOINT": os.getenv("S3_ENDPOINT", ""),
        }
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools
                tools_response = await session.list_tools()
                available_tools = {tool.name: tool for tool in tools_response.tools}

                # Use Gemini to interpret user message and call tools
                import google.generativeai as genai

                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    return {
                        'type': 'error',
                        'text': 'GEMINI_API_KEY not configured. Please set it in .env file.'
                    }

                genai.configure(api_key=gemini_api_key)

                # Load CAN variable definitions for LLM context
                can_defines = load_can_defines()

                # Build system instruction
                system_instruction = """You are a helpful assistant for analyzing CAN bus data from electric race cars.

When users ask questions, use the available tools to help them. Be concise and direct.
For graphs, explain what the graph shows briefly."""

                # Append CAN definitions if available
                if can_defines:
                    system_instruction += f"""

Available CAN Variables (from CanDefines.xml):
{can_defines}

Note: Not all variables may be present in the current CSV log file.
Use search_can_variables to see what's actually available in this specific dataset.
Use get_can_variable_info for detailed statistics about a variable."""

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
                                            k: _convert_schema_to_proto(v)
                                            for k, v in tool.inputSchema.get('properties', {}).items()
                                        },
                                        required=tool.inputSchema.get('required', [])
                                    )
                                )
                                for tool in available_tools.values()
                            ]
                        )
                    ],
                    system_instruction=system_instruction
                )

                chat = model.start_chat()
                gemini_response = chat.send_message(user_message)

                # Handle function calls
                responses = []
                result_image = None  # Initialize before loop
                while gemini_response.candidates[0].content.parts[0].function_call:
                    function_call = gemini_response.candidates[0].content.parts[0].function_call
                    tool_name = function_call.name
                    tool_args = _convert_proto_to_native(dict(function_call.args))

                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

                    # Call MCP tool
                    result = await session.call_tool(tool_name, tool_args)

                    # Extract content (text and images)
                    result_text = ""
                    result_image = None
                    for content in result.content:
                        if hasattr(content, 'text'):
                            result_text += content.text
                        elif content.type == 'image':
                            result_image = content.data

                    # Send result back to Gemini
                    gemini_response = chat.send_message(
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

                # Get final response
                if not gemini_response.candidates[0].content.parts[0].function_call:
                    response_dict = {
                        'type': 'assistant',
                        'text': gemini_response.text
                    }
                    # Add image if present
                    if result_image:
                        response_dict['image'] = result_image
                    return response_dict
                else:
                    return {
                        'type': 'error',
                        'text': 'Unexpected response from assistant'
                    }

    except Exception as e:
        logger.error(f"MCP communication error: {e}", exc_info=True)
        return {
            'type': 'error',
            'text': f'Error: {str(e)}'
        }


def _convert_schema_to_proto(schema_prop):
    """Convert JSON schema property to Gemini proto schema."""
    prop_type = schema_prop.get('type', 'string')

    type_map = {
        'string': genai.protos.Type.STRING,
        'integer': genai.protos.Type.INTEGER,
        'number': genai.protos.Type.NUMBER,
        'boolean': genai.protos.Type.BOOLEAN,
        'array': genai.protos.Type.ARRAY,
        'object': genai.protos.Type.OBJECT,
    }

    proto_type = type_map.get(prop_type, genai.protos.Type.STRING)

    proto_schema = genai.protos.Schema(
        type=proto_type,
        description=schema_prop.get('description', '')
    )

    # Handle array items
    if prop_type == 'array' and 'items' in schema_prop:
        items_schema = schema_prop['items']
        proto_schema.items = genai.protos.Schema(
            type=type_map.get(items_schema.get('type', 'string'), genai.protos.Type.STRING),
            description=items_schema.get('description', '')
        )

    # Handle enum
    if 'enum' in schema_prop:
        proto_schema.enum = schema_prop['enum']

    return proto_schema


def _convert_proto_to_native(obj):
    """Convert protobuf objects to native Python types."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): _convert_proto_to_native(v) for k, v in obj.items()}
    elif hasattr(obj, 'items'):
        return {str(k): _convert_proto_to_native(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return [_convert_proto_to_native(item) for item in obj]
    else:
        return str(obj)


if __name__ == '__main__':
    print("=" * 60)
    print("PER MCP Server - Web Frontend")
    print("=" * 60)
    print()
    print("Starting web server on http://localhost:5001")
    print("Open your browser and navigate to http://localhost:5001")
    print()

    app.run(debug=True, host='0.0.0.0', port=5001)
