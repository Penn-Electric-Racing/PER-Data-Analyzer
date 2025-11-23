#!/usr/bin/env python3
"""
Web Frontend for PER MCP Server

Flask-based web interface with chat UI and file upload for CAN data analysis.
"""

import os
import asyncio
import json
import logging
import threading
import uuid
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

# Global dictionary to store chat objects per session
# Key: session ID, Value: {'chat': chat_object, 'model': model_object}
_chat_sessions = {}
_chat_history = {}
_chat_lock = threading.Lock()  # For thread-safe access


WELCOME_MESSAGE = {
    'type': 'assistant',
    'text': 'Welcome to CHUKKA - Car Hardware Unified Kinematics & Knowledge Analyzer! Upload a CSV file to get started, or ask questions about the default dataset.'
}


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


def _ensure_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def _get_history_snapshot(session_id):
    with _chat_lock:
        history = _chat_history.setdefault(session_id, [WELCOME_MESSAGE.copy()])
        return [dict(msg) for msg in history]


def _append_history_message(session_id, message):
    with _chat_lock:
        history = _chat_history.setdefault(session_id, [WELCOME_MESSAGE.copy()])
        history.append(message)


def _reset_history(session_id, initial_message=None):
    base_message = initial_message or WELCOME_MESSAGE.copy()
    with _chat_lock:
        _chat_history[session_id] = [base_message]


@app.route('/')
def index():
    """Serve the main chat interface."""
    session_id = _ensure_session_id()

    if 'active_csv_path' not in session:
        session['active_csv_path'] = None
    if 'active_file' not in session:
        session['active_file'] = None

    history = _get_history_snapshot(session_id)

    return render_template('index.html',
                         messages=history,
                         active_file=session['active_file'])


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    session_id = _ensure_session_id()

    if 'file' not in request.files:
        _append_history_message(session_id, {'type': 'error', 'text': 'No file provided'})
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        _append_history_message(session_id, {'type': 'error', 'text': 'No file selected'})
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        _append_history_message(session_id, {'type': 'error', 'text': 'Only CSV files are allowed'})
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

    _append_history_message(session_id, {
        'type': 'assistant',
        'text': f'Successfully uploaded {original_name}. You can now analyze this data!'
    })

    return redirect(url_for('index'))


@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history and start fresh."""
    session_id = _ensure_session_id()
    _reset_history(session_id, {'type': 'assistant', 'text': 'Chat history cleared! How can I help you?'})

    # Clear the chat object for this session
    with _chat_lock:
        if session_id in _chat_sessions:
            del _chat_sessions[session_id]
            logger.info(f"Cleared chat session: {session_id[:8]}...")

    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - communicate with MCP server."""
    data = request.get_json()
    message = data.get('message', '') if data else ''

    if not message:
        error_response = {'type': 'error', 'text': 'No message provided'}
        session_id = _ensure_session_id()
        _append_history_message(session_id, error_response)
        return jsonify(error_response)

    # Get or create a unique session ID for this user and store message server-side
    session_id = _ensure_session_id()
    _append_history_message(session_id, {'type': 'user', 'text': message})

    # Run async MCP communication
    try:
        response = asyncio.run(send_to_mcp(message, session.get('active_csv_path'), session_id))
        _append_history_message(session_id, response)
        # Return just the response (not the user message)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error communicating with MCP: {e}", exc_info=True)
        error_response = {'type': 'error', 'text': f'Error: {str(e)}'}
        _append_history_message(session_id, error_response)
        return jsonify(error_response)


async def send_to_mcp(user_message: str, active_csv_path: str = None, session_id: str = None):
    """
    Send message to MCP server and get response.
    Uses Gemini to interpret user intent and call appropriate tools.

    Args:
        user_message: The user's message
        active_csv_path: Path to active CSV file
        session_id: Unique session ID to maintain chat history
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

                # Get or create chat object for this session
                chat = None
                model = None

                if session_id:
                    with _chat_lock:
                        if session_id in _chat_sessions:
                            # Reuse existing chat object
                            chat = _chat_sessions[session_id]['chat']
                            model = _chat_sessions[session_id]['model']
                            logger.info(f"Reusing chat session: {session_id[:8]}...")

                if chat is None:
                    # Create new chat session
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

                    # Store in global dictionary
                    if session_id:
                        with _chat_lock:
                            _chat_sessions[session_id] = {'chat': chat, 'model': model}
                        logger.info(f"Created new chat session: {session_id[:8]}...")

                gemini_response = chat.send_message(user_message)

                # Handle function calls
                result_images = []  # Preserve all generated images in order
                while True:
                    function_call = _get_next_function_call(gemini_response)
                    if not function_call:
                        break

                    tool_name = function_call.name
                    tool_args = _convert_proto_to_native(dict(function_call.args))

                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

                    # Call MCP tool
                    result = await session.call_tool(tool_name, tool_args)

                    # Extract content (text and images)
                    result_text = ""
                    result_image_data = []
                    for content in result.content:
                        if hasattr(content, 'text'):
                            result_text += content.text
                        elif content.type == 'image':
                            result_image_data.append(content.data)

                    if result_image_data:
                        result_images.extend(result_image_data)

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

                # Build final response text from all non-function-call parts
                assistant_text = _extract_text_from_response(gemini_response)
                if not assistant_text:
                    return {
                        'type': 'error',
                        'text': 'Assistant did not return a textual response.'
                    }

                response_dict = {
                    'type': 'assistant',
                    'text': assistant_text
                }

                if result_images:
                    response_dict['images'] = result_images
                    response_dict['image'] = result_images[0]  # Back-compat for single image handling

                return response_dict

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


def _get_next_function_call(response):
    """Return the next function call part from a Gemini response, if any."""
    candidates = getattr(response, 'candidates', [])
    if not candidates:
        return None

    content = getattr(candidates[0], 'content', None)
    if not content:
        return None

    for part in getattr(content, 'parts', []) or []:
        function_call = getattr(part, 'function_call', None)
        if function_call:
            return function_call

    return None


def _extract_text_from_response(response) -> str:
    """Safely extract concatenated text parts from a Gemini response."""
    candidates = getattr(response, 'candidates', [])
    if not candidates:
        return ""

    content = getattr(candidates[0], 'content', None)
    if not content:
        return ""

    texts = []
    for part in getattr(content, 'parts', []) or []:
        text_value = getattr(part, 'text', None)
        if text_value:
            texts.append(text_value)

    if texts:
        return "\n".join(texts).strip()

    # Fallback to Gemini's text helper if no part.text entries exist
    try:
        return response.text
    except ValueError:
        return ""


if __name__ == '__main__':
    print("=" * 60)
    print("PER MCP Server - Web Frontend")
    print("=" * 60)
    print()
    print("Starting web server on http://localhost:5001")
    print("Open your browser and navigate to http://localhost:5001")
    print()

    app.run(debug=True, host='0.0.0.0', port=5001)
