import os
import asyncio
import logging
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .can_service import load_can_defines

logger = logging.getLogger(__name__)

# Global dictionary to store chat objects per session
# Key: session ID, Value: {'chat': chat_object, 'model': model_object}
_chat_sessions = {}
_chat_history = {}
_chat_lock = threading.Lock()  # For thread-safe access

WELCOME_MESSAGE = {
    'type': 'assistant',
    'text': 'Welcome to CHUKKA - Car Hardware Unified Kinematics & Knowledge Analyzer! Upload a CSV file to get started, or ask questions about the default dataset.'
}

def ensure_session_id(session: Dict) -> str:
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_history_snapshot(session_id: str) -> List[Dict[str, Any]]:
    with _chat_lock:
        history = _chat_history.setdefault(session_id, [WELCOME_MESSAGE.copy()])
        return [dict(msg) for msg in history]

def append_history_message(session_id: str, message: Dict[str, Any]):
    with _chat_lock:
        history = _chat_history.setdefault(session_id, [WELCOME_MESSAGE.copy()])
        history.append(message)

def reset_history(session_id: str, initial_message: Optional[Dict[str, Any]] = None):
    base_message = initial_message or WELCOME_MESSAGE.copy()
    with _chat_lock:
        _chat_history[session_id] = [base_message]
    
    # Clear the chat object for this session
    with _chat_lock:
        if session_id in _chat_sessions:
            del _chat_sessions[session_id]
            logger.info(f"Cleared chat session: {session_id[:8]}...")

async def send_to_mcp(user_message: str, active_csv_path: str = None, session_id: str = None) -> Dict[str, Any]:
    """
    Send message to MCP server and get response.
    Uses Gemini to interpret user intent and call appropriate tools.

    Args:
        user_message: The user's message
        active_csv_path: Path to active CSV file
        session_id: Unique session ID to maintain chat history
    """
    # MCP server parameters
    # Note: We assume the server module is available in the python path
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "per_mcp_server"],
        cwd=str(Path(__file__).parent.parent.parent), # mcp root
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
                    function_calls = _get_function_calls(gemini_response)
                    
                    # Check for empty response with STOP reason - retry once
                    if not function_calls:
                        # Check if there's text content
                        text_content = _extract_text_from_response(gemini_response)
                        if not text_content:
                            # Check finish reason
                            candidates = getattr(gemini_response, 'candidates', [])
                            if candidates:
                                finish_reason = candidates[0].finish_reason
                                # Check for STOP (1)
                                if finish_reason == 1 or str(finish_reason) == 'STOP':
                                    logger.warning("Empty response with STOP. Retrying once...")
                                    # Send retry message
                                    gemini_response = chat.send_message("You returned an empty response. Please answer the user's request.")
                                    # Continue loop to check if retry produced function calls or text
                                    continue
                        # No function calls and either has text or not a STOP reason - exit loop
                        break

                    function_responses = []
                    for function_call in function_calls:
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
                        
                        # Add to function responses
                        function_responses.append(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text}
                                )
                            )
                        )

                    # Send all results back to Gemini
                    gemini_response = chat.send_message(
                        genai.protos.Content(parts=function_responses)
                    )

                # Build final response text from all non-function-call parts
                assistant_text = _extract_text_from_response(gemini_response)

                if not assistant_text and not result_images:
                    # Log detailed info about why response is still empty
                    candidates = getattr(gemini_response, 'candidates', [])
                    if candidates:
                        finish_reason = candidates[0].finish_reason
                        logger.warning(f"Gemini response empty after processing. Finish reason: {finish_reason}")
                        logger.warning(f"Full candidate: {candidates[0]}")

                        # Handle specific finish reasons (using integer values for robustness)
                        # 3=SAFETY, 4=RECITATION, etc.
                        if hasattr(finish_reason, 'name'):
                             reason_name = finish_reason.name
                        else:
                             reason_name = str(finish_reason)

                        if reason_name == 'SAFETY' or finish_reason == 3:
                            return {
                                'type': 'error',
                                'text': 'I cannot answer that due to safety filters. Please try rephrasing.'
                            }
                        elif reason_name == 'RECITATION' or finish_reason == 4:
                            return {
                                'type': 'error',
                                'text': 'I cannot answer that due to recitation checks. Please try rephrasing.'
                            }
                    else:
                        logger.warning("Gemini response contained no candidates.")
                        logger.warning(f"Full response: {gemini_response}")

                    return {
                        'type': 'error',
                        'text': 'Assistant did not return a textual response. Please check server logs for details.'
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

def _get_function_calls(response):
    """Return all function calls from a Gemini response."""
    candidates = getattr(response, 'candidates', [])
    if not candidates:
        return []

    content = getattr(candidates[0], 'content', None)
    if not content:
        return []

    function_calls = []
    for part in getattr(content, 'parts', []) or []:
        function_call = getattr(part, 'function_call', None)
        if function_call:
            function_calls.append(function_call)

    return function_calls

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
