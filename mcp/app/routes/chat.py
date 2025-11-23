from flask import Blueprint, request, session, jsonify
import asyncio
import logging

from ..services.chat_service import ensure_session_id, append_history_message, send_to_mcp

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages - communicate with MCP server."""
    data = request.get_json()
    message = data.get('message', '') if data else ''

    if not message:
        error_response = {'type': 'error', 'text': 'No message provided'}
        session_id = ensure_session_id(session)
        append_history_message(session_id, error_response)
        return jsonify(error_response)

    # Get or create a unique session ID for this user and store message server-side
    session_id = ensure_session_id(session)
    append_history_message(session_id, {'type': 'user', 'text': message})

    # Run async MCP communication
    try:
        response = asyncio.run(send_to_mcp(message, session.get('active_csv_path'), session_id))
        append_history_message(session_id, response)
        # Return just the response (not the user message)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error communicating with MCP: {e}", exc_info=True)
        error_response = {'type': 'error', 'text': f'Error: {str(e)}'}
        append_history_message(session_id, error_response)
        return jsonify(error_response)
