from flask import Blueprint, render_template, session, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

from ..services.chat_service import ensure_session_id, get_history_snapshot, append_history_message, reset_history
from ..services.file_service import allowed_file

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Serve the main chat interface."""
    session_id = ensure_session_id(session)

    if 'active_csv_path' not in session:
        session['active_csv_path'] = None
    if 'active_file' not in session:
        session['active_file'] = None

    history = get_history_snapshot(session_id)

    return render_template('index.html',
                         messages=history,
                         active_file=session['active_file'])

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    session_id = ensure_session_id(session)

    if 'file' not in request.files:
        append_history_message(session_id, {'type': 'error', 'text': 'No file provided'})
        return redirect(url_for('main.index'))

    file = request.files['file']

    if file.filename == '':
        append_history_message(session_id, {'type': 'error', 'text': 'No file selected'})
        return redirect(url_for('main.index'))

    if not allowed_file(file.filename):
        append_history_message(session_id, {'type': 'error', 'text': 'Only CSV files are allowed'})
        return redirect(url_for('main.index'))

    # Save file with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = secure_filename(file.filename)
    filename = f"{timestamp}_{original_name}"
    filepath = current_app.config['UPLOAD_FOLDER'] / filename

    file.save(str(filepath))

    # Set as active CSV in session (relative path from mcp directory)
    relative_path = f"uploads/{filename}"
    session['active_csv_path'] = relative_path
    session['active_file'] = original_name

    logger.info(f"Uploaded and activated CSV: {filepath}")

    append_history_message(session_id, {
        'type': 'assistant',
        'text': f'Successfully uploaded {original_name}. You can now analyze this data!'
    })

    return redirect(url_for('main.index'))

@main_bp.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history and start fresh."""
    session_id = ensure_session_id(session)
    reset_history(session_id, {'type': 'assistant', 'text': 'Chat history cleared! How can I help you?'})

    return redirect(url_for('main.index'))
