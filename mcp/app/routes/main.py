from flask import Blueprint, render_template, session, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import os

from ..services.chat_service import ensure_session_id, get_history_snapshot, append_history_message, reset_history
from ..services.file_service import allowed_file

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Serve the main chat interface."""
    # Check authentication
    if not session.get('authenticated'):
        return redirect(url_for('auth.login'))
    
    session_id = ensure_session_id(session)

    if 'active_csv_path' not in session:
        session['active_csv_path'] = None
    if 'active_file' not in session:
        session['active_file'] = None

    history = get_history_snapshot(session_id)
    
    # Check if using a different model
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    show_model_warning = model_name != 'gemini-2.5-flash'

    return render_template('index.html',
                         messages=history,
                         active_file=session['active_file'],
                         model_name=model_name,
                         show_model_warning=show_model_warning)

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    # Check authentication
    if not session.get('authenticated'):
        return redirect(url_for('auth.login'))
    
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
    # Check authentication
    if not session.get('authenticated'):
        return redirect(url_for('auth.login'))
    
    session_id = ensure_session_id(session)
    reset_history(session_id, {'type': 'assistant', 'text': 'Chat history cleared! How can I help you?'})

    return redirect(url_for('main.index'))

@main_bp.route('/api/logs', methods=['GET'])
def list_logs():
    """List all log files from S3 bucket."""
    # Check authentication
    if not session.get('authenticated'):
        return {'error': 'Unauthorized'}, 401
    
    from ..services.s3_service import s3_service
    
    files, status_code = s3_service.list_log_files()
    
    if status_code == 200:
        return {'logs': files}, 200
    else:
        return {'error': 'Failed to fetch logs from S3'}, status_code

@main_bp.route('/api/logs/select', methods=['POST'])
def select_log():
    """Download and activate a log file from S3."""
    # Check authentication
    if not session.get('authenticated'):
        return {'error': 'Unauthorized'}, 401

    session_id = ensure_session_id(session)

    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return {'error': 'Filename required'}, 400

    from ..services.s3_service import s3_service

    # Download file from S3 to uploads directory
    upload_dir = current_app.config['UPLOAD_FOLDER']
    local_path = s3_service.download_log_file(filename, upload_dir)

    if local_path is None:
        append_history_message(session_id, {
            'type': 'error',
            'text': f'Failed to download {filename} from S3'
        })
        return {'error': 'Download failed'}, 500

    # Set as active CSV in session
    relative_path = str(local_path.relative_to(upload_dir.parent))
    session['active_csv_path'] = relative_path
    session['active_file'] = filename

    logger.info(f"Downloaded and activated S3 log: {filename}")

    append_history_message(session_id, {
        'type': 'assistant',
        'text': f'Successfully loaded {filename} from S3. You can now analyze this data!'
    })

    return {'success': True, 'filename': filename}, 200

@main_bp.route('/api/local-logs', methods=['GET'])
def list_local_logs():
    """List all log files from local logs directory."""
    # Check authentication
    if not session.get('authenticated'):
        return {'error': 'Unauthorized'}, 401

    from ..services.local_logs_service import local_logs_service

    logs = local_logs_service.list_local_logs()

    return {'logs': logs}, 200

@main_bp.route('/api/local-logs/select', methods=['POST'])
def select_local_log():
    """Activate a log file from local logs directory."""
    # Check authentication
    if not session.get('authenticated'):
        return {'error': 'Unauthorized'}, 401

    session_id = ensure_session_id(session)

    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath:
        return {'error': 'Filepath required'}, 400

    from ..services.local_logs_service import local_logs_service

    # Validate that file exists
    abs_path = local_logs_service.get_absolute_path(filepath)

    if abs_path is None:
        append_history_message(session_id, {
            'type': 'error',
            'text': f'Local log file not found or invalid: {filepath}'
        })
        return {'error': 'File not found'}, 404

    # Set as active CSV in session (use the relative path)
    session['active_csv_path'] = filepath
    session['active_file'] = abs_path.name

    logger.info(f"Activated local log: {filepath}")

    append_history_message(session_id, {
        'type': 'assistant',
        'text': f'Successfully loaded {abs_path.name} from local storage. You can now analyze this data!'
    })

    return {'success': True, 'filename': abs_path.name}, 200
