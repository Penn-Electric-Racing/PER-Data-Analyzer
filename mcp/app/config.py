import os
from pathlib import Path

class Config:
    SECRET_KEY = os.urandom(24)
    UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Ensure upload directory exists
    UPLOAD_FOLDER.mkdir(exist_ok=True)
