#!/usr/bin/env python3
"""
Web Frontend for PER MCP Server

Flask-based web interface with chat UI and file upload for CAN data analysis.
"""

import logging
from app import create_app

# Setup logging
logging.basicConfig(level=logging.INFO)

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("PER MCP Server - Web Frontend")
    print("=" * 60)
    print()
    print("Starting web server on http://localhost:5001")
    print("Open your browser and navigate to http://localhost:5001")
    print()

    app.run(debug=True, host='0.0.0.0', port=5001)
