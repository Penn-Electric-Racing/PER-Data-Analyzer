"""Local logs service for listing log files from local directory."""
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class LocalLogsService:
    """Service for reading log files from local logs directory."""

    def __init__(self):
        """Initialize the service with the logs directory path."""
        # Navigate up to PER directory, then into logs/
        # From mcp/app/services/ -> up to mcp -> up to PER-Data-Analyzer -> up to PER -> logs/
        self._logs_dir = Path(__file__).parent.parent.parent.parent.parent / 'logs'

    def list_local_logs(self) -> List[Dict]:
        """
        List all CSV log files from local logs directory, grouped by folder.

        Returns:
            List of folder groups, each containing:
            {
                'folder': 'relative/path/from/logs',
                'files': [
                    {
                        'name': 'filename.csv',
                        'path': '../../logs/relative/path/filename.csv',
                        'size': bytes,
                        'modified': ISO datetime string
                    },
                    ...
                ]
            }
        """
        if not self._logs_dir.exists():
            return []

        # Find all CSV files recursively
        csv_files = list(self._logs_dir.rglob('*.csv'))

        if not csv_files:
            return []

        # Group files by parent directory
        folders = {}
        for csv_file in csv_files:
            # Get relative path from logs directory
            try:
                rel_path = csv_file.relative_to(self._logs_dir)
                folder = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'

                # Get file stats
                stats = csv_file.stat()
                modified_time = datetime.fromtimestamp(stats.st_mtime)

                # Build relative path from mcp directory
                # mcp -> PER-Data-Analyzer -> PER/logs
                path_from_mcp = f"../../logs/{rel_path}"

                file_info = {
                    'name': csv_file.name,
                    'path': path_from_mcp,
                    'size': stats.st_size,
                    'modified': modified_time.isoformat()
                }

                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(file_info)

            except ValueError:
                # Skip files that aren't within logs directory
                continue

        # Sort files within each folder by modification time (newest first)
        for folder in folders:
            folders[folder].sort(key=lambda x: x['modified'], reverse=True)

        # Convert to list format and sort folders alphabetically
        result = [
            {
                'folder': folder,
                'files': files
            }
            for folder, files in sorted(folders.items())
        ]

        return result

    def get_absolute_path(self, relative_path: str) -> Optional[Path]:
        """
        Convert a relative path (from mcp directory) to absolute path and validate it exists.

        Args:
            relative_path: Path relative to mcp directory (e.g., '../../../logs/folder/file.csv')

        Returns:
            Absolute Path if file exists, None otherwise
        """
        # Start from mcp directory
        # From mcp/app/services/ -> up to mcp/app/ -> up to mcp/
        mcp_dir = Path(__file__).parent.parent.parent

        # Resolve the relative path
        abs_path = (mcp_dir / relative_path).resolve()

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Resolving path: {relative_path}")
        logger.info(f"MCP dir: {mcp_dir}")
        logger.info(f"Absolute path: {abs_path}")
        logger.info(f"Logs dir: {self._logs_dir.resolve()}")
        logger.info(f"File exists: {abs_path.exists()}")

        # Verify it exists and is within logs directory
        if abs_path.exists() and abs_path.is_file():
            try:
                # Ensure it's actually within the logs directory (security check)
                abs_path.relative_to(self._logs_dir.resolve())
                return abs_path
            except ValueError as e:
                # Path is outside logs directory
                logger.error(f"Path outside logs directory: {e}")
                return None

        return None


# Global service instance
local_logs_service = LocalLogsService()
