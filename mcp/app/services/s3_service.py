"""S3 service for listing and downloading log files."""
import os
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from posixpath import normpath


class S3Service:
    """Service for reading log files from S3 bucket (shared with car-data-server)."""

    def __init__(self):
        """Initialize S3 client with credentials from environment."""
        self._client = None
        self._bucket = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of S3 client."""
        if self._initialized:
            return

        # Get S3 credentials from environment
        s3_key = os.getenv('S3_KEY')
        s3_secret = os.getenv('S3_SECRET')
        s3_endpoint = os.getenv('S3_ENDPOINT')
        self._bucket = os.getenv('S3_BUCKET')

        if not all([s3_key, s3_secret, s3_endpoint, self._bucket]):
            raise ValueError("S3 credentials not configured. Check S3_KEY, S3_SECRET, S3_ENDPOINT, and S3_BUCKET in .env")

        # Initialize boto3 S3 client
        self._client = boto3.client(
            's3',
            aws_access_key_id=s3_key,
            aws_secret_access_key=s3_secret,
            endpoint_url=s3_endpoint
        )
        self._initialized = True

    def list_log_files(self) -> Tuple[List[Dict], int]:
        """
        List all CSV log files in the S3 bucket.

        Returns:
            tuple: (list of file info dicts, status_code)
                   Each dict contains: name, upload_time, size
        """
        try:
            self._ensure_initialized()
            
            response = self._client.list_objects_v2(Bucket=self._bucket)
            files = []
            
            for item in response.get('Contents', []):
                # Only include CSV files
                if item['Key'].lower().endswith('.csv'):
                    files.append({
                        'name': item['Key'],
                        'upload_time': item['LastModified'].isoformat() if 'LastModified' in item else None,
                        'size': item.get('Size')
                    })
            
            # Sort by upload time (most recent first)
            files.sort(key=lambda x: x['upload_time'] or '', reverse=True)
            
            return (files, 200)
        except ClientError as e:
            print(f"[ERROR] S3 list error: {e}", flush=True)
            return ([], 500)
        except ValueError as e:
            print(f"[ERROR] S3 configuration error: {e}", flush=True)
            return ([], 500)

    def download_log_file(self, filename: str, upload_dir: Path) -> Optional[Path]:
        """
        Download a log file from S3 to the local uploads directory.

        Args:
            filename: Name of the file in S3
            upload_dir: Local directory to save the file

        Returns:
            Path to the downloaded file, or None if download failed
        """
        try:
            self._ensure_initialized()

            # Ensure upload directory exists
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Normalize and validate S3 key so it cannot escape upload_dir.
            normalized_key = normpath(filename).lstrip("/")
            if normalized_key in ("", ".") or normalized_key.startswith("../") or "/../" in normalized_key:
                print(f"[ERROR] Invalid S3 key path: {filename}", flush=True)
                return None

            # Download to local file (preserve S3 key folder structure).
            local_path = upload_dir / normalized_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, filename, str(local_path))

            print(f"[INFO] Downloaded {filename} from S3 to {local_path}", flush=True)
            return local_path

        except ClientError as e:
            print(f"[ERROR] S3 download error for {filename}: {e}", flush=True)
            return None
        except ValueError as e:
            print(f"[ERROR] S3 configuration error: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected S3 download error for {filename}: {e}", flush=True)
            return None


# Global S3 service instance
s3_service = S3Service()
