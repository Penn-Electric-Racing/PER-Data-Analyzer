from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class S3Credentials(BaseModel):
    """Read-only S3 credentials returned by the server."""

    access_key: str
    secret_key: str
    bucket_name: str
    endpoint: str


class LogEntry(BaseModel):
    """A single S3 object (file or folder) with optional metadata.

    Combines S3 listing info with server-side metadata for display
    when browsing available logs.
    """

    key: str
    name: str
    is_folder: bool
    size: int | None = None
    last_modified: str | None = None

    # Metadata fields (populated when metadata exists on server)
    title: str = ""
    description: str = ""
    tags: List[str] = []
    string_metadata: Dict[str, str] = {}
    numeric_metadata: Dict[str, float] = {}

    def __str__(self) -> str:
        if self.is_folder:
            return f"[DIR]  {self.name}/"

        parts = [f"       {self.name}"]
        if self.title:
            parts.append(f'  title: "{self.title}"')
        if self.tags:
            parts.append(f"  tags: [{', '.join(self.tags)}]")
        if self.size is not None:
            size_mb = self.size / (1024 * 1024)
            parts.append(f"  size: {size_mb:.1f} MB")
        if self.last_modified:
            parts.append(f"  modified: {self.last_modified}")
        if self.description:
            desc = self.description[:80] + ("..." if len(self.description) > 80 else "")
            parts.append(f'  desc: "{desc}"')
        if self.string_metadata:
            for k, v in self.string_metadata.items():
                parts.append(f"  {k}: {v}")
        if self.numeric_metadata:
            for k, v in self.numeric_metadata.items():
                parts.append(f"  {k}: {v}")

        return "\n".join(parts)
