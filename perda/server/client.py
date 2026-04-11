import tempfile
from pathlib import Path
from typing import List

import boto3
import requests
from tqdm import tqdm

from ..analyzer.analyzer import Analyzer
from ..constants import DELIMITER
from .config import LogEntry, S3Credentials


class ServerClient:
    """Client for accessing log data from the PER data server.

    Authenticates with the gateway, browses available logs with metadata,
    and downloads CSV files directly from S3 for analysis.

    Parameters
    ----------
    url : str
        Base URL of the data server (e.g. ``"https://data.example.com"``)
    cache_dir : str | None
        Local directory for caching downloaded logs. When set, repeated
        ``load()`` calls for the same log skip the download. ``None``
        disables caching and uses a temporary file each time.

    Examples
    --------
    >>> client = ServerClient("https://data.example.com")
    >>> client.login("password")
    >>> logs = client.list_logs("REV 11/")
    >>> aly = client.load("REV 11/2026-04-01/test.csv")
    >>> aly.plot("ams.pack.voltage")
    """

    def __init__(self, url: str, cache_dir: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._session = requests.Session()
        self._s3_credentials: S3Credentials | None = None

        self._cache_dir: Path | None = None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir).expanduser()
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def login(self, password: str) -> None:
        """Authenticate with the data server.

        Parameters
        ----------
        password : str
            Server password

        Raises
        ------
        ConnectionError
            If login fails or the server is unreachable.
        """
        resp = self._session.post(
            f"{self._url}/api/v1/auth/login",
            json={"password": password},
        )
        if resp.status_code != 200:
            raise ConnectionError(
                f"Login failed (HTTP {resp.status_code}): {resp.text}"
            )

    def list_logs(self, prefix: str = "") -> List[LogEntry]:
        """List available logs under an S3 prefix, with metadata.

        Parameters
        ----------
        prefix : str
            S3 key prefix to browse (e.g. ``"REV 11/"``).
            Use ``""`` for the bucket root.

        Returns
        -------
        List[LogEntry]
            Combined S3 listing and metadata for each entry.

        Examples
        --------
        >>> logs = client.list_logs("REV 11/")
        >>> for log in logs:
        ...     print(log)
        """
        objects = self._list_s3_objects(prefix)
        metadata_list = self._get_page_metadata(prefix)

        # Index metadata by log_key for fast lookup
        meta_by_key = {m["log_key"]: m for m in metadata_list}

        entries: List[LogEntry] = []
        for obj in objects:
            meta = meta_by_key.get(obj["key"], {})
            entries.append(
                LogEntry(
                    key=obj["key"],
                    name=obj["name"],
                    is_folder=obj["isFolder"],
                    size=obj.get("size"),
                    last_modified=obj.get("lastModified"),
                    title=meta.get("title", ""),
                    description=meta.get("description", ""),
                    tags=meta.get("tags", []),
                    string_metadata=meta.get("string_metadata", {}),
                    numeric_metadata=meta.get("numeric_metadata", {}),
                )
            )
        return entries

    def _download(self, log_key: str) -> str:
        """Download a log CSV from S3.

        Parameters
        ----------
        log_key : str
            Full S3 key of the log file (e.g. ``"REV 11/2026-04-01/test.csv"``)

        Returns
        -------
        str
            Local file path to the downloaded CSV.
        """
        # Check cache first
        if self._cache_dir is not None:
            cached_path = self._cache_dir / log_key
            if cached_path.exists():
                return str(cached_path)

        # Download from S3 with progress bar
        creds = self._get_s3_credentials()
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            endpoint_url=creds.endpoint,
        )

        if self._cache_dir is not None:
            dest = self._cache_dir / log_key
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", prefix="perda_", delete=False
            )
            tmp.close()
            dest = Path(tmp.name)

        head = s3.head_object(Bucket=creds.bucket_name, Key=log_key)
        total_bytes = head["ContentLength"]
        filename = log_key.rsplit("/", 1)[-1]

        with tqdm(
            total=total_bytes, unit="B", unit_scale=True, desc=f"Downloading {filename}"
        ) as pbar:
            s3.download_file(
                creds.bucket_name, log_key, str(dest), Callback=pbar.update
            )
        return str(dest)

    def load(
        self,
        log_key: str,
        ts_offset: int = 0,
        parsing_errors_limit: int = 100,
    ) -> Analyzer:
        """Download a log from S3 and return an Analyzer instance.

        Parameters
        ----------
        log_key : str
            Full S3 key of the log file
        ts_offset : int
            Timestamp offset passed to ``Analyzer``. Default is 0.
        parsing_errors_limit : int
            Maximum parsing errors passed to ``Analyzer``. Default is 100.

        Returns
        -------
        Analyzer
            Ready-to-use Analyzer loaded with the downloaded data.

        Examples
        --------
        >>> aly = client.load("REV 11/2026-04-01/test.csv")
        >>> print(aly)
        """
        local_path = self._download(log_key)
        return Analyzer(
            local_path,
            ts_offset=ts_offset,
            parsing_errors_limit=parsing_errors_limit,
        )

    def print_logs(self, prefix: str = "") -> None:
        """Print a formatted listing of available logs.

        Parameters
        ----------
        prefix : str
            S3 key prefix to browse. Default is ``""`` (bucket root).

        Examples
        --------
        >>> client.print_logs("REV 11/")
        """
        entries = self.list_logs(prefix)
        print(f'Logs under "{prefix}"')
        print(DELIMITER)
        for entry in entries:
            print(entry)
        print(DELIMITER)
        file_count = sum(1 for e in entries if not e.is_folder)
        folder_count = sum(1 for e in entries if e.is_folder)
        print(f"{folder_count} folders, {file_count} files")

    # ── Private helpers ──────────────────────────────────────────────

    def _api_get(self, path: str) -> requests.Response:
        """Send an authenticated GET request to the gateway.

        Parameters
        ----------
        path : str
            API path (e.g. ``"/api/v1/metadata/some_key"``)

        Returns
        -------
        requests.Response
            Server response.

        Raises
        ------
        ConnectionError
            If the request fails with a non-2xx status.
        """
        resp = self._session.get(f"{self._url}{path}")
        if not resp.ok:
            raise ConnectionError(
                f"GET {path} failed (HTTP {resp.status_code}): {resp.text}"
            )
        return resp

    def _api_post(self, path: str, json: dict | None = None) -> requests.Response:
        """Send an authenticated POST request to the gateway.

        Parameters
        ----------
        path : str
            API path (e.g. ``"/api/v1/s3/list"``)
        json : dict | None
            JSON payload.

        Returns
        -------
        requests.Response
            Server response.

        Raises
        ------
        ConnectionError
            If the request fails with a non-2xx status.
        """
        resp = self._session.post(f"{self._url}{path}", json=json or {})
        if not resp.ok:
            raise ConnectionError(
                f"POST {path} failed (HTTP {resp.status_code}): {resp.text}"
            )
        return resp

    def _get_s3_credentials(self) -> S3Credentials:
        """Fetch and cache read-only S3 credentials from the gateway.

        Returns
        -------
        S3Credentials
            Credentials for direct S3 access.
        """
        if self._s3_credentials is None:
            resp = self._api_post("/api/v1/s3/credentials")
            self._s3_credentials = S3Credentials(**resp.json())
        return self._s3_credentials

    def _list_s3_objects(self, prefix: str) -> list:
        """Call the gateway S3 list endpoint.

        Parameters
        ----------
        prefix : str
            S3 key prefix.

        Returns
        -------
        list
            Raw list of S3 object dicts from the server.
        """
        resp = self._api_post("/api/v1/s3/list", json={"prefix": prefix})
        return resp.json()

    def _get_page_metadata(self, path: str) -> list:
        """Fetch metadata for direct children of a path.

        Parameters
        ----------
        path : str
            S3 key prefix.

        Returns
        -------
        list
            List of metadata dicts from the server.
        """
        resp = self._api_post("/api/v1/metadata/page", json={"path": path})
        return resp.json()
