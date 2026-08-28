import tempfile
from pathlib import Path

import boto3
import requests
from mypy_boto3_s3.client import S3Client

from .analyzer.analyzer import Analyzer

# SERVER_URL = "https://data-server.pennelectricracing.com"
SERVER_URL = "http://127.0.0.1:5000"
TOKEN_ENDPOINT = "/api/v1/auth/team-internal-programmatic/token"
CREDENTIALS_ENDPOINT = "/api/v1/s3/credentials"


def access_remote_log(password: str, log_path: str, **kwargs) -> Analyzer:
    """
    Construct an Analyzer from a log file stored on the data server.

    Parameters
    ----------
    password : str
        Team-internal programmatic password.
    log_path : str
        Path to the log file, relative to the bucket root, e.g.
        ``"REV 11/2026-04-01/test.csv"``.
    **kwargs
        Additional keyword arguments passed to the Analyzer constructor.

    Returns
    -------
    Analyzer
        Analyzer loaded with the downloaded log.

    Examples
    --------
    >>> aly = access_remote_log("password", "REV 11/2026-04-01/test.csv")
    """
    s3_client, bucket_name = _create_s3_client_via_data_server(password)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".csv", prefix="perda_", delete=False
    )
    temp_file.close()
    local_path = Path(temp_file.name)

    try:
        s3_client.download_file(bucket_name, log_path, str(local_path))
        return Analyzer(str(local_path), **kwargs)
    finally:
        local_path.unlink(missing_ok=True)


def _create_s3_client_via_data_server(password: str) -> tuple[S3Client, str]:
    try:
        token_response = requests.post(
            f"{SERVER_URL}{TOKEN_ENDPOINT}",
            json={"password": password},
        )
    except requests.RequestException as error:
        raise ConnectionError(f"Could not reach {SERVER_URL}: {error}") from error

    if token_response.status_code == 401:
        raise ConnectionError("Login failed: incorrect password")
    if not token_response.ok:
        raise ConnectionError(
            f"Login failed (HTTP {token_response.status_code}): {token_response.text}"
        )

    token = token_response.json()["token"]
    credentials_response = requests.post(
        f"{SERVER_URL}{CREDENTIALS_ENDPOINT}",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    if not credentials_response.ok:
        raise ConnectionError(
            f"Could not fetch S3 credentials "
            f"(HTTP {credentials_response.status_code}): {credentials_response.text}"
        )

    credentials = credentials_response.json()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret_key"],
        endpoint_url=credentials["endpoint"],
    )
    return s3_client, credentials["bucket_name"]
