from datetime import datetime
from pathlib import Path
from typing import Sequence

from .csv import parse_header_creation_time
from .run_collection import RunCollection, RunMetadata


def _read_run_metadata(path: Path) -> RunMetadata:
    """
    Read a log's index metadata by parsing only its first line.

    Falls back to the file's modification time when the header carries no date.

    Parameters
    ----------
    path : Path
        Path to the log file.

    Returns
    -------
    RunMetadata
        Metadata for the run, with ``date`` left as None if no date could be determined.
    """
    date = None
    try:
        with open(path, "r") as f:
            date = parse_header_creation_time(f.readline())
    except OSError:
        pass

    if date is None:
        try:
            date = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            pass

    return RunMetadata(file_path=path, date=date)


def from_paths(paths: Sequence[Path | str]) -> RunCollection:
    """
    Build a RunCollection from an explicit list of log files.

    Parameters
    ----------
    paths : Sequence[Path | str]
        Paths to the log files. Duplicates are removed.

    Returns
    -------
    RunCollection
        Collection of the given runs, ordered chronologically.

    Examples
    --------
    >>> col = from_paths(["logs/practice.csv", "logs/endurance.csv"])
    """
    unique_paths = sorted({Path(path) for path in paths})
    return RunCollection([_read_run_metadata(path) for path in unique_paths])


def from_directory(
    directory: Path | str, pattern: str = "*.csv", recursive: bool = False
) -> RunCollection:
    """
    Build a RunCollection from every matching log file in a directory.

    Parameters
    ----------
    directory : Path | str
        Directory to scan.
    pattern : str, optional
        Filename glob pattern to match. Default is "*.csv".
    recursive : bool, optional
        Search subdirectories as well. Default is False.

    Returns
    -------
    RunCollection
        Collection of the matching runs, ordered chronologically.

    Examples
    --------
    >>> col = from_directory("csv_files", recursive=True)
    """
    directory = Path(directory)
    matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return from_paths([path for path in matches if path.is_file()])
