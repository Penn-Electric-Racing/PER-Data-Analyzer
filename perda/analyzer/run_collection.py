import os
import glob
import re
from datetime import datetime
from typing import List, Callable, Union, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go

from .analyzer import Analyzer
from .comparison import plot_comparison, compare_summary


class RunCollection:
    """A collection of telemetry run logs (represented by Analyzer instances).

    Scans directories quickly by parsing only the first line header for date metadata,
    and lazy-loads full datasets only when accessed.
    """

    def __init__(self, items: List[Dict[str, Any]]):
        # items is a list of dicts: {"file_path": str, "filename": str, "date": Optional[datetime], "analyzer": Optional[Analyzer]}
        self._items = items

    @classmethod
    def from_paths(
        cls, paths: Union[str, List[str]], recursive: bool = False, pattern: str = "*.csv"
    ) -> "RunCollection":
        """Build a RunCollection from a single path or a list of paths (directories or files).

        If a path is a directory, it scans files matching the pattern inside it (recursively if recursive=True).
        If it is a file, it adds it directly.
        """
        # Normalize input to list of paths
        if isinstance(paths, str):
            path_list = [paths]
        else:
            path_list = list(paths)

        resolved_files = []
        for p in path_list:
            if os.path.isdir(p):
                if recursive:
                    # Recursive scanning using glob double asterisk **
                    search_pattern = os.path.join(p, "**", pattern)
                    files = glob.glob(search_pattern, recursive=True)
                else:
                    search_pattern = os.path.join(p, pattern)
                    files = glob.glob(search_pattern)
                resolved_files.extend(files)
            elif os.path.isfile(p):
                resolved_files.append(p)
            else:
                # Support direct glob pattern strings
                files = glob.glob(p, recursive=recursive)
                if files:
                    resolved_files.extend(files)

        # De-duplicate resolved files
        resolved_files = sorted(list(set(resolved_files)))

        items = []
        for filepath in resolved_files:
            filename = os.path.basename(filepath)
            date = None
            try:
                with open(filepath, "r") as f:
                    header_line = f.readline()
                    match = re.search(r"PER Log:\s*(.*?)(?:\s+v\d+\.\d+)?$", header_line)
                    if match:
                        date_str = match.group(1).strip()
                        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")
            except Exception:
                try:
                    date = datetime.fromtimestamp(os.path.getmtime(filepath))
                except Exception:
                    pass

            items.append({
                "file_path": filepath,
                "filename": filename,
                "date": date,
                "analyzer": None
            })

        # Sort runs chronologically by date
        items.sort(key=lambda x: (x["date"] is None, x["date"], x["filename"]))
        return cls(items)

    @classmethod
    def from_directory(
        cls, dir_path: str, pattern: str = "*.csv", recursive: bool = False
    ) -> "RunCollection":
        """Scan a directory of CSV logs and construct a collection index using header dates.

        Supports recursive scanning of subdirectories if recursive=True.
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"'{dir_path}' is not a valid directory.")
        return cls.from_paths(dir_path, recursive=recursive, pattern=pattern)

    @classmethod
    def from_files(cls, filepaths: List[str]) -> "RunCollection":
        """Build a collection from a specific list of file paths."""
        return cls.from_paths(filepaths)

    @property
    def runs(self) -> List[Analyzer]:
        """Load and return all Analyzer instances in this collection."""
        for item in self._items:
            if item["analyzer"] is None:
                item["analyzer"] = Analyzer(item["file_path"], verbose=0)
        return [item["analyzer"] for item in self._items]

    def get_run(self, index: int) -> Analyzer:
        """Load and return a specific Analyzer by index."""
        if index < 0 or index >= len(self._items):
            raise IndexError("Collection index out of range.")
        item = self._items[index]
        if item["analyzer"] is None:
            item["analyzer"] = Analyzer(item["file_path"], verbose=0)
        return item["analyzer"]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Analyzer:
        return self.get_run(index)

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "RunCollection":
        """Filter the collection by metadata items and return a new RunCollection."""
        filtered_items = [item for item in self._items if predicate(item)]
        return RunCollection(filtered_items)

    def filter_by_date(
        self, start_date: Union[str, datetime] = None, end_date: Union[str, datetime] = None
    ) -> "RunCollection":
        """Filter runs within a specific date range (inclusive)."""
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        def predicate(item: Dict[str, Any]) -> bool:
            d = item["date"]
            if d is None:
                return False
            if start_date and d < start_date:
                return False
            if end_date and d > end_date:
                return False
            return True

        return self.filter(predicate)

    def check_any(self, cpp_name: str, condition_fn: Callable[[np.ndarray], bool]) -> List[str]:
        """Find log filenames in the collection where a condition holds for a given variable.

        Evaluates runs lazily to keep memory usage bounded.

        Parameters
        ----------
        cpp_name : str
            C++ name of the variable to check.
        condition_fn : Callable[[np.ndarray], bool]
            Boolean condition function operating on the variable's numpy array values.

        Returns
        -------
        List[str]
            List of filenames where the condition evaluated to True.
        """
        matching_filenames = []
        for i in range(len(self._items)):
            item = self._items[i]
            # Use cached analyzer if available, otherwise load temporarily without caching
            # to keep memory footprint low and avoid OOM crashes
            if item["analyzer"] is not None:
                aly = item["analyzer"]
            else:
                try:
                    aly = Analyzer(item["file_path"], verbose=0)
                except Exception:
                    continue  # Skip unreadable file

            if cpp_name in aly.data:
                di = aly.data[cpp_name]
                if condition_fn(di.value_np):
                    matching_filenames.append(item["filename"])
        return matching_filenames

    def plot_comparison(self, cpp_name: str, max_points: int | None = None) -> go.Figure:
        """Overlay traces of the variable across all runs in the collection on a single plot.

        Parameters
        ----------
        cpp_name : str
            C++ name of the variable to compare.
        max_points : int | None, optional
            Maximum data points to plot per run (enables zero-copy step downsampling). Default is None (full fidelity).

        Returns
        -------
        go.Figure
            Plotly Figure with overlaid traces from all runs.
        """
        return plot_comparison(self.runs, cpp_name, max_points=max_points)

    def compare_summary(self, cpp_name: str) -> Dict[str, Dict[str, float]]:
        """Compare variable statistics across all runs in the collection."""
        return compare_summary(self.runs, cpp_name)
