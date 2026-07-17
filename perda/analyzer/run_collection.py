import os
import glob
import re
from datetime import datetime
from typing import List, Callable, Union, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from .analyzer import Analyzer
from .comparison import plot_comparison, compare_summary


class RunMetadata(BaseModel):
    """Internal model for telemetry run index metadata."""
    file_path: str = Field(description="Absolute path to the log file")
    filename: str = Field(description="Display filename of the log")
    date: Optional[datetime] = Field(default=None, description="Recording date parsed from header or file stats")
    analyzer: Optional[Analyzer] = Field(default=None, description="Lazily-loaded Analyzer instance")

    model_config = {
        "arbitrary_types_allowed": True
    }


class RunCollection:
    """A collection of telemetry run logs (represented by Analyzer instances).

    Scans directories quickly by parsing only the first line header for date metadata,
    and lazy-loads full datasets only when accessed.
    """

    def __init__(
        self,
        paths: Optional[Union[str, List[str]]] = None,
        recursive: bool = False,
        pattern: str = "*.csv",
        items: Optional[List[RunMetadata]] = None,
    ):
        """Construct a RunCollection directly from paths (or a pre-built list of RunMetadata).

        Parameters
        ----------
        paths : Union[str, List[str]], optional
            Directory paths, list of files, or wildcards to scan.
        recursive : bool, optional
            Recursively search directories if True. Default is False.
        pattern : str, optional
            Filename glob pattern to match. Default is "*.csv".
        items : List[RunMetadata], optional
            Internal list of pre-built items. Used internally for filtering.
        """
        if items is not None:
            self._items = items
            return

        if paths is None:
            self._items = []
            return

        # Normalize input to list of paths
        if isinstance(paths, str):
            path_list = [paths]
        else:
            path_list = list(paths)

        resolved_files = []
        for p in path_list:
            if os.path.isdir(p):
                if recursive:
                    search_pattern = os.path.join(p, "**", pattern)
                    files = glob.glob(search_pattern, recursive=True)
                else:
                    search_pattern = os.path.join(p, pattern)
                    files = glob.glob(search_pattern)
                resolved_files.extend(files)
            elif os.path.isfile(p):
                resolved_files.append(p)
            else:
                files = glob.glob(p, recursive=recursive)
                if files:
                    resolved_files.extend(files)

        # De-duplicate and sort resolved files
        resolved_files = sorted(list(set(resolved_files)))

        self._items = []
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

            self._items.append(
                RunMetadata(
                    file_path=filepath,
                    filename=filename,
                    date=date,
                    analyzer=None
                )
            )

        # Sort runs chronologically by date
        self._items.sort(key=lambda x: (x.date is None, x.date, x.filename))

    @property
    def runs(self) -> List[Analyzer]:
        """Load and return all Analyzer instances in this collection."""
        for item in self._items:
            if item.analyzer is None:
                item.analyzer = Analyzer(item.file_path, verbose=0)
        return [item.analyzer for item in self._items]

    def get_run(self, index: int) -> Analyzer:
        """Load and return a specific Analyzer by index."""
        if index < 0 or index >= len(self._items):
            raise IndexError("Collection index out of range.")
        item = self._items[index]
        if item.analyzer is None:
            item.analyzer = Analyzer(item.file_path, verbose=0)
        return item.analyzer

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Analyzer:
        return self.get_run(index)

    def filter(self, predicate: Callable[[RunMetadata], bool]) -> "RunCollection":
        """Filter the collection by metadata items and return a new RunCollection."""
        filtered_items = [item for item in self._items if predicate(item)]
        return RunCollection(items=filtered_items)

    def filter_by_date(
        self, start_date: Union[str, datetime] = None, end_date: Union[str, datetime] = None
    ) -> "RunCollection":
        """Filter runs within a specific date range (inclusive)."""
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        def predicate(item: RunMetadata) -> bool:
            d = item.date
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
            if item.analyzer is not None:
                aly = item.analyzer
            else:
                try:
                    aly = Analyzer(item.file_path, verbose=0)
                except Exception:
                    continue  # Skip unreadable file

            if cpp_name in aly.data:
                di = aly.data[cpp_name]
                if condition_fn(di.value_np):
                    matching_filenames.append(item.filename)
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
