from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..core_data_structures.data_instance import DataInstance
from ..plotting.data_instance_plotter import plot_single_axis
from ..units import Timescale, convert_time
from ..utils.data_summary import DataInstanceSummary, data_instance_summary
from .analyzer import Analyzer

# Common unit that every run's timestamps are converted to before overlaying, so
# that collections mixing v1 (ms) and v2 (us) logs share one x-axis scale.
COMPARISON_TIMESTAMP_UNIT = Timescale.US


class RunMetadata(BaseModel):
    """Cheaply-obtained index metadata for a single telemetry run."""

    file_path: Path = Field(description="Path to the log file")
    date: datetime | None = Field(
        default=None, description="Recording date parsed from the header or file stats"
    )

    def __str__(self) -> str:
        """
        One-line description of the run.

        Returns
        -------
        str
            ``<filename> (<date>)``
        """
        return f"{self.file_path.name} ({self.date if self.date else 'unknown date'})"


class RunCollection:
    """A chronologically ordered collection of telemetry runs.

    Holds only lightweight metadata. Logs are parsed into ``Analyzer`` instances lazily,
    one at a time, so that a collection spanning an entire season stays cheap to hold.

    Notes
    -----
    Construct via the factory functions in ``run_collection_factory`` (``from_directory``,
    ``from_paths``) rather than calling ``__init__`` directly.
    """

    def __init__(self, run_metadata: list[RunMetadata]) -> None:
        """
        Build a collection from run metadata, sorted chronologically.

        Parameters
        ----------
        run_metadata : list[RunMetadata]
            Metadata for each run. Runs without a date sort last, by filename.
        """
        self._runs_by_date = sorted(
            run_metadata,
            key=lambda run: (
                run.date is None,
                run.date or datetime.min,
                run.file_path.name,
            ),
        )
        self._loaded: dict[Path, Analyzer] = {}

    def __str__(self) -> str:
        """
        Listing of every run in chronological order, without loading any of them.

        Returns
        -------
        str
            One line per run, preceded by a count.
        """
        listing = "\n".join(
            f"  [{i}] {run}" for i, run in enumerate(self._runs_by_date)
        )
        return f"RunCollection with {len(self)} run(s):\n{listing}"

    def __len__(self) -> int:
        """Number of runs in the collection."""
        return len(self._runs_by_date)

    def __getitem__(self, index: int) -> Analyzer:
        """Load the run at the given chronological index."""
        return self.get_run_by_chronological_index(index)

    @property
    def filenames(self) -> list[str]:
        """Filenames of every run, in chronological order. Loads nothing."""
        return [run.file_path.name for run in self._runs_by_date]

    def get_run_by_chronological_index(self, index: int) -> Analyzer:
        """
        Load the run at a position in the collection's chronological ordering.

        Parameters
        ----------
        index : int
            Position in chronological order, not the order paths were supplied.

        Returns
        -------
        Analyzer
            Analyzer for that run. Repeated calls reuse the same instance.
        """
        return self._load(self._runs_by_date[index])

    def _load(self, run: RunMetadata) -> Analyzer:
        """
        Parse a run into an Analyzer, reusing an already-parsed one if available.

        Parameters
        ----------
        run : RunMetadata
            Metadata identifying the run to load.

        Returns
        -------
        Analyzer
            Analyzer for that run.
        """
        if run.file_path not in self._loaded:
            self._loaded[run.file_path] = Analyzer(str(run.file_path), verbose=0)
        return self._loaded[run.file_path]

    def _iter_analyzers(self) -> Iterator[tuple[RunMetadata, Analyzer]]:
        """
        Iterate over every run, parsing one log at a time.

        Unreadable logs are reported and skipped rather than aborting the iteration.

        Yields
        ------
        tuple[RunMetadata, Analyzer]
            Metadata and loaded Analyzer for each readable run.
        """
        for run in self._runs_by_date:
            try:
                analyzer = self._load(run)
            except Exception as error:
                print(f"Warning: skipping unreadable log {run.file_path}: {error}")
                continue
            yield run, analyzer

    def filter(self, predicate: Callable[[RunMetadata], bool]) -> RunCollection:
        """
        Select runs by their metadata, without loading any log.

        Parameters
        ----------
        predicate : Callable[[RunMetadata], bool]
            Returns True for runs to keep.

        Returns
        -------
        RunCollection
            New collection holding only the matching runs.

        Examples
        --------
        >>> col.filter(lambda run: "endurance" in run.file_path.name)
        """
        return RunCollection([run for run in self._runs_by_date if predicate(run)])

    def filter_by_date(
        self,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> RunCollection:
        """
        Select runs recorded within a date range, inclusive on both ends.

        Runs with no known date are excluded.

        Parameters
        ----------
        start_date : str | datetime | None, optional
            Earliest date to keep, as a datetime or ISO 8601 string. Default is None (no lower bound).
        end_date : str | datetime | None, optional
            Latest date to keep, as a datetime or ISO 8601 string. Default is None (no upper bound).

        Returns
        -------
        RunCollection
            New collection holding only the matching runs.

        Examples
        --------
        >>> col.filter_by_date("2026-06-01", "2026-06-30")
        """
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        return self.filter(
            lambda run: (
                run.date is not None
                and (start_date is None or run.date >= start_date)
                and (end_date is None or run.date <= end_date)
            )
        )

    def filter_by_data(self, predicate: Callable[[Analyzer], bool]) -> RunCollection:
        """
        Select runs by their contents, parsing one log at a time.

        Unlike ``filter``, this must read every log, so prefer narrowing the collection
        with ``filter`` or ``filter_by_date`` first.

        Parameters
        ----------
        predicate : Callable[[Analyzer], bool]
            Returns True for runs to keep. Receives the loaded Analyzer.

        Returns
        -------
        RunCollection
            New collection holding only the matching runs.

        Examples
        --------
        >>> overvolted = col.filter_by_data(
        ...     lambda aly: (aly.data["ams.pack.voltage"].value_np > 10.0).any()
        ... )
        >>> overvolted.filenames
        """
        return RunCollection(
            [run for run, analyzer in self._iter_analyzers() if predicate(analyzer)]
        )

    def plot_comparison(
        self, cpp_name: str, max_points: int | None = None
    ) -> go.Figure:
        """
        Overlay a variable from every run in the collection on a single plot.

        Each run's timestamps are shifted so that all runs start at t = 0.

        Parameters
        ----------
        cpp_name : str
            C++ name of the variable to compare.
        max_points : int | None, optional
            Maximum data points to plot per run. Default is None (full fidelity).

        Returns
        -------
        go.Figure
            Plotly Figure with one trace per run that contains the variable.

        Examples
        --------
        >>> col.plot_comparison("pcm.wheelSpeeds.frontLeft", max_points=20000).show()
        """
        overlaid = []
        for run, analyzer in self._iter_analyzers():
            if cpp_name not in analyzer.data:
                continue
            di = analyzer.data[cpp_name]
            if len(di) == 0:
                continue

            relative_timestamps = di.timestamp_np - analyzer.data.data_start_time
            overlaid.append(
                DataInstance(
                    timestamp_np=convert_time(
                        relative_timestamps,
                        analyzer.data.timestamp_unit,
                        COMPARISON_TIMESTAMP_UNIT,
                    ),
                    value_np=di.value_np,
                    label=f"{run.file_path.name} ({di.label})",
                    var_id=di.var_id,
                    cpp_name=di.cpp_name,
                )
            )

        return plot_single_axis(
            overlaid,
            title=f"Multi-Log Comparison: {cpp_name}",
            y_axis_title=cpp_name,
            timestamp_unit=COMPARISON_TIMESTAMP_UNIT,
            max_points=max_points,
        )

    def compare_summary(self, cpp_name: str) -> dict[str, DataInstanceSummary]:
        """
        Summarize a variable's statistics across every run in the collection.

        Parameters
        ----------
        cpp_name : str
            C++ name of the variable to summarize.

        Returns
        -------
        dict[str, DataInstanceSummary]
            Summary per run filename, for runs that contain the variable.

        Examples
        --------
        >>> for filename, summary in col.compare_summary("bms.pack.voltage").items():
        ...     print(f"{filename}: {summary.max_value}")
        """
        return {
            run.file_path.name: data_instance_summary(
                analyzer.data[cpp_name], source_time_unit=analyzer.data.timestamp_unit
            )
            for run, analyzer in self._iter_analyzers()
            if cpp_name in analyzer.data
        }
