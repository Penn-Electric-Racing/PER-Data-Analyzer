from __future__ import annotations

import getpass
from datetime import date
from typing import Sequence

import numpy as np
import polars as pl
import requests

from ..core_data_structures.data_instance import DataInstance
from ..server import SERVER_URL, TOKEN_ENDPOINT

QUERY_ENDPOINT = "/api/v1/timeline/query"
DEFAULT_MAX_ROWS = 1_000_000

__all__ = ["TimelineClient"]


def _request_token(password: str) -> str:
    try:
        response = requests.post(
            f"{SERVER_URL}{TOKEN_ENDPOINT}", json={"password": password}
        )
    except requests.RequestException as error:
        raise ConnectionError(f"Could not reach {SERVER_URL}: {error}") from error
    if response.status_code == 401:
        raise ConnectionError("Login failed: incorrect password")
    if not response.ok:
        raise ConnectionError(
            f"Login failed (HTTP {response.status_code}): {response.text}"
        )
    return response.json()["token"]


class TimelineClient:
    # Sends SQL to the timeline server (read only)

    def __init__(self, password: str | None = None) -> None:
        if password is None:
            password = getpass.getpass("team-internal programmatic password: ")
        self._token = _request_token(password)
        self._var_id_cache: dict[str, int | None] = {}

    def sizes(self) -> pl.DataFrame:
        return self.sql(
            "SELECT 'samples (hypertable)' AS object,"
            "       pg_size_pretty(hypertable_size('timeline_samples')) AS size"
            " UNION ALL SELECT t, pg_size_pretty(pg_total_relation_size(t))"
            " FROM unnest(ARRAY['timeline_session_var_stats','timeline_sessions','timeline_variables']) t"
        )

    def sql(
        self,
        query: str,
        params: Sequence[object] | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> pl.DataFrame:
        response = requests.post(
            f"{SERVER_URL}{QUERY_ENDPOINT}",
            headers={"Authorization": f"Bearer {self._token}"},
            json={
                "sql": query,
                "params": list(params) if params else None,
                "max_rows": max_rows,
            },
        )
        if not response.ok:
            detail = response.json().get("error", response.text)
            raise RuntimeError(f"Query failed (HTTP {response.status_code}): {detail}")
        payload = response.json()
        columns = payload["columns"]
        rows = payload["rows"]
        if not rows:
            return pl.DataFrame({column: [] for column in columns})
        return pl.DataFrame(
            {c: [r[i] for r in rows] for i, c in enumerate(columns)}, strict=False
        )

    def overview(self) -> pl.DataFrame:
        return self.sql(
            "SELECT test_day, count(*) AS sessions,"
            "       round(sum(duration_s)/60.0) AS minutes,"
            "       sum(n_rows) AS rows, max(n_variables) AS max_vars"
            " FROM timeline_sessions GROUP BY test_day ORDER BY test_day"
        )

    def stats(
        self,
        var_key: str,
        month: str | None = None,
        test_day: date | None = None,
    ) -> pl.DataFrame:
        """Per-session summary rows for one variable.

        Parameters
        ----------
        var_key : str
            Exact C++ identifier for the variable.
        month : str | None
            Restrict to a month, formatted ``"YYYY-MM"``.
        test_day : date | None
            Restrict to a single test day.

        Returns
        -------
        pl.DataFrame
            One row per session, ordered by start time.
        """
        clauses: list[str] = ["var_key = %s"]
        params: list[object] = [var_key]
        if month is not None:
            clauses.append("to_char(test_day, 'YYYY-MM') = %s")
            params.append(month)
        if test_day is not None:
            clauses.append("test_day = %s")
            params.append(test_day)
        return self.sql(
            "SELECT test_day, start_utc, session_id, source_key, n,"
            "       v_min, v_max, v_mean, v_p01, v_p50, v_p95, v_p99,"
            "       n_changes, n_rising, hz, max_gap_us"
            " FROM v_stats WHERE " + " AND ".join(clauses) + " ORDER BY start_utc",
            params,
        )

    def find(
        self,
        var_key: str,
        above: float | None = None,
        below: float | None = None,
        month: str | None = None,
        min_samples: int = 1,
    ) -> pl.DataFrame:
        """Find sessions where a variable crossed a threshold.
        Answered entirely from the summary tier -- no raw samples are read.

        Parameters
        ----------
        var_key : str
            Exact C++ identifier for the variable.
        above : float | None
            Keep sessions whose maximum exceeded this.
        below : float | None
            Keep sessions whose minimum fell under this.
        month : str | None
            Restrict to a month, formatted ``"YYYY-MM"``.
        min_samples : int
            Ignore sessions with fewer samples than this.

        Returns
        -------
        pl.DataFrame
            Matching sessions, worst-first.
        """
        clauses: list[str] = ["var_key = %s", "n >= %s"]
        params: list[object] = [var_key, min_samples]
        if above is not None:
            clauses.append("v_max > %s")
            params.append(above)
        if below is not None:
            clauses.append("v_min < %s")
            params.append(below)
        if month is not None:
            clauses.append("to_char(test_day, 'YYYY-MM') = %s")
            params.append(month)
        order = "v_max DESC" if above is not None else "v_min ASC"
        return self.sql(
            "SELECT test_day, start_utc, session_id, source_key,"
            "       v_min, v_max, v_mean, n"
            " FROM v_stats WHERE " + " AND ".join(clauses) + f" ORDER BY {order}",
            params,
        )


    def _time_bounds(
        self, session_id: int | None = None, test_day: date | None = None
    ) -> tuple[int, int] | None:
        """Look up the absolute microsecond span covered by a session or day.

        Filtering only on ``session_id`` scans every chunk, because the
        hypertable is partitioned on ``t_us``. Turning that filter into an
        explicit integer time range is what lets the planner prune chunks --
        and the bound must be a plain bigint, since a computed double
        precision expression prunes nothing.

        Parameters
        ----------
        session_id : int | None
            Session to bound.
        test_day : date | None
            Test day to bound.

        Returns
        -------
        tuple[int, int] | None
            ``(low_us, high_us)``, or None when neither filter was given.
        """
        if session_id is None and test_day is None:
            return None
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = %s")
            params.append(session_id)
        if test_day is not None:
            clauses.append("test_day = %s")
            params.append(test_day)
        bounds = self.sql(
            "SELECT (extract(epoch FROM min(start_utc)) * 1000000)::bigint AS lo,"
            "       (extract(epoch FROM max(coalesce(end_utc, start_utc)))"
            "        * 1000000)::bigint AS hi"
            " FROM timeline_sessions WHERE " + " AND ".join(clauses),
            params,
        )
        if not bounds.height or bounds["lo"][0] is None:
            return None
        # One second of slack absorbs rounding at the session edges.
        return int(bounds["lo"][0]) - 1_000_000, int(bounds["hi"][0]) + 1_000_000

    def samples(
        self,
        var_key: str,
        session_id: int | None = None,
        test_day: date | None = None,
    ) -> pl.DataFrame:
        """Fetch raw samples for one variable.

        Parameters
        ----------
        var_key : str
            Exact dotted C++ path.
        session_id : int | None
            Restrict to a single session.
        test_day : date | None
            Restrict to one test day.

        Returns
        -------
        pl.DataFrame
            Columns ``session_id``, ``ts_utc``, ``t_rel_us`` and ``value``.
        """
        var_id = self.var_id(var_key)
        if var_id is None:
            return pl.DataFrame(
                {"session_id": [], "ts_utc": [], "t_rel_us": [], "value": []}
            )

        # Query `samples` directly rather than through v_samples: joining
        # `variables` to resolve var_key turns into a hash join applied *after*
        # the scan, so the whole hypertable gets read. Filtering on a literal
        # var_id lets the columnstore prune by segment instead.
        clauses: list[str] = ["var_id = %s"]
        params: list[object] = [var_id]
        bounds = self._time_bounds(session_id=session_id, test_day=test_day)
        if bounds is not None:
            clauses.append("t_us BETWEEN %s AND %s")
            params.extend(bounds)
        if session_id is not None:
            clauses.append("session_id = %s")
            params.append(session_id)

        frame = self.sql(
            "SELECT session_id, t_us, value FROM timeline_samples"
            " WHERE " + " AND ".join(clauses) + " ORDER BY t_us",
            params,
        )
        if not frame.height:
            return pl.DataFrame(
                {"session_id": [], "ts_utc": [], "t_rel_us": [], "value": []}
            )

        starts = self.sql(
            "SELECT session_id,"
            " (extract(epoch FROM start_utc) * 1000000)::bigint AS start_us"
            " FROM timeline_sessions WHERE session_id = ANY(%s)",
            (frame["session_id"].unique().to_list(),),
        )
        return (
            frame.join(starts, on="session_id", how="left")
            .with_columns(
                pl.from_epoch(pl.col("t_us"), time_unit="us").alias("ts_utc"),
                (pl.col("t_us") - pl.col("start_us")).alias("t_rel_us"),
            )
            .select(["session_id", "ts_utc", "t_rel_us", "value"])
        )

    def var_id(self, var_key: str) -> int | None:
        """Resolve a variable key to its canonical id, with a local cache.

        Parameters
        ----------
        var_key : str
            Exact dotted C++ path.

        Returns
        -------
        int | None
            The canonical ``var_id``, or None if the key is not catalogued.
        """
        if var_key not in self._var_id_cache:
            found = self.sql(
                "SELECT var_id FROM timeline_variables WHERE var_key = %s", (var_key,)
            )
            self._var_id_cache[var_key] = (
                int(found["var_id"][0]) if found.height else None
            )
        return self._var_id_cache[var_key]

    def load(
        self,
        var_key: str,
        session_id: int | None = None,
        test_day: date | None = None,
    ) -> DataInstance:
        """Load a variable as a PERDA ``DataInstance`` for plotting and maths.

        Bridges the timeline back into the existing analysis stack, so a
        signal pulled from months of history behaves exactly like one parsed
        from a single log.

        Parameters
        ----------
        var_key : str
            Exact dotted C++ path.
        session_id : int | None
            Restrict to a single session.
        test_day : date | None
            Restrict to one test day.

        Returns
        -------
        DataInstance
            Timestamps in microseconds relative to the session start.
        """
        frame = self.samples(
            var_key, session_id=session_id, test_day=test_day
        )
        meta = self.sql(
            "SELECT var_id, description FROM timeline_variables WHERE var_key = %s", (var_key,)
        )
        var_id = int(meta["var_id"][0]) if meta.height else -1
        label = meta["description"][0] if meta.height else ""
        return DataInstance(
            timestamp_np=frame["t_rel_us"].to_numpy().astype(np.float64),
            value_np=frame["value"].to_numpy().astype(np.float64),
            label=label or var_key,
            var_id=var_id,
            cpp_name=var_key,
        )
