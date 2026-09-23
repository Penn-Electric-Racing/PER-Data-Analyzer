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


def default_dsn(password: str | None = None) -> str:
    """Build the connection string, asking for the password if need be.

    ``$TIMELINE_DSN`` overrides everything, which is how the data server
    itself points at its own local database.

    Parameters
    ----------
    password : str | None
        Password for ``timeline_ro``. Prompted for when omitted and
        ``$TIMELINE_PASSWORD`` is unset.

    Returns
    -------
    str
        libpq connection string.
    """
    override = os.getenv("TIMELINE_DSN")
    if override:
        return override
    if password is None:
        password = os.getenv("TIMELINE_PASSWORD") or getpass.getpass(
            f"password for {TIMELINE_USER}@{TIMELINE_HOST}: "
        )
    return (
        f"postgresql://{TIMELINE_USER}:{quote(password)}"
        f"@{TIMELINE_HOST}:{TIMELINE_PORT}/{TIMELINE_DB}"
    )


def _request_token(password: str) -> str:
    """Exchange the team-internal programmatic password for a bearer token.

    Parameters
    ----------
    password : str
        Team-internal programmatic password.

    Returns
    -------
    str
        Bearer token for the timeline endpoint.
    """
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
    """Query the global timeline through the data server.

    Sends SQL to the server, which runs it read-only and sends the rows back,
    the same way ``access_remote_log`` goes through the server for an S3 key.
    Nothing here talks to the database, so nothing here can write to it.

    Parameters
    ----------
    password : str | None
        Team-internal programmatic password. Prompted for when omitted.

    Examples
    --------
    >>> tl = TimelineClient()
    team-internal programmatic password:
    >>> tl.find("bms.stack.mma.cellV.min", below=3.0, month="2026-05")
    """

    def __init__(self, password: str | None = None) -> None:
        if password is None:
            password = getpass.getpass("team-internal programmatic password: ")
        self._token = _request_token(password)
        self._var_id_cache: dict[str, int | None] = {}

    def sizes(self) -> pl.DataFrame:
        """On-disk size of each timeline table.

        Returns
        -------
        pl.DataFrame
            Columns ``object`` and ``size``.
        """
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
        """Run arbitrary SQL and return the result as a Polars frame.

        This is the escape hatch the LLM layer will eventually target: the
        views ``v_sessions``, ``v_stats`` and ``v_samples`` are stable names
        that hide ids, epoch microseconds and partition layout.

        Parameters
        ----------
        query : str
            SQL text.
        params : Sequence[object] | None
            Bind parameters.
        max_rows : int
            Row cap. The server clamps this to its own ceiling.

        Returns
        -------
        pl.DataFrame
            Result set; empty frame when the query returns no rows.
        """
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
        """Summarise what the timeline currently holds, by test day.

        Returns
        -------
        pl.DataFrame
            One row per test day with session counts, duration and row counts.
        """
        return self.sql(
            "SELECT test_day, count(*) AS sessions,"
            "       round(sum(duration_s)/60.0) AS minutes,"
            "       sum(n_rows) AS rows, max(n_variables) AS max_vars"
            " FROM timeline_sessions GROUP BY test_day ORDER BY test_day"
        )

    def search(self, text: str, limit: int = 25) -> pl.DataFrame:
        """Find catalogued variables whose key or description matches text.

        Parameters
        ----------
        text : str
            Case-insensitive substring.
        limit : int
            Maximum rows to return.

        Returns
        -------
        pl.DataFrame
            Matching variables with dtype and session coverage.
        """
        return self.sql(
            "SELECT var_key, dtype, description, n_sessions, first_seen, last_seen"
            " FROM timeline_variables"
            " WHERE var_key ILIKE %s OR description ILIKE %s"
            " ORDER BY n_sessions DESC, var_key LIMIT %s",
            (f"%{text}%", f"%{text}%", limit),
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
            Exact dotted C++ path.
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
            "SELECT test_day, start_utc, session_id, source_key, n, n_invalid,"
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
            Exact dotted C++ path.
        above : float | None
            Keep sessions whose maximum exceeded this.
        below : float | None
            Keep sessions whose minimum fell under this.
        month : str | None
            Restrict to a month, formatted ``"YYYY-MM"``.
        min_samples : int
            Ignore sessions with fewer valid samples than this.

        Returns
        -------
        pl.DataFrame
            Matching sessions, worst-first.
        """
        clauses: list[str] = ["var_key = %s", "(n - n_invalid) >= %s"]
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
            "       v_min, v_max, v_mean, n, n_invalid"
            " FROM v_stats WHERE " + " AND ".join(clauses) + f" ORDER BY {order}",
            params,
        )

    def invalid_report(self, min_fraction: float = 0.01) -> pl.DataFrame:
        """Rank variables by how often they emit sentinel values.

        Surfaces signals that are silently broken -- a sentinel decodes to a
        near-zero float, so a naive aggregate reports it as a plausible
        reading rather than as missing data.

        Parameters
        ----------
        min_fraction : float
            Only report variables whose invalid share exceeds this.

        Returns
        -------
        pl.DataFrame
            Variables ordered by invalid fraction, worst first.
        """
        return self.sql(
            "SELECT var_key, dtype, max(sentinel_bits) AS sentinel_bits,"
            "       sum(n_invalid) AS n_invalid, sum(n) AS n,"
            "       sum(n_invalid)::float / NULLIF(sum(n), 0) AS invalid_fraction,"
            "       count(*) FILTER (WHERE n_invalid > 0) AS sessions_affected,"
            "       count(*) AS sessions_total"
            " FROM v_stats GROUP BY var_key, dtype"
            " HAVING sum(n_invalid)::float / NULLIF(sum(n), 0) > %s"
            " ORDER BY invalid_fraction DESC",
            (min_fraction,),
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
        drop_invalid: bool = True,
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
        drop_invalid : bool
            Exclude denormal sentinel samples.

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
        if drop_invalid:
            clauses.append("(value = 0 OR abs(value) >= 1.17549435e-38)")

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
        drop_invalid: bool = True,
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
        drop_invalid : bool
            Exclude denormal sentinel samples.

        Returns
        -------
        DataInstance
            Timestamps in microseconds relative to the session start.
        """
        frame = self.samples(
            var_key, session_id=session_id, test_day=test_day, drop_invalid=drop_invalid
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
