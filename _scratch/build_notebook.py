"""Generate the timeline demo notebook.

Kept as a script so the notebook can be regenerated after schema changes
rather than hand-edited.
"""

import nbformat as nbf

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

cells = [
    MD(
        "# PER Global Data Timeline\n"
        "\n"
        "Every sample from every REV 11 log, in one queryable store.\n"
        "\n"
        "The problem this solves: today a question like *\"which testing sessions in May \n"
        "had a cell voltage sag?\"* means downloading and parsing logs one at a time. \n"
        "There is no way to ask a question **across** the season.\n"
        "\n"
        "### How it's built\n"
        "\n"
        "| Layer | Where | Size | What it answers |\n"
        "|---|---|---|---|\n"
        "| `test_days` / `sessions` | Postgres | KB | *when did we test* |\n"
        "| `variables` (catalog) | Postgres | MB | *what does the car log* |\n"
        "| `session_var_stats` | Postgres | ~50 MB | **most questions** |\n"
        "| `samples` | TimescaleDB hypertable, columnstore | ~4 bytes/row | waveform detail |\n"
        "\n"
        "The trick is the third row. Per-session-per-variable summaries are small enough "
        "to scan instantly, so threshold questions never touch the billions of raw samples. "
        "Raw data is read only after the summary tier has narrowed things to a handful "
        "of sessions.\n"
        "\n"
        "Two facts that shape everything:\n"
        "\n"
        "- **The numeric variable ids in a log header are not stable** — not even between "
        "two REV 11 builds. The dotted C++ path (`bms.stack.mma.cellV.min`) is the only "
        "durable key, so ingest remaps every file's local ids onto a canonical catalog.\n"
        "- **Timestamps are stored as absolute epoch microseconds**, not session-relative. "
        "That is what lets the hypertable prune by calendar date. A log's absolute start "
        "comes from its header line (UTC) — the filename is local Philadelphia time, and "
        "`pcm.startTime` is an RTC uptime counter that never got set."
    ),
    CODE(
        "import time\n"
        "\n"
        "import plotly.graph_objects as go\n"
        "import polars as pl\n"
        "\n"
        "from perda.timeline.client import TimelineClient\n"
        "\n"
        "pl.Config.set_tbl_rows(20)\n"
        "pl.Config.set_fmt_str_lengths(60)\n"
        "\n"
        "tl = TimelineClient()\n"
        "tl.overview()"
    ),
    MD(
        "## 1. The variable catalog\n"
        "\n"
        "Built from the header block of every log. `dtype` is inferred from observed "
        "values and only ever widens (`bool` → `int` → `float`), so a variable that "
        "happens to sit in {0,1} for one short session cannot be mislabelled forever."
    ),
    CODE(
        "print('catalog size:', tl.sql('SELECT count(*) AS n FROM timeline_variables')['n'][0])\n"
        "display(tl.sql('SELECT dtype, count(*) AS n FROM timeline_variables GROUP BY dtype ORDER BY n DESC'))\n"
        "tl.search('cell voltage', limit=8)"
    ),
    MD(
        "Note how much of the catalog is `bool` and `int`. The car's telemetry is "
        "overwhelmingly discrete — fault flags, states, counters — and it is logged "
        "on change rather than at a fixed rate. That combination is why the columnstore "
        "gets it down to roughly 4 bytes per sample, and also why a naive "
        "\"downsample to 1 Hz\" tier would save nothing: most variables already "
        "average only a couple of samples per second."
    ),
    MD(
        "## 2. The actual question\n"
        "\n"
        "> *\"Find me incidents over all testing sessions in May where the minimum cell "
        "voltage sagged.\"*\n"
        "\n"
        "This is answered entirely from `session_var_stats`. **Zero raw samples are read.**"
    ),
    CODE(
        "VAR = 'bms.stack.mma.cellV.min'\n"
        "\n"
        "started = time.perf_counter()\n"
        "hits = tl.find(VAR, below=3.2, month='2026-05', min_samples=100)\n"
        "elapsed = (time.perf_counter() - started) * 1000\n"
        "\n"
        "print(f'{hits.height} sessions matched in {elapsed:.1f} ms')\n"
        "hits.head(12)"
    ),
    MD(
        "Same shape of question, different signal — nothing about the query needs to know "
        "which variable it is, so this generalises to any of the ~1,100 signals the car logs."
    ),
    CODE(
        "started = time.perf_counter()\n"
        "fast = tl.find('pcm.moc.motor.wheelSpeed', above=15.0, month='2026-05')\n"
        "print(f'{fast.height} sessions above 15 m/s in {(time.perf_counter()-started)*1000:.1f} ms')\n"
        "fast.head(8)"
    ),
    MD(
        "## 3. What a season-wide view finds that a single log cannot\n"
        "\n"
        "Every value in a PER log is written into a float32 field. When firmware writes a "
        "raw integer instead of a measurement, the bit pattern decodes as a tiny denormal.\n"
        "\n"
        "`pcm.pedals.accel` emits `2.8026e-45` — bit pattern `2` — to mean *pedal reading "
        "invalid*. That value **looks like zero to every naive aggregate**, so an invalid "
        "pedal silently reads as \"pedal not pressed\".\n"
        "\n"
        "The timeline flags these as `n_invalid` and excludes them from every statistic."
    ),
    CODE(
        "report = tl.invalid_report(min_fraction=0.001)\n"
        "report.select(['var_key', 'dtype', 'sentinel_bits', 'invalid_fraction',\n"
        "               'sessions_affected', 'sessions_total', 'n_invalid'])"
    ),
    CODE(
        "# Per-session breakdown for the pedal: some sessions are entirely invalid.\n"
        "pedal = tl.stats('pcm.pedals.accel', month='2026-05').with_columns(\n"
        "    (pl.col('n_invalid') / pl.col('n')).alias('invalid_frac')\n"
        ")\n"
        "fig = go.Figure(go.Bar(x=pedal['start_utc'], y=pedal['invalid_frac']))\n"
        "fig.update_layout(\n"
        "    title='pcm.pedals.accel — fraction of samples that are the invalid sentinel',\n"
        "    xaxis_title='session start (UTC)', yaxis_title='invalid fraction',\n"
        "    height=380, yaxis_tickformat='.0%',\n"
        ")\n"
        "fig.show()"
    ),
    MD(
        "## 4. Drilling down to raw samples\n"
        "\n"
        "Only now do we touch the hypertable — and only for the sessions the summary tier "
        "already selected. `load()` returns a PERDA `DataInstance`, so anything downstream "
        "(arithmetic, joins, `Analyzer` plotting) works unchanged."
    ),
    CODE(
        "# Among the matching sessions, take the one with the most samples of this\n"
        "# signal -- the shortest session would make for a thin plot.\n"
        "target = int(\n"
        "    hits.sort('n', descending=True)['session_id'][0] if hits.height\n"
        "    else tl.sql('SELECT session_id FROM timeline_sessions ORDER BY n_rows DESC LIMIT 1')['session_id'][0]\n"
        ")\n"
        "\n"
        "started = time.perf_counter()\n"
        "signal = tl.load(VAR, session_id=target)\n"
        "print(f'{len(signal.value_np):,} samples in {(time.perf_counter()-started)*1000:.0f} ms')\n"
        "print(type(signal).__name__, '->', signal.cpp_name)\n"
        "\n"
        "fig = go.Figure(go.Scatter(x=signal.timestamp_np / 1e6, y=signal.value_np, mode='lines'))\n"
        "fig.update_layout(\n"
        "    title=f'{VAR} — session {target}',\n"
        "    xaxis_title='seconds into session', yaxis_title='volts', height=380,\n"
        ")\n"
        "fig.show()"
    ),
    MD(
        "## 5. The season at a glance\n"
        "\n"
        "One row per session, straight from the summary tier — the view that simply did "
        "not exist before."
    ),
    CODE(
        "season = tl.stats(VAR, month='2026-05')\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=season['start_utc'], y=season['v_min'],\n"
        "                         mode='markers', name='min'))\n"
        "fig.add_trace(go.Scatter(x=season['start_utc'], y=season['v_p50'],\n"
        "                         mode='markers', name='median'))\n"
        "fig.add_trace(go.Scatter(x=season['start_utc'], y=season['v_max'],\n"
        "                         mode='markers', name='max'))\n"
        "fig.update_layout(title=f'{VAR} across every May session',\n"
        "                  xaxis_title='session start (UTC)', yaxis_title='volts', height=420)\n"
        "fig.show()"
    ),
    MD(
        "## 6. Raw SQL — the surface the LLM layer will target\n"
        "\n"
        "Three stable views (`v_sessions`, `v_stats`, `v_samples`) hide ids, epoch "
        "microseconds and partition layout. A text-to-SQL model writes against these "
        "names and never needs to know the storage design."
    ),
    CODE(
        "tl.sql('''\n"
        "    SELECT test_day,\n"
        "           count(DISTINCT session_id) AS sessions,\n"
        "           round(max(v_max) FILTER (WHERE var_key = 'bms.pack.current')::numeric, 1)\n"
        "               AS peak_pack_current,\n"
        "           round(min(v_min) FILTER (WHERE var_key = 'bms.stack.mma.cellV.min')::numeric, 3)\n"
        "               AS lowest_cell_v\n"
        "    FROM v_stats\n"
        "    WHERE var_key IN ('bms.pack.current', 'bms.stack.mma.cellV.min')\n"
        "    GROUP BY test_day\n"
        "    ORDER BY test_day\n"
        "''')"
    ),
    CODE(
        "# Which variables were most *active* (value changes per second) across the month?\n"
        "tl.sql('''\n"
        "    SELECT var_key, dtype,\n"
        "           round(avg(hz)::numeric, 1)                     AS avg_hz,\n"
        "           sum(n_changes)                                  AS total_changes,\n"
        "           round(max(max_gap_us)/1e6::numeric, 1)          AS worst_gap_s\n"
        "    FROM v_stats\n"
        "    WHERE dtype = 'bool'\n"
        "    GROUP BY var_key, dtype\n"
        "    HAVING sum(n_changes) > 0\n"
        "    ORDER BY total_changes DESC\n"
        "    LIMIT 10\n"
        "''')"
    ),
    MD(
        "## 7. Storage and speed\n"
        "\n"
        "The numbers that decide whether this is worth running."
    ),
    CODE(
        "display(tl.sizes())\n"
        "\n"
        "totals = tl.sql('''\n"
        "    SELECT count(*) AS sessions, sum(n_rows) AS samples,\n"
        "           round(sum(source_bytes)/1e9::numeric, 1) AS source_gb\n"
        "    FROM timeline_sessions\n"
        "''')\n"
        "display(totals)\n"
        "\n"
        "rows = int(totals['samples'][0])\n"
        "source_bytes = float(totals['source_gb'][0]) * 1e9\n"
        "compressed = float(tl.sql(\"SELECT hypertable_size('timeline_samples') AS b\")['b'][0])\n"
        "print(f'\\n{rows:,} samples, {compressed/rows:.2f} bytes per sample '\n"
        "      f'({source_bytes/compressed:.1f}x smaller than the source CSV)')"
    ),
    CODE(
        "def timed(fn, repeats=3):\n"
        "    runs = []\n"
        "    for _ in range(repeats):\n"
        "        started = time.perf_counter()\n"
        "        fn()\n"
        "        runs.append((time.perf_counter() - started) * 1000)\n"
        "    return min(runs)\n"
        "\n"
        "day = tl.sql(\n"
        "    'SELECT test_day FROM timeline_sessions WHERE session_id = %s', (target,)\n"
        ")['test_day'][0]\n"
        "\n"
        "cases = {\n"
        "    'threshold scan (summary tier)':\n"
        "        lambda: tl.find(VAR, below=3.2, month='2026-05'),\n"
        "    'per-session stats, one month':\n"
        "        lambda: tl.stats(VAR, month='2026-05'),\n"
        "    'one variable, one session (raw)':\n"
        "        lambda: tl.samples(VAR, session_id=target),\n"
        "    'one variable, one whole test day':\n"
        "        lambda: tl.samples(VAR, test_day=day),\n"
        "    'full raw scan (no pruning possible)':\n"
        "        lambda: tl.sql('SELECT count(*), avg(value) FROM timeline_samples'),\n"
        "}\n"
        "\n"
        "for label, fn in cases.items():\n"
        "    print(f'{label:36s} {timed(fn):8.1f} ms')"
    ),
    MD(
        "---\n"
        "\n"
        "### Where this goes next\n"
        "\n"
        "1. **Backfill the rest of REV 11** — February through June, ~1,865 logs / 218 GB. "
        "Ingest runs at roughly 150k samples/s per worker and the store lands near 25 GB.\n"
        "2. **Hook it to the ingest pipeline** — one more Celery task after `autotag`, so "
        "new logs join the timeline automatically.\n"
        "3. **Point a text-to-SQL model at the three views**, with the variable catalog as "
        "retrieval context.\n"
        "4. **Cross-revision aliasing** (`ams.*` → `bms.*`) if the 2022–2025 logs are worth "
        "pulling in — the only piece that needs hand-verification.\n"
        "\n"
        "Open design question: whether `perda.timeline` stays a client to a shared "
        "TimescaleDB, or also ships a self-contained reader so a teammate can use it "
        "without a server."
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "perda", "language": "python", "name": "perda"},
    "language_info": {"name": "python"},
}
nbf.write(notebook, "_scratch/notebooks/timeline_demo.ipynb")
print("wrote _scratch/notebooks/timeline_demo.ipynb")
