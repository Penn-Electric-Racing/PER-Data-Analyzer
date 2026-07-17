import textwrap
from datetime import datetime
import pytest

from perda.analyzer.csv import parse_csv
from perda.analyzer.run_collection import RunCollection

def test_creation_time_parsing(tmp_path):
    # Header format: "PER Log: Thu Jun 11 17:06:37 2026 v2.0"
    content = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 17:06:37 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
        1000,1,12.6
    """
    )
    p = tmp_path / "test_date.csv"
    p.write_text(content)

    srd = parse_csv(str(p), verbose=0)
    assert srd.creation_time == datetime(2026, 6, 11, 17, 6, 37)


def test_no_date_header_graceful_fallback(tmp_path):
    content = textwrap.dedent(
        """\
        Standard Generic Log File Header With No Date
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
    """
    )
    p = tmp_path / "no_date.csv"
    p.write_text(content)

    srd = parse_csv(str(p), verbose=0)
    assert srd.creation_time is None


def test_run_collection_sorting_and_filtering(tmp_path):
    # Older log
    content1 = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 10:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
    """
    )
    p1 = tmp_path / "log_old.csv"
    p1.write_text(content1)

    # Newer log
    content2 = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 12:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,13.0
    """
    )
    p2 = tmp_path / "log_new.csv"
    p2.write_text(content2)

    # Load with order p2, p1 to ensure sorting works
    col = RunCollection([str(p2), str(p1)])

    # Check chronological ordering: old (10:00) should be first
    assert len(col) == 2
    assert col.runs[0].data.creation_time == datetime(2026, 6, 11, 10, 0, 0)
    assert col.runs[1].data.creation_time == datetime(2026, 6, 11, 12, 0, 0)

    # Check date filtering
    filtered = col.filter_by_date(
        datetime(2026, 6, 11, 9, 0, 0),
        datetime(2026, 6, 11, 11, 0, 0)
    )
    assert len(filtered) == 1
    assert filtered.runs[0].data.creation_time == datetime(2026, 6, 11, 10, 0, 0)


def test_run_collection_check_any(tmp_path):
    content1 = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 10:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,5.0
        1000,1,6.0
    """
    )
    p1 = tmp_path / "log1.csv"
    p1.write_text(content1)

    content2 = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 12:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,15.0
    """
    )
    p2 = tmp_path / "log2.csv"
    p2.write_text(content2)

    col = RunCollection([str(p1), str(p2)])

    # Query if voltage ever exceeded 10.0
    matching = col.check_any("ams.pack.voltage", lambda val: (val > 10.0).any())
    assert matching == ["log2.csv"]


def test_run_collection_direct_init(tmp_path):
    content = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 10:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
    """
    )
    p = tmp_path / "log.csv"
    p.write_text(content)

    col = RunCollection(str(p))
    assert len(col) == 1
    assert col.runs[0].data.creation_time == datetime(2026, 6, 11, 10, 0, 0)
