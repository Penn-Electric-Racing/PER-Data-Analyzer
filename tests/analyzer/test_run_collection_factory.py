from datetime import datetime

import pytest

from perda.analyzer.run_collection_factory import (
    _read_run_metadata,
    from_directory,
    from_paths,
)


def test_from_paths_reads_header_date(write_log):
    path = write_log("log.csv", "PER Log: Thu Jun 11 17:06:37 2026 v2.0")
    col = from_paths([path])
    assert col.filenames == ["log.csv"]
    assert col._runs_by_date[0].date == datetime(2026, 6, 11, 17, 6, 37)


def test_from_paths_deduplicates(write_log):
    path = write_log("log.csv")
    assert len(from_paths([path, path, str(path)])) == 1


def test_from_paths_accepts_strings(write_log):
    assert from_paths([str(write_log("log.csv"))]).filenames == ["log.csv"]


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("Standard Generic Log File Header With No Date", id="no_per_prefix"),
        pytest.param("PER Log: not a real date v2.0", id="unparseable_date"),
    ],
)
def test_missing_header_date_falls_back_to_mtime(write_log, header):
    path = write_log("log.csv", header)
    metadata = _read_run_metadata(path)
    assert metadata.date == datetime.fromtimestamp(path.stat().st_mtime)


def test_from_directory_matches_pattern_only(write_log, tmp_path):
    write_log("log_a.csv")
    write_log("log_b.csv", "PER Log: Thu Jun 11 12:00:00 2026 v2.0")
    (tmp_path / "notes.txt").write_text("ignore me")

    assert from_directory(tmp_path).filenames == ["log_a.csv", "log_b.csv"]


def test_from_directory_recursive(write_log, tmp_path):
    write_log("log_a.csv")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "log_b.csv").write_text((tmp_path / "log_a.csv").read_text())

    assert len(from_directory(tmp_path)) == 1
    assert len(from_directory(tmp_path, recursive=True)) == 2


def test_from_directory_empty(tmp_path):
    assert len(from_directory(tmp_path)) == 0
