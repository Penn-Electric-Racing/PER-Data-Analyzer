import textwrap
from datetime import datetime

import numpy as np
import pytest

from perda.analyzer.csv import parse_csv, parse_header_creation_time
from perda.units import Timescale


def test_parse_csv_ms_header_unit(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    assert srd.timestamp_unit == Timescale.MS


def test_parse_csv_us_header_unit(us_csv):
    srd = parse_csv(us_csv, verbose=0)
    assert srd.timestamp_unit == Timescale.US


def test_parse_csv_variable_registered_by_name(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    assert "ams.pack.voltage" in srd.cpp_name_to_id


def test_parse_csv_no_descript_cpp_name_used(no_descript_csv):
    srd = parse_csv(no_descript_csv, verbose=0)
    assert "ams.pack.voltage" in srd.cpp_name_to_id


def test_parse_csv_two_variables_both_present(two_var_csv):
    srd = parse_csv(two_var_csv, verbose=0)
    assert "ams.pack.voltage" in srd
    assert "ams.pack.current" in srd


def test_parse_csv_values_correct(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    di = srd["ams.pack.voltage"]
    np.testing.assert_allclose(di.value_np, [12.5, 12.6, 12.7])


def test_parse_csv_timestamps_sorted(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    ts = srd["ams.pack.voltage"].timestamp_np
    assert np.all(ts[:-1] <= ts[1:])


def test_parse_csv_data_start_end_time(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    assert srd.data_start_time == 0
    assert srd.data_end_time == 2000


def test_parse_csv_total_data_points(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    assert srd.total_data_points == 3


def test_parse_csv_two_vars_total_points(two_var_csv):
    srd = parse_csv(two_var_csv, verbose=0)
    assert srd.total_data_points == 4


def test_parse_csv_ts_offset_shifts_timestamps(ts_offset_csv):
    srd_no_offset = parse_csv(ts_offset_csv, verbose=0)
    srd_offset = parse_csv(ts_offset_csv, ts_offset=500, verbose=0)
    orig_ts = srd_no_offset["test.sig"].timestamp_np
    shifted_ts = srd_offset["test.sig"].timestamp_np
    np.testing.assert_array_equal(shifted_ts, orig_ts + 500)


def test_parse_csv_ts_offset_does_not_change_values(ts_offset_csv):
    srd_no_offset = parse_csv(ts_offset_csv, verbose=0)
    srd_offset = parse_csv(ts_offset_csv, ts_offset=1000, verbose=0)
    np.testing.assert_allclose(
        srd_no_offset["test.sig"].value_np,
        srd_offset["test.sig"].value_np,
    )


def test_parse_csv_empty_data_raises(tmp_path):
    content = textwrap.dedent(
        """\
        Log file header
        Value voltage (ams.pack.voltage): 1
    """
    )
    p = tmp_path / "empty.csv"
    p.write_text(content)
    with pytest.raises(Exception):
        parse_csv(str(p), verbose=0)


def test_parse_csv_verbose_prints_header(ms_csv, capsys):
    parse_csv(ms_csv, verbose=1)
    out = capsys.readouterr().out
    assert "Header" in out or "Timestamp" in out


@pytest.mark.parametrize(
    "header, expected",
    [
        pytest.param(
            "PER Log: Thu Jun 11 17:06:37 2026 v2.0",
            datetime(2026, 6, 11, 17, 6, 37),
            id="with_version_suffix",
        ),
        pytest.param(
            "PER Log: Thu Jun 11 17:06:37 2026",
            datetime(2026, 6, 11, 17, 6, 37),
            id="without_version_suffix",
        ),
        pytest.param("Generic Header With No Date", None, id="no_per_prefix"),
        pytest.param("PER Log: not a real date", None, id="unparseable_date"),
    ],
)
def test_parse_header_creation_time(header, expected):
    assert parse_header_creation_time(header) == expected


def test_parse_csv_sets_creation_time(write_log):
    path = write_log("dated.csv", "PER Log: Thu Jun 11 17:06:37 2026 v2.0")
    srd = parse_csv(str(path), verbose=0)
    assert srd.creation_time == datetime(2026, 6, 11, 17, 6, 37)


def test_parse_csv_no_date_header_graceful_fallback(write_log):
    path = write_log("undated.csv", "Standard Generic Log File Header With No Date")
    assert parse_csv(str(path), verbose=0).creation_time is None
