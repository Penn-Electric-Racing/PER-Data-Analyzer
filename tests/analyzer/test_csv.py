import os
import textwrap

import numpy as np
import pytest

from perda.analyzer.csv import parse_csv
from perda.units import Timescale

# ---------------------------------------------------------------------------
# Fixtures: minimal CSV files written to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def ms_csv(tmp_path):
    """Minimal v1 (ms) CSV: one variable, three data rows."""
    content = textwrap.dedent(
        """\
        Log file header
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
        1000,1,12.6
        2000,1,12.7
    """
    )
    p = tmp_path / "test_ms.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture()
def us_csv(tmp_path):
    """Minimal v2.0 (us) CSV: one variable, two data rows."""
    content = textwrap.dedent(
        """\
        Log file header v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
        1000000,1,12.7
    """
    )
    p = tmp_path / "test_us.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture()
def two_var_csv(tmp_path):
    """CSV with two variables interleaved."""
    content = textwrap.dedent(
        """\
        Log file header
        Value voltage (ams.pack.voltage): 1
        Value current (ams.pack.current): 2
        0,1,12.5
        0,2,100.0
        1000,1,12.6
        1000,2,101.0
    """
    )
    p = tmp_path / "test_two_var.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture()
def no_descript_csv(tmp_path):
    """CSV where variable name has no parenthesised description."""
    content = textwrap.dedent(
        """\
        Log file header
        Value ams.pack.voltage: 1
        0,1,3.7
        500,1,3.8
    """
    )
    p = tmp_path / "test_no_desc.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture()
def ts_offset_csv(tmp_path):
    """CSV to test ts_offset parameter."""
    content = textwrap.dedent(
        """\
        Log file header
        Value sig (test.sig): 1
        100,1,1.0
        200,1,2.0
    """
    )
    p = tmp_path / "test_offset.csv"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# Timestamp unit detection
# ---------------------------------------------------------------------------


def test_parse_csv_ms_header_unit(ms_csv):
    srd = parse_csv(ms_csv, verbose=0)
    assert srd.timestamp_unit == Timescale.MS


def test_parse_csv_us_header_unit(us_csv):
    srd = parse_csv(us_csv, verbose=0)
    assert srd.timestamp_unit == Timescale.US


# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data values
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ts_offset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Empty / error cases
# ---------------------------------------------------------------------------


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
