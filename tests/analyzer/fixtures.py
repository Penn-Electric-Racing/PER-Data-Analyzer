import textwrap

import pytest

from perda.analyzer.run_collection_factory import from_paths


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def write_log(tmp_path):
    """Factory writing a log file with a PER header, returning its path."""

    def _write(name, header="PER Log: Thu Jun 11 10:00:00 2026 v2.0", values=(12.5,)):
        lines = [header, "Value voltage (ams.pack.voltage): 1"]
        lines += [f"{i * 1000},1,{value}" for i, value in enumerate(values)]
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n")
        return path

    return _write


@pytest.fixture
def two_run_collection(write_log):
    """Collection of two logs, supplied newest-first to exercise chronological sorting."""
    older = write_log(
        "log_old.csv", "PER Log: Thu Jun 11 10:00:00 2026 v2.0", (5.0, 6.0)
    )
    newer = write_log("log_new.csv", "PER Log: Thu Jun 11 12:00:00 2026 v2.0", (15.0,))
    return from_paths([newer, older])
