import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale
from perda.utils.data_summary import data_instance_summary, single_run_summary


def test_data_instance_summary_runs_without_error(capsys):
    di = DataInstance(
        timestamp_np=np.array([0, 500, 1000], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        label="test",
    )
    data_instance_summary(
        di, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    out = capsys.readouterr().out
    assert "Count" in out
    assert "Min" in out
    assert "Max" in out
    assert "Average" in out


def test_data_instance_summary_empty_prints_empty_message(capsys):
    di = DataInstance(
        timestamp_np=np.array([], dtype=np.int64),
        value_np=np.array([], dtype=np.float64),
        label="empty",
    )
    data_instance_summary(di)
    out = capsys.readouterr().out
    assert "Empty" in out


def test_data_instance_summary_us_to_s(capsys):
    di = DataInstance(
        timestamp_np=np.array([0, 1_000_000], dtype=np.int64),
        value_np=np.array([0.0, 10.0], dtype=np.float64),
        label="us_sig",
    )
    data_instance_summary(
        di, source_time_unit=Timescale.US, target_time_unit=Timescale.S
    )
    out = capsys.readouterr().out
    assert "1.0000" in out


@pytest.mark.parametrize(
    "vals, expected_min, expected_max",
    [
        pytest.param([3.0, 1.0, 2.0], 1.0, 3.0, id="unsorted_values"),
        pytest.param([5.0, 5.0, 5.0], 5.0, 5.0, id="constant"),
    ],
)
def test_data_instance_summary_min_max_reported(
    capsys, vals, expected_min, expected_max
):
    di = DataInstance(
        timestamp_np=np.arange(len(vals), dtype=np.int64),
        value_np=np.array(vals, dtype=np.float64),
    )
    data_instance_summary(di)
    out = capsys.readouterr().out
    assert f"{expected_min:.4f}" in out
    assert f"{expected_max:.4f}" in out


def test_single_run_summary_runs_without_error(srd_basic, capsys):
    single_run_summary(srd_basic)
    out = capsys.readouterr().out
    assert "Data Summary" in out
    assert "Total Variable" in out


def test_single_run_summary_us_unit(capsys):
    srd = SingleRunData(
        id_to_instance={},
        cpp_name_to_id={},
        id_to_cpp_name={},
        id_to_descript={},
        total_data_points=0,
        data_start_time=0,
        data_end_time=1_000_000,
        timestamp_unit=Timescale.US,
    )
    single_run_summary(srd, time_unit=Timescale.S)
    out = capsys.readouterr().out
    assert "1.0" in out
