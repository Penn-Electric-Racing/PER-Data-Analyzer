import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale


def test_data_instance_str_reports_stats():
    di = DataInstance(
        timestamp_np=np.array([0, 500, 1000], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        label="test",
    )
    summary = str(di)
    assert summary.startswith("test | ID: None")
    assert "Count:      3" in summary
    assert "Time range: 0 to 1000 (raw)" in summary
    assert "Average:    2.0000" in summary


def test_data_instance_str_empty():
    di = DataInstance(
        timestamp_np=np.array([], dtype=np.int64),
        value_np=np.array([], dtype=np.float64),
        label="empty",
    )
    assert "Empty" in str(di)


@pytest.mark.parametrize(
    "vals, expected_min, expected_max",
    [
        pytest.param([3.0, 1.0, 2.0], 1.0, 3.0, id="unsorted_values"),
        pytest.param([5.0, 5.0, 5.0], 5.0, 5.0, id="constant"),
    ],
)
def test_data_instance_str_min_max(vals, expected_min, expected_max):
    di = DataInstance(
        timestamp_np=np.arange(len(vals), dtype=np.int64),
        value_np=np.array(vals, dtype=np.float64),
    )
    summary = str(di)
    assert f"Min:        {expected_min:.4f}" in summary
    assert f"Max:        {expected_max:.4f}" in summary


def test_single_run_data_str(srd_basic):
    summary = str(srd_basic)
    assert "Data Summary" in summary
    assert "Total Variable:     2" in summary


def test_single_run_data_str_converts_us_to_seconds():
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
    assert "0.0 to 1.0 (s)" in str(srd)
