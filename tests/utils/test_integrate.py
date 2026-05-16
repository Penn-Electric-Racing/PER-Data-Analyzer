import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.units import Timescale
from perda.utils.integrate import (
    average_over_time_range,
    get_data_slice_by_timestamp,
    integrate_over_time_range,
)


def test_integrate_constant_one_over_one_second():
    # 0..1000 ms = 1 s; constant 1.0 -> integral = 1.0 s
    di = DataInstance(
        timestamp_np=np.array([0, 1000], dtype=np.int64),
        value_np=np.array([1.0, 1.0], dtype=np.float64),
    )
    result = integrate_over_time_range(
        di, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    assert pytest.approx(result, abs=1e-9) == 1.0


def test_integrate_linear_ramp():
    # 0..1000 ms; values 0->1; area = 0.5 s
    di = DataInstance(
        timestamp_np=np.array([0, 1000], dtype=np.int64),
        value_np=np.array([0.0, 1.0], dtype=np.float64),
    )
    result = integrate_over_time_range(
        di, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    assert pytest.approx(result, abs=1e-9) == 0.5


def test_integrate_single_point_returns_zero():
    di = DataInstance(
        timestamp_np=np.array([500], dtype=np.int64),
        value_np=np.array([99.0], dtype=np.float64),
    )
    result = integrate_over_time_range(di)
    assert result == 0.0


def test_integrate_zero_length_returns_zero():
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    result = integrate_over_time_range(di, start_time=1, end_time=1)
    assert result == 0.0


def test_integrate_end_minus_one_uses_full_range():
    di = DataInstance(
        timestamp_np=np.array([0, 1000], dtype=np.int64),
        value_np=np.array([2.0, 2.0], dtype=np.float64),
    )
    full = integrate_over_time_range(
        di, end_time=-1, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    explicit = integrate_over_time_range(
        di, end_time=1, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    assert pytest.approx(full, abs=1e-9) == explicit


def test_integrate_us_to_s():
    # 0..1_000_000 us = 1 s; constant 3 -> integral = 3 s
    di = DataInstance(
        timestamp_np=np.array([0, 1_000_000], dtype=np.int64),
        value_np=np.array([3.0, 3.0], dtype=np.float64),
    )
    result = integrate_over_time_range(
        di, source_time_unit=Timescale.US, target_time_unit=Timescale.S
    )
    assert pytest.approx(result, abs=1e-6) == 3.0


@pytest.mark.parametrize(
    "ts, vals, start, end, expected",
    [
        pytest.param([0, 500, 1000], [0.0, 0.0, 0.0], 0, -1, 0.0, id="all_zeros"),
        # 5-point grid at 0,250,500,750,1000 ms -> 0,0.25,0.5,0.75,1 s;
        # constant 2.0; sub_range [0.25, 0.75] s -> integral = 2.0 * 0.5 = 1.0
        pytest.param(
            [0, 250, 500, 750, 1000],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            0.25,
            0.75,
            1.0,
            id="sub_range",
        ),
    ],
)
def test_integrate_parametrized(ts, vals, start, end, expected):
    di = DataInstance(
        timestamp_np=np.array(ts, dtype=np.int64),
        value_np=np.array(vals, dtype=np.float64),
    )
    result = integrate_over_time_range(
        di,
        start_time=start,
        end_time=end,
        source_time_unit=Timescale.MS,
        target_time_unit=Timescale.S,
    )
    assert result == pytest.approx(expected, abs=1e-9)


def test_average_constant_signal():
    di = DataInstance(
        timestamp_np=np.array([0, 1000], dtype=np.int64),
        value_np=np.array([5.0, 5.0], dtype=np.float64),
    )
    avg = average_over_time_range(
        di, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    assert pytest.approx(avg, abs=1e-9) == 5.0


def test_average_linear_ramp_midpoint():
    # 0->2 linearly; average should be 1.0
    di = DataInstance(
        timestamp_np=np.array([0, 1000], dtype=np.int64),
        value_np=np.array([0.0, 2.0], dtype=np.float64),
    )
    avg = average_over_time_range(
        di, source_time_unit=Timescale.MS, target_time_unit=Timescale.S
    )
    assert pytest.approx(avg, abs=1e-9) == 1.0


def test_average_single_point_returns_value():
    di = DataInstance(
        timestamp_np=np.array([42], dtype=np.int64),
        value_np=np.array([7.0], dtype=np.float64),
    )
    avg = average_over_time_range(di)
    assert avg == 7.0


def test_average_zero_time_range_returns_zero():
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    avg = average_over_time_range(di, start_time=5, end_time=5)
    assert avg == 0.0


def test_slice_start_only():
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    result = get_data_slice_by_timestamp(di, start_time=2, end_time=-1)
    np.testing.assert_array_equal(result.timestamp_np, [2, 3, 4])


def test_slice_bounded_range():
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    result = get_data_slice_by_timestamp(di, start_time=1, end_time=3)
    # end_time is exclusive
    np.testing.assert_array_equal(result.timestamp_np, [1, 2])


def test_slice_preserves_values(di_simple):
    result = get_data_slice_by_timestamp(di_simple, start_time=1, end_time=3)
    np.testing.assert_allclose(result.value_np, [1.0, 2.0])


def test_slice_returns_new_instance(di_simple):
    result = get_data_slice_by_timestamp(di_simple)
    assert result is not di_simple


def test_slice_empty_when_out_of_range(di_simple):
    result = get_data_slice_by_timestamp(di_simple, start_time=100, end_time=200)
    assert len(result) == 0
