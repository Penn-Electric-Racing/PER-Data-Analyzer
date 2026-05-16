import operator

import numpy as np
import pytest
from pydantic import ValidationError

from perda.core_data_structures.data_instance import (
    DataInstance,
    FilterOptions,
    apply_ufunc_filter,
    apply_ufunc_inner_join,
    apply_ufunc_left_join,
    apply_ufunc_outer_join,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "timestamp_np": np.array([[0, 1], [2, 3]], dtype=np.int64),
                "value_np": np.array([0.0, 1.0, 2.0, 3.0]),
            },
            id="timestamp_2d",
        ),
        pytest.param(
            {
                "timestamp_np": np.array([0, 1], dtype=np.int64),
                "value_np": np.array([[0.0, 1.0], [2.0, 3.0]]),
            },
            id="value_2d",
        ),
        pytest.param(
            {
                "timestamp_np": np.array([-1, 0, 1], dtype=np.int64),
                "value_np": np.array([0.0, 1.0, 2.0]),
            },
            id="negative_timestamp",
        ),
        pytest.param(
            {
                "timestamp_np": np.array([3, 2, 1], dtype=np.int64),
                "value_np": np.array([0.0, 1.0, 2.0]),
            },
            id="decreasing_timestamp",
        ),
        pytest.param(
            {
                "timestamp_np": np.array([0, 1, 2], dtype=np.int64),
                "value_np": np.array([0.0, 1.0]),
            },
            id="mismatched_lengths",
        ),
        pytest.param(
            {"timestamp_np": [0, 1, 2], "value_np": [0.0, 1.0, 2.0]},
            id="list_input_rejected",
        ),
    ],
)
def test_validation_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        DataInstance(**kwargs)


def test_timestamp_coerced_to_int64():
    di = DataInstance(
        timestamp_np=np.array([0.0, 1.0, 2.0]),
        value_np=np.array([0.0, 1.0, 2.0]),
    )
    assert di.timestamp_np.dtype == np.int64


def test_value_coerced_to_float64():
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([0, 1, 2]),
    )
    assert di.value_np.dtype == np.float64


def test_str_representation(di_with_metadata):
    s = str(di_with_metadata)
    assert "meta" in s
    assert "99" in s
    assert "test.meta" in s


def test_len_returns_array_length(di_simple):
    assert len(di_simple) == 4


@pytest.mark.parametrize(
    "op, scalar, expected",
    [
        pytest.param(operator.add, 3.0, [3.0, 4.0, 5.0, 6.0], id="add"),
        pytest.param(operator.sub, 1.0, [-1.0, 0.0, 1.0, 2.0], id="sub"),
        pytest.param(operator.mul, 2.0, [0.0, 2.0, 4.0, 6.0], id="mul"),
        pytest.param(operator.truediv, 2.0, [0.0, 0.5, 1.0, 1.5], id="truediv"),
        pytest.param(operator.pow, 2, [0.0, 1.0, 4.0, 9.0], id="pow"),
    ],
)
def test_scalar_op_values(di_simple, op, scalar, expected):
    result = op(di_simple, scalar)
    np.testing.assert_allclose(result.value_np, expected)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
)
def test_scalar_op_preserves_metadata(di_simple, op):
    result = op(di_simple, 1.0)
    assert result.label == di_simple.label
    assert result.var_id == di_simple.var_id


def test_neg_negates_all_values(di_simple):
    result = -di_simple
    np.testing.assert_allclose(result.value_np, -di_simple.value_np)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)


@pytest.mark.parametrize(
    "op, bad_value",
    [
        pytest.param(operator.add, "not_a_number", id="add_str"),
        pytest.param(operator.sub, "x", id="sub_str"),
        pytest.param(operator.mul, [1, 2], id="mul_list"),
        pytest.param(operator.truediv, None, id="truediv_none"),
        pytest.param(operator.pow, "two", id="pow_str"),
    ],
)
def test_scalar_op_invalid_type_raises(di_simple, op, bad_value):
    with pytest.raises(TypeError):
        op(di_simple, bad_value)


@pytest.mark.parametrize(
    "op, a_ts, a_vals, b_ts, b_vals, expected_vals",
    [
        pytest.param(
            operator.sub,
            np.array([0, 2, 4], dtype=np.int64),
            np.array([10.0, 20.0, 30.0]),
            np.array([0, 4], dtype=np.int64),
            np.array([0.0, 20.0]),
            [10.0, 10.0, 10.0],  # b interpolated at ts=2 -> 10.0
            id="sub",
        ),
        pytest.param(
            operator.mul,
            np.array([0, 2], dtype=np.int64),
            np.array([2.0, 4.0]),
            np.array([0, 2], dtype=np.int64),
            np.array([3.0, 5.0]),
            [6.0, 20.0],
            id="mul",
        ),
        pytest.param(
            operator.truediv,
            np.array([0, 2], dtype=np.int64),
            np.array([4.0, 8.0]),
            np.array([0, 2], dtype=np.int64),
            np.array([2.0, 4.0]),
            [2.0, 2.0],
            id="truediv",
        ),
        pytest.param(
            operator.pow,
            np.array([0, 1], dtype=np.int64),
            np.array([2.0, 3.0]),
            np.array([0, 1], dtype=np.int64),
            np.array([2.0, 3.0]),
            [4.0, 27.0],
            id="pow",
        ),
    ],
)
def test_binary_op_values(op, a_ts, a_vals, b_ts, b_vals, expected_vals):
    a = DataInstance(timestamp_np=a_ts, value_np=a_vals)
    b = DataInstance(timestamp_np=b_ts, value_np=b_vals)
    result = op(a, b)
    np.testing.assert_allclose(result.value_np, expected_vals)


def test_add_two_instances_timestamps_match_left(di_simple, di_sparse):
    result = di_simple + di_sparse
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)


@pytest.mark.parametrize("op", [operator.add, operator.mul])
def test_binary_op_preserves_left_metadata(di_simple, di_sparse, op):
    result = op(di_simple, di_sparse)
    assert result.label == di_simple.label
    assert result.var_id == di_simple.var_id


def test_trim_both_bounds(di_simple):
    result = di_simple.trim(ts_start=1, ts_end=2)
    np.testing.assert_array_equal(result.timestamp_np, [1, 2])
    np.testing.assert_allclose(result.value_np, [1.0, 2.0])


def test_trim_start_only(di_simple):
    result = di_simple.trim(ts_start=2)
    np.testing.assert_array_equal(result.timestamp_np, [2, 3])


def test_trim_end_only(di_simple):
    result = di_simple.trim(ts_end=1)
    np.testing.assert_array_equal(result.timestamp_np, [0, 1])


def test_trim_exact_boundary_inclusive(di_simple):
    result = di_simple.trim(ts_start=0, ts_end=3)
    assert len(result) == len(di_simple)


def test_trim_result_is_empty_when_range_has_no_points(di_simple):
    result = di_simple.trim(ts_start=0, ts_end=0)
    assert len(result) == 1
    result_empty = di_simple.trim(ts_start=4, ts_end=10)
    assert len(result_empty) == 0


def test_trim_preserves_metadata(di_with_metadata):
    result = di_with_metadata.trim(ts_start=1, ts_end=2)
    assert result.label == di_with_metadata.label
    assert result.var_id == di_with_metadata.var_id
    assert result.cpp_name == di_with_metadata.cpp_name


def test_trim_returns_new_instance(di_simple):
    result = di_simple.trim(ts_start=1, ts_end=2)
    assert result is not di_simple


def test_filter_values_keeps_matching_rows(di_simple):
    result = apply_ufunc_filter(di_simple, lambda v: v > 1.0)
    np.testing.assert_array_equal(result.value_np, [2.0, 3.0])
    np.testing.assert_array_equal(result.timestamp_np, [2, 3])


def test_filter_timestamps_keeps_matching_rows(di_simple):
    result = apply_ufunc_filter(
        di_simple, lambda ts: ts >= 2, apply_to=FilterOptions.TIMESTAMPS
    )
    np.testing.assert_array_equal(result.timestamp_np, [2, 3])


def test_filter_both_receives_timestamp_and_value(di_simple):
    result = apply_ufunc_filter(
        di_simple,
        lambda ts, v: (ts > 0) & (v < 3.0),
        apply_to=FilterOptions.BOTH,
    )
    np.testing.assert_array_equal(result.timestamp_np, [1, 2])


def test_filter_all_false_returns_empty(di_simple):
    result = apply_ufunc_filter(di_simple, lambda v: v > 1000.0)
    assert len(result) == 0


def test_filter_all_true_returns_unchanged(di_simple):
    result = apply_ufunc_filter(di_simple, lambda v: v >= 0)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)
    np.testing.assert_array_equal(result.value_np, di_simple.value_np)


def test_filter_preserves_metadata(di_with_metadata):
    result = apply_ufunc_filter(di_with_metadata, lambda v: v > 0)
    assert result.label == di_with_metadata.label
    assert result.var_id == di_with_metadata.var_id
    assert result.cpp_name is None  # apply_ufunc_filter does not forward cpp_name


def test_filter_values_default_apply_to(di_simple):
    result = apply_ufunc_filter(di_simple, lambda v: v == 0.0)
    assert len(result) == 1
    assert result.value_np[0] == 0.0


def test_apply_ufunc_left_join_add_matches_operator(di_simple, di_sparse):
    result_ufunc = apply_ufunc_left_join(di_simple, di_sparse, np.add)
    result_op = di_simple + di_sparse
    np.testing.assert_allclose(result_ufunc.value_np, result_op.value_np)
    np.testing.assert_array_equal(result_ufunc.timestamp_np, result_op.timestamp_np)


def test_apply_ufunc_left_join_timestamps_equal_left(di_simple, di_sparse):
    result = apply_ufunc_left_join(di_simple, di_sparse, np.add)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)


def test_apply_ufunc_outer_join_timestamps_are_union():
    a = DataInstance(
        timestamp_np=np.array([0, 2], dtype=np.int64), value_np=np.array([0.0, 2.0])
    )
    b = DataInstance(
        timestamp_np=np.array([1, 3], dtype=np.int64), value_np=np.array([1.0, 3.0])
    )
    result = apply_ufunc_outer_join(a, b, np.add, drop_nan=False, fill=0.0)
    np.testing.assert_array_equal(result.timestamp_np, [0, 1, 2, 3])


def test_apply_ufunc_outer_join_drop_nan_true_no_nans():
    a = DataInstance(
        timestamp_np=np.array([0, 4], dtype=np.int64), value_np=np.array([0.0, 4.0])
    )
    b = DataInstance(
        timestamp_np=np.array([0, 4], dtype=np.int64), value_np=np.array([0.0, 4.0])
    )
    result = apply_ufunc_outer_join(a, b, np.add, drop_nan=True)
    assert not np.any(np.isnan(result.value_np))


def test_apply_ufunc_outer_join_drop_nan_false_uses_fill():
    a = DataInstance(
        timestamp_np=np.array([0, 2], dtype=np.int64), value_np=np.array([1.0, 3.0])
    )
    b = DataInstance(
        timestamp_np=np.array([5, 10], dtype=np.int64), value_np=np.array([10.0, 20.0])
    )
    result = apply_ufunc_outer_join(a, b, np.add, drop_nan=False, fill=0.0)
    assert not np.any(np.isnan(result.value_np))


def test_apply_ufunc_inner_join_tolerance_zero_exact_only():
    a = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    b = DataInstance(
        timestamp_np=np.array([0, 2], dtype=np.int64), value_np=np.array([10.0, 30.0])
    )
    result = apply_ufunc_inner_join(a, b, np.add, tolerance=0)
    np.testing.assert_array_equal(result.timestamp_np, [0, 2])


def test_apply_ufunc_inner_join_tolerance_filters_unmatched():
    a = DataInstance(
        timestamp_np=np.array([0, 5, 10], dtype=np.int64),
        value_np=np.array([0.0, 5.0, 10.0]),
    )
    b = DataInstance(
        timestamp_np=np.array([0, 10], dtype=np.int64), value_np=np.array([0.0, 10.0])
    )
    result = apply_ufunc_inner_join(a, b, np.add, tolerance=3)
    np.testing.assert_array_equal(result.timestamp_np, [0, 10])
