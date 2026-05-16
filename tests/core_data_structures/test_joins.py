import operator

import numpy as np
import pytest

from perda.core_data_structures.data_instance import (
    DataInstance,
    inner_join_data_instances,
    left_join_data_instances,
    outer_join_data_instances,
)
from perda.core_data_structures.joins import inner_join, left_join, outer_join
from perda.core_data_structures.resampling import ResampleMethod, resample_to_freq


@pytest.mark.parametrize(
    "method, right_ts, right_vals, query_ts, expected_rv",
    [
        pytest.param(
            ResampleMethod.LINEAR,
            np.array([0, 10], dtype=np.int64),
            np.array([0.0, 10.0]),
            np.array([0, 1, 2, 3], dtype=np.int64),
            [0.0, 1.0, 2.0, 3.0],
            id="linear",
        ),
        pytest.param(
            ResampleMethod.ZOH,
            np.array([0, 2], dtype=np.int64),
            np.array([0.0, 4.0]),
            np.array([0, 1, 2, 3, 4], dtype=np.int64),
            [0.0, 0.0, 4.0, 4.0, 4.0],
            id="zoh",
        ),
        pytest.param(
            ResampleMethod.NEAREST,
            np.array([0, 10], dtype=np.int64),
            np.array([0.0, 10.0]),
            np.array([0, 5, 10], dtype=np.int64),
            None,  # only check boundary values
            id="nearest",
        ),
    ],
)
def test_left_join_interpolation_method(
    method, right_ts, right_vals, query_ts, expected_rv
):
    left_vals = np.ones(len(query_ts))
    ts, _, rv = left_join(query_ts, left_vals, right_ts, right_vals, method=method)
    np.testing.assert_array_equal(ts, query_ts)
    if expected_rv is not None:
        np.testing.assert_allclose(rv, expected_rv)
    else:
        # NEAREST: just verify first and last clamp correctly
        assert rv[0] == right_vals[0]
        assert rv[-1] == right_vals[-1]


def test_left_join_cubic_exact_at_knots():
    xs = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    ys = xs.astype(float) ** 2
    _, _, rv = left_join(xs, ys, xs, ys, method=ResampleMethod.CUBIC)
    np.testing.assert_allclose(rv, ys, atol=1e-6)


def test_left_join_cubic_fallback_when_too_few_points():
    # < 4 source points must not raise; falls back to linear
    left_join(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0.0, 1.0, 2.0]),
        np.array([0, 2], dtype=np.int64),
        np.array([0.0, 2.0]),
        method=ResampleMethod.CUBIC,
    )


def test_left_join_preserves_left_timestamps(di_simple, di_sparse):
    ts, _, _ = left_join(
        di_simple.timestamp_np,
        di_simple.value_np,
        di_sparse.timestamp_np,
        di_sparse.value_np,
    )
    np.testing.assert_array_equal(ts, di_simple.timestamp_np)


def test_left_join_returns_copy_not_view():
    ts = np.array([0, 1, 2], dtype=np.int64)
    val = np.array([1.0, 2.0, 3.0])
    result_ts, _, _ = left_join(ts, val, ts, val)
    result_ts[0] = 999
    assert ts[0] == 0


def test_left_join_single_right_point_constant():
    left_ts = np.array([0, 1, 2, 3], dtype=np.int64)
    left_val = np.array([0.0, 1.0, 2.0, 3.0])
    right_ts = np.array([5], dtype=np.int64)
    right_val = np.array([99.0])
    _, _, rv = left_join(left_ts, left_val, right_ts, right_val)
    np.testing.assert_allclose(rv, [99.0, 99.0, 99.0, 99.0])


def test_left_join_right_wider_than_left_no_clamping():
    left_ts = np.array([2, 3, 4], dtype=np.int64)
    left_val = np.array([2.0, 3.0, 4.0])
    right_ts = np.array([0, 6], dtype=np.int64)
    right_val = np.array([0.0, 6.0])
    _, _, rv = left_join(left_ts, left_val, right_ts, right_val)
    np.testing.assert_allclose(rv, [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "join_fn, kwargs",
    [
        pytest.param(
            left_join,
            {
                "left_ts": np.array([], dtype=np.int64),
                "left_val": np.array([]),
                "right_ts": np.array([0], dtype=np.int64),
                "right_val": np.array([1.0]),
            },
            id="left_join_empty_left",
        ),
        pytest.param(
            left_join,
            {
                "left_ts": np.array([0, 1], dtype=np.int64),
                "left_val": np.array([0.0, 1.0]),
                "right_ts": np.array([], dtype=np.int64),
                "right_val": np.array([]),
            },
            id="left_join_empty_right",
        ),
        pytest.param(
            outer_join,
            {
                "left_ts": np.array([], dtype=np.int64),
                "left_val": np.array([]),
                "right_ts": np.array([0, 1], dtype=np.int64),
                "right_val": np.array([0.0, 1.0]),
            },
            id="outer_join_empty_left",
        ),
        pytest.param(
            outer_join,
            {
                "left_ts": np.array([0, 1], dtype=np.int64),
                "left_val": np.array([0.0, 1.0]),
                "right_ts": np.array([], dtype=np.int64),
                "right_val": np.array([]),
            },
            id="outer_join_empty_right",
        ),
        pytest.param(
            lambda **kw: inner_join(**kw, tolerance=1),
            {
                "left_ts": np.array([], dtype=np.int64),
                "left_val": np.array([]),
                "right_ts": np.array([0, 1], dtype=np.int64),
                "right_val": np.array([0.0, 1.0]),
            },
            id="inner_join_empty_left",
        ),
        pytest.param(
            lambda **kw: inner_join(**kw, tolerance=1),
            {
                "left_ts": np.array([0, 1], dtype=np.int64),
                "left_val": np.array([0.0, 1.0]),
                "right_ts": np.array([], dtype=np.int64),
                "right_val": np.array([]),
            },
            id="inner_join_empty_right",
        ),
    ],
)
def test_join_empty_input_raises(join_fn, kwargs):
    with pytest.raises(ValueError):
        join_fn(**kwargs)


def test_outer_join_timestamps_are_union():
    left_ts = np.array([0, 2, 4], dtype=np.int64)
    right_ts = np.array([1, 3, 5], dtype=np.int64)
    ts, _, _ = outer_join(
        left_ts, np.ones(3), right_ts, np.ones(3), drop_nan=False, fill=0.0
    )
    np.testing.assert_array_equal(ts, np.union1d(left_ts, right_ts))


def test_outer_join_identical_timestamps_no_extra_points():
    ts_shared = np.array([0, 1, 2], dtype=np.int64)
    val = np.array([1.0, 2.0, 3.0])
    ts, lv, rv = outer_join(ts_shared, val, ts_shared, val)
    np.testing.assert_array_equal(ts, ts_shared)
    np.testing.assert_allclose(lv, val)
    np.testing.assert_allclose(rv, val)


def test_outer_join_drop_nan_true_no_nans():
    left_ts = np.array([0, 2], dtype=np.int64)
    right_ts = np.array([5, 10], dtype=np.int64)
    ts, lv, rv = outer_join(
        left_ts,
        np.array([1.0, 2.0]),
        right_ts,
        np.array([5.0, 10.0]),
        drop_nan=True,
    )
    assert not np.any(np.isnan(lv))
    assert not np.any(np.isnan(rv))


def test_outer_join_values_interpolated_at_union_points():
    left_ts = np.array([0, 4], dtype=np.int64)
    right_ts = np.array([2, 4], dtype=np.int64)
    ts, lv, rv = outer_join(
        left_ts,
        np.array([0.0, 4.0]),
        right_ts,
        np.array([10.0, 20.0]),
    )
    idx2 = np.where(ts == 2)[0][0]
    np.testing.assert_allclose(lv[idx2], 2.0)


def test_inner_join_zero_tolerance_exact_matches_only():
    left_ts = np.array([0, 1, 2, 3], dtype=np.int64)
    right_ts = np.array([0, 2], dtype=np.int64)
    ts, lv, rv = inner_join(
        left_ts,
        np.array([0.0, 1.0, 2.0, 3.0]),
        right_ts,
        np.array([0.0, 20.0]),
        tolerance=0,
    )
    np.testing.assert_array_equal(ts, [0, 2])


@pytest.mark.parametrize(
    "tolerance, expected_ts",
    [
        pytest.param(5, [0, 5, 10], id="tolerance_boundary_exact_kept"),
        pytest.param(4, [0, 10], id="tolerance_just_over_dropped"),
        pytest.param(100, [0, 5, 10], id="large_tolerance_all_kept"),
    ],
)
def test_inner_join_tolerance(tolerance, expected_ts):
    left_ts = np.array([0, 5, 10], dtype=np.int64)
    right_ts = np.array([0, 10], dtype=np.int64)
    ts, _, _ = inner_join(
        left_ts, np.ones(3), right_ts, np.ones(2), tolerance=tolerance
    )
    np.testing.assert_array_equal(ts, expected_ts)


def test_inner_join_no_matches_returns_empty():
    left_ts = np.array([100, 200, 300], dtype=np.int64)
    right_ts = np.array([0, 1], dtype=np.int64)
    ts, lv, rv = inner_join(left_ts, np.ones(3), right_ts, np.ones(2), tolerance=1)
    assert len(ts) == 0
    assert len(lv) == 0
    assert len(rv) == 0


def test_inner_join_preserves_left_values():
    left_ts = np.array([0, 2, 4], dtype=np.int64)
    left_val = np.array([11.0, 22.0, 33.0])
    right_ts = np.array([0, 4], dtype=np.int64)
    _, lv, _ = inner_join(left_ts, left_val, right_ts, np.ones(2), tolerance=0)
    np.testing.assert_allclose(lv, [11.0, 33.0])


def test_inner_join_right_values_interpolated_not_snapped():
    left_ts = np.array([0, 5, 10], dtype=np.int64)
    right_ts = np.array([0, 10], dtype=np.int64)
    _, _, rv = inner_join(
        left_ts, np.ones(3), right_ts, np.array([0.0, 10.0]), tolerance=5
    )
    np.testing.assert_allclose(rv[1], 5.0)


def test_resample_to_freq_produces_correct_spacing():
    ts_src = np.array([0, 1_000_000], dtype=np.int64)
    val_src = np.array([0.0, 1.0])
    ts, val = resample_to_freq(ts_src, val_src, freq_hz=2.0, timestamp_divisor=1e6)
    assert len(ts) == 2
    assert ts[0] == 0
    assert ts[1] == 500_000


def test_resample_to_freq_linear_values():
    ts_src = np.array([0, 1_000_000], dtype=np.int64)
    val_src = np.array([0.0, 10.0])
    ts, val = resample_to_freq(ts_src, val_src, freq_hz=2.0, timestamp_divisor=1e6)
    np.testing.assert_allclose(val, [0.0, 5.0])


def test_resample_to_freq_starts_at_first_timestamp():
    ts_src = np.array([500, 1_000_500], dtype=np.int64)
    val_src = np.array([0.0, 1.0])
    ts, _ = resample_to_freq(ts_src, val_src, freq_hz=1.0, timestamp_divisor=1e3)
    assert ts[0] == 500


def test_resample_to_freq_does_not_include_last_timestamp():
    ts_src = np.array([0, 1_000_000], dtype=np.int64)
    val_src = np.array([0.0, 1.0])
    ts, _ = resample_to_freq(ts_src, val_src, freq_hz=1.0, timestamp_divisor=1e6)
    assert ts[-1] < ts_src[-1]


def test_resample_to_freq_microseconds_100hz():
    ts_src = np.array([0, 100_000], dtype=np.int64)
    val_src = np.array([0.0, 10.0])
    ts, _ = resample_to_freq(ts_src, val_src, freq_hz=100.0, timestamp_divisor=1e6)
    diffs = np.diff(ts)
    np.testing.assert_allclose(diffs, 10_000)


def test_left_join_data_instances_single_right_returns_two(di_simple, di_sparse):
    result = left_join_data_instances(di_simple, di_sparse)
    assert len(result) == 2


def test_left_join_data_instances_list_right_returns_n_plus_one(
    di_simple, di_sparse, di_two_points
):
    result = left_join_data_instances(di_simple, [di_sparse, di_two_points])
    assert len(result) == 3


def test_left_join_data_instances_all_share_timestamps(
    di_simple, di_sparse, di_two_points
):
    la, rb, rc = left_join_data_instances(di_simple, [di_sparse, di_two_points])
    np.testing.assert_array_equal(la.timestamp_np, rb.timestamp_np)
    np.testing.assert_array_equal(la.timestamp_np, rc.timestamp_np)


def test_left_join_data_instances_method_forwarded(di_simple, di_two_points):
    _, rb_zoh = left_join_data_instances(
        di_simple, di_two_points, method=ResampleMethod.ZOH
    )
    _, rb_lin = left_join_data_instances(
        di_simple, di_two_points, method=ResampleMethod.LINEAR
    )
    assert not np.allclose(rb_zoh.value_np, rb_lin.value_np)


@pytest.mark.parametrize(
    "join_fn, kwargs",
    [
        pytest.param(left_join_data_instances, {}, id="left"),
        pytest.param(outer_join_data_instances, {}, id="outer"),
        pytest.param(inner_join_data_instances, {"tolerance": 100}, id="inner"),
    ],
)
def test_join_data_instances_labels_preserved(di_simple, di_sparse, join_fn, kwargs):
    left, right = join_fn(di_simple, di_sparse, **kwargs)
    assert left.label == di_simple.label
    assert right.label == di_sparse.label


@pytest.mark.parametrize(
    "join_fn, kwargs",
    [
        pytest.param(left_join_data_instances, {}, id="left"),
        pytest.param(outer_join_data_instances, {}, id="outer"),
        pytest.param(inner_join_data_instances, {"tolerance": 100}, id="inner"),
    ],
)
def test_join_data_instances_shared_timestamps(di_simple, di_sparse, join_fn, kwargs):
    left, right = join_fn(di_simple, di_sparse, **kwargs)
    np.testing.assert_array_equal(left.timestamp_np, right.timestamp_np)


def test_inner_join_data_instances_zero_tolerance_exact_only(di_simple, di_sparse):
    left, right = inner_join_data_instances(di_simple, di_sparse, tolerance=0)
    np.testing.assert_array_equal(left.timestamp_np, [0])


def test_outer_join_data_instances_union_timestamps(di_simple, di_sparse):
    left, right = outer_join_data_instances(di_simple, di_sparse)
    for ts in np.intersect1d(di_simple.timestamp_np, di_sparse.timestamp_np):
        assert ts in left.timestamp_np


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_binary_op_uses_left_timestamps(di_simple, di_sparse, op):
    result = op(di_simple, di_sparse)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)
