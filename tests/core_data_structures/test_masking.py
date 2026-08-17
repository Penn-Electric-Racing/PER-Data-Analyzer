import numpy as np

from perda.core_data_structures.masking import FilterOptions, apply_ufunc_filter


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
    assert result.cpp_name == di_with_metadata.cpp_name


def test_filter_values_default_apply_to(di_simple):
    result = apply_ufunc_filter(di_simple, lambda v: v == 0.0)
    assert len(result) == 1
    assert result.value_np[0] == 0.0
