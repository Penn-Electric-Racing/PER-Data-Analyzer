import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.deduplication import deduplicate


def test_deduplicate_removes_consecutive_duplicates():
    di_with_duplicates = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([1.0, 1.0, 2.0, 2.0, 3.0]),
        label="speed",
        var_id=42,
        cpp_name="vehicle.speed",
    )

    result = deduplicate(di_with_duplicates)
    np.testing.assert_array_equal(result.value_np, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.timestamp_np, [0, 2, 4])


def test_deduplicate_no_duplicates_unchanged(di_simple):
    result = deduplicate(di_simple)
    np.testing.assert_array_equal(result.value_np, di_simple.value_np)
    np.testing.assert_array_equal(result.timestamp_np, di_simple.timestamp_np)


@pytest.mark.parametrize(
    "timestamps, values, expected_ts",
    [
        pytest.param(
            np.array([0, 1, 2, 3, 4], dtype=np.int64),
            np.array([7.0, 7.0, 7.0, 7.0, 7.0]),
            0,
            id="five_same_values",
        ),
        pytest.param(
            np.array([10, 20, 30, 40, 50], dtype=np.int64),
            np.array([3.14, 3.14, 3.14, 3.14, 3.14]),
            10,
            id="five_same_non_zero_start",
        ),
    ],
)
def test_deduplicate_all_same_keeps_only_first(timestamps, values, expected_ts):
    d = DataInstance(timestamp_np=timestamps, value_np=values)
    result = deduplicate(d)
    assert len(result) == 1
    assert result.timestamp_np[0] == expected_ts
    np.testing.assert_allclose(result.value_np[0], values[0])


def test_deduplicate_non_consecutive_duplicates_all_kept():
    # Pattern [1, 2, 1]: the third value is not consecutive with the first, so all kept
    d = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 1.0]),
    )
    result = deduplicate(d)
    assert len(result) == 3


def test_deduplicate_preserves_first_timestamp_of_each_run():
    # Values [5, 5, 5, 9, 9] -> keeps timestamps at index 0 and 3
    d = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([5.0, 5.0, 5.0, 9.0, 9.0]),
    )
    result = deduplicate(d)
    np.testing.assert_array_equal(result.timestamp_np, [0, 3])
    np.testing.assert_array_equal(result.value_np, [5.0, 9.0])


def test_deduplicate_nan_consecutive_both_kept():
    # NaN != NaN, so two consecutive NaNs are treated as distinct and both kept
    d = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([float("nan"), float("nan"), 1.0]),
    )
    result = deduplicate(d)
    assert len(result) == 3


def test_deduplicate_alternating_values_unchanged():
    d = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 1.0, 2.0, 1.0]),
    )
    result = deduplicate(d)
    assert len(result) == 5


def test_deduplicate_returns_new_instance(di_simple):
    result = deduplicate(di_simple)
    assert result is not di_simple


@pytest.mark.parametrize(
    "values, expected_len, expected_first_ts",
    [
        pytest.param(np.array([1.0, 2.0]), 2, 0, id="no_dup"),
        pytest.param(np.array([5.0, 5.0]), 1, 0, id="with_dup"),
    ],
)
def test_deduplicate_two_point(values, expected_len, expected_first_ts):
    d = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=values,
    )
    result = deduplicate(d)
    assert len(result) == expected_len
    assert result.timestamp_np[0] == expected_first_ts
