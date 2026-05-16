import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance


@pytest.fixture
def di_sparse():
    """Irregular, wide-gap DataInstance — primary 'right' side for join tests."""
    return DataInstance(
        timestamp_np=np.array([0, 10, 100], dtype=np.int64),
        value_np=np.array([5.0, 15.0, 25.0]),
        label="sparse",
        var_id=2,
        cpp_name="test.sparse",
    )


@pytest.fixture
def di_with_metadata():
    """All five fields explicitly set; use when testing metadata round-trips."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3], dtype=np.int64),
        value_np=np.array([10.0, 20.0, 30.0, 40.0]),
        label="meta",
        var_id=99,
        cpp_name="test.meta",
    )


@pytest.fixture
def di_two_points():
    """Two-point DataInstance at ts=[0, 10] for join tests."""
    return DataInstance(
        timestamp_np=np.array([0, 10], dtype=np.int64),
        value_np=np.array([0.0, 10.0]),
        label="two_pts",
        var_id=6,
        cpp_name="test.two_pts",
    )
