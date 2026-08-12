import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.plotting.data_instance_plotter import plot_single_axis


@pytest.fixture
def long_data_instance():
    """DataInstance with 105 points, so downsampling has something to cut."""
    return DataInstance(
        timestamp_np=np.arange(105, dtype=np.int64) * 100,
        value_np=np.arange(105, dtype=np.float64) * 0.1,
        label="voltage",
    )


@pytest.mark.parametrize(
    "max_points, expected_points",
    [
        # Ceiling division step: (105 + 20 - 1) // 20 = 6, so ceil(105 / 6) = 18
        pytest.param(20, 18, id="downsampled"),
        pytest.param(200, 105, id="limit_above_length"),
        pytest.param(None, 105, id="no_limit"),
    ],
)
def test_downsampling_point_count(long_data_instance, max_points, expected_points):
    fig = plot_single_axis([long_data_instance], max_points=max_points)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == expected_points
