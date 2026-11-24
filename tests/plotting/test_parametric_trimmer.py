import numpy as np
import pytest

pytest.importorskip("ipywidgets")

from perda.plotting.parametric_trimmer import plot_parametric_trimmer
from perda.units import Timescale


@pytest.mark.parametrize(
    "n_x, n_y, n_timestamps",
    [
        pytest.param(5, 4, 5, id="x_y_mismatch"),
        pytest.param(5, 5, 3, id="timestamps_mismatch"),
        pytest.param(0, 0, 0, id="empty"),
    ],
)
def test_rejects_inconsistent_lengths(n_x, n_y, n_timestamps):
    with pytest.raises(ValueError):
        plot_parametric_trimmer(
            np.zeros(n_x),
            np.zeros(n_y),
            np.arange(n_timestamps, dtype=np.int64),
            Timescale.MS,
        )


def test_colors_points_by_elapsed_seconds():
    widget = plot_parametric_trimmer(
        np.arange(3.0),
        np.arange(3.0),
        np.array([1000, 1500, 3000], dtype=np.int64),
        Timescale.MS,
    )
    np.testing.assert_allclose(widget.children[0].data[0].marker.color, [0.0, 0.5, 2.0])
