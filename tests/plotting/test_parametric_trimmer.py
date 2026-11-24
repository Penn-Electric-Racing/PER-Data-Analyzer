import numpy as np
import pytest

pytest.importorskip("ipywidgets")

from perda.plotting.parametric_trimmer import plot_parametric_trimmer
from perda.units import Timescale


@pytest.mark.parametrize(
    "n_x, n_y, n_timestamps",
    [
        pytest.param(5, 4, None, id="x_y_mismatch"),
        pytest.param(5, 5, 3, id="timestamps_mismatch"),
        pytest.param(0, 0, None, id="empty"),
    ],
)
def test_rejects_inconsistent_lengths(n_x, n_y, n_timestamps):
    timestamps = (
        None if n_timestamps is None else np.arange(n_timestamps, dtype=np.float64)
    )
    with pytest.raises(ValueError):
        plot_parametric_trimmer(
            np.zeros(n_x),
            np.zeros(n_y),
            Timescale.MS,
            timestamps=timestamps,
        )


def test_accepts_matching_lengths():
    widget = plot_parametric_trimmer(
        np.arange(5.0), np.arange(5.0), Timescale.MS, timestamps=np.arange(5.0)
    )
    assert len(widget.children[0].data[0].x) == 5
