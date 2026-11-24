import numpy as np
import pytest

from perda.plotting.binned_count_plotter import count_per_bin, plot_binned_counts


@pytest.mark.parametrize(
    "values, n_bins, expected_counts",
    [
        # Bins of width 1.0 from 0.0: [0,1) gets 0.1 and 0.9, [1,2) gets 1.5
        pytest.param([0.1, 0.9, 1.5], 2, [2, 1], id="two_bins"),
        # 5.0 lies past the last bin edge, so it is clamped into the last bin
        pytest.param([0.5, 5.0], 2, [1, 1], id="clamped_above"),
        pytest.param([], 3, [0, 0, 0], id="empty"),
    ],
)
def test_count_per_bin(values, n_bins, expected_counts):
    counts = count_per_bin(
        np.array(values, dtype=np.float64), bin_start=0.0, bin_width=1.0, n_bins=n_bins
    )
    np.testing.assert_array_equal(counts, expected_counts)


def test_series_share_one_bin_grid():
    fig = plot_binned_counts(
        {
            "all": np.array([0.0, 0.5, 1.2, 2.9]),
            "subset": np.array([1.2]),
        },
        bin_width=1.0,
    )
    # Grid spans 0.0 to 2.9 across all series: int(2.9 / 1.0) + 1 = 3 bins
    assert [trace.name for trace in fig.data] == ["all", "subset"]
    np.testing.assert_array_equal(fig.data[0].x, fig.data[1].x)
    np.testing.assert_array_equal(fig.data[0].y, [2, 1, 1])
    np.testing.assert_array_equal(fig.data[1].y, [0, 1, 0])


def test_all_empty_series_give_empty_figure():
    fig = plot_binned_counts({"a": np.array([]), "b": np.array([])}, bin_width=1.0)
    assert len(fig.data) == 0
