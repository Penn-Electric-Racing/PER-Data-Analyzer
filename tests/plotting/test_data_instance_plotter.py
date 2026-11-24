import pytest

from perda.plotting.data_instance_plotter import plot_time_series
from perda.units import Timescale


def test_subplots_share_x_and_place_right_axis_traces(srd_basic, di_simple):
    var_a, var_b = srd_basic["cpp.*"]
    fig = plot_time_series(
        rows=[[var_a, var_b], [di_simple]],
        right_rows=[[], [var_a]],
        y_axis_titles=["Left 1", "Left 2"],
        right_y_axis_titles=[None, "Right 2"],
        timestamp_unit=Timescale.MS,
    )
    assert [(trace.xaxis, trace.yaxis) for trace in fig.data] == [
        ("x", "y"),
        ("x", "y"),
        ("x2", "y2"),
        ("x2", "y3"),
    ]
    assert fig.layout.xaxis.matches == "x2"
    assert fig.layout.yaxis3.side == "right"
    assert fig.layout.yaxis.title.text == "Left 1"
    assert fig.layout.yaxis3.title.text == "Right 2"


def test_vlines_drawn_once_per_subplot(di_simple):
    fig = plot_time_series(
        rows=[[di_simple], [di_simple]],
        right_rows=[[], [di_simple]],
        vlines=[0.5],
        timestamp_unit=Timescale.MS,
    )
    assert len(fig.layout.shapes) == 2


@pytest.mark.parametrize(
    "log_x, log_y, expected_x, expected_y",
    [
        pytest.param(False, False, "linear", "linear", id="linear"),
        pytest.param(True, True, "log", "log", id="log_both"),
    ],
)
def test_log_scaling_applies_to_every_axis(
    di_simple, log_x, log_y, expected_x, expected_y
):
    fig = plot_time_series(
        rows=[[di_simple], [di_simple]],
        right_rows=[[di_simple], []],
        log_x=log_x,
        log_y=log_y,
        timestamp_unit=Timescale.MS,
    )
    layout = fig.layout.to_plotly_json()
    x_types = {v["type"] for k, v in layout.items() if k.startswith("xaxis")}
    y_types = {v["type"] for k, v in layout.items() if k.startswith("yaxis")}
    assert x_types == {expected_x}
    assert y_types == {expected_y}


@pytest.mark.parametrize(
    "build_rows, build_right_rows",
    [
        pytest.param(lambda di: [[]], lambda di: None, id="no_data"),
        pytest.param(
            lambda di: [[di.trim(100, 200)]], lambda di: None, id="empty_data_instance"
        ),
        pytest.param(
            lambda di: [[di]], lambda di: [[], []], id="right_rows_length_mismatch"
        ),
    ],
)
def test_plot_time_series_invalid_raises(di_simple, build_rows, build_right_rows):
    with pytest.raises(ValueError):
        plot_time_series(
            rows=build_rows(di_simple),
            right_rows=build_right_rows(di_simple),
            timestamp_unit=Timescale.MS,
        )
