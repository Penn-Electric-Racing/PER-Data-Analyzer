from datetime import datetime

import pytest

from perda.analyzer.run_collection_factory import from_paths

VOLTAGE = "ams.pack.voltage"


def test_runs_sorted_chronologically(two_run_collection):
    assert two_run_collection.filenames == ["log_old.csv", "log_new.csv"]
    assert two_run_collection[0].data.creation_time == datetime(2026, 6, 11, 10, 0, 0)
    assert two_run_collection[1].data.creation_time == datetime(2026, 6, 11, 12, 0, 0)


def test_len(two_run_collection):
    assert len(two_run_collection) == 2


def test_filenames_does_not_load_runs(two_run_collection):
    assert two_run_collection.filenames == ["log_old.csv", "log_new.csv"]
    assert two_run_collection._loaded == {}


def test_str_lists_runs_without_loading(two_run_collection):
    listing = str(two_run_collection)
    assert "log_old.csv" in listing
    assert "log_new.csv" in listing
    assert two_run_collection._loaded == {}


def test_get_run_by_chronological_index_caches(two_run_collection):
    first = two_run_collection.get_run_by_chronological_index(0)
    assert two_run_collection.get_run_by_chronological_index(0) is first


@pytest.mark.parametrize("index", [5, -5])
def test_out_of_range_index_raises(two_run_collection, index):
    with pytest.raises(IndexError):
        two_run_collection.get_run_by_chronological_index(index)


@pytest.mark.parametrize(
    "start_date, end_date, expected",
    [
        pytest.param(
            datetime(2026, 6, 11, 9),
            datetime(2026, 6, 11, 11),
            ["log_old.csv"],
            id="range",
        ),
        pytest.param("2026-06-11T11:00:00", None, ["log_new.csv"], id="iso_start_only"),
        pytest.param(None, "2026-06-11T11:00:00", ["log_old.csv"], id="iso_end_only"),
        pytest.param(None, None, ["log_old.csv", "log_new.csv"], id="unbounded"),
        pytest.param(datetime(2027, 1, 1), None, [], id="no_matches"),
    ],
)
def test_filter_by_date(two_run_collection, start_date, end_date, expected):
    assert two_run_collection.filter_by_date(start_date, end_date).filenames == expected


def test_filter_on_metadata(two_run_collection):
    filtered = two_run_collection.filter(lambda run: "old" in run.file_path.name)
    assert filtered.filenames == ["log_old.csv"]


def test_filter_by_data(two_run_collection):
    overvolted = two_run_collection.filter_by_data(
        lambda aly: (aly.data[VOLTAGE].value_np > 10.0).any()
    )
    assert overvolted.filenames == ["log_new.csv"]


def test_filters_chain(two_run_collection):
    chained = two_run_collection.filter_by_date("2026-06-11T09:00:00").filter_by_data(
        lambda aly: len(aly.data[VOLTAGE]) == 2
    )
    assert chained.filenames == ["log_old.csv"]


def test_compare_summary(two_run_collection):
    summaries = two_run_collection.compare_summary(VOLTAGE)
    assert set(summaries) == {"log_old.csv", "log_new.csv"}
    assert summaries["log_old.csv"].max_value == 6.0
    assert summaries["log_new.csv"].count == 1


def test_compare_summary_skips_runs_missing_variable(two_run_collection):
    assert two_run_collection.compare_summary("not.a.variable") == {}


def test_plot_comparison_overlays_every_run(two_run_collection):
    fig = two_run_collection.plot_comparison(VOLTAGE)
    assert len(fig.data) == 2
    assert all(trace.x[0] == 0 for trace in fig.data)
    assert {trace.name for trace in fig.data} == {
        "log_old.csv (voltage)",
        "log_new.csv (voltage)",
    }


def test_plot_comparison_aligns_mixed_timestamp_units(write_log):
    ms_log = write_log("ms_log.csv", "PER Log: Thu Jun 11 10:00:00 2026", (0.0, 1.0))
    us_log = write_log(
        "us_log.csv", "PER Log: Thu Jun 11 12:00:00 2026 v2.0", (0.0, 1.0)
    )
    us_log.write_text(
        "PER Log: Thu Jun 11 12:00:00 2026 v2.0\n"
        "Value voltage (ams.pack.voltage): 1\n"
        "500000,1,0.0\n1500000,1,1.0\n"
    )

    fig = from_paths([ms_log, us_log]).plot_comparison(VOLTAGE)
    # A ms log and a us log, each spanning 1 second, must land on the same x-axis
    assert [list(trace.x) for trace in fig.data] == [[0.0, 1.0], [0.0, 1.0]]
