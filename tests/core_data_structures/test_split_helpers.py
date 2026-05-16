import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.core_data_structures.split_helpers import (
    split_single_run_data,
    trim_single_run_data,
)
from perda.units import Timescale

# ---------------------------------------------------------------------------
# trim_single_run_data
# ---------------------------------------------------------------------------


def test_trim_single_run_data_clips_all_variables(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=1, ts_end=2)
    for di in result.id_to_instance.values():
        assert di.timestamp_np.min() >= 1
        assert di.timestamp_np.max() <= 2


def test_trim_single_run_data_preserves_variable_count(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=0, ts_end=2)
    assert len(result.id_to_instance) == len(srd_basic.id_to_instance)


def test_trim_single_run_data_updates_metadata(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=1, ts_end=2)
    assert result.data_start_time == 1
    assert result.data_end_time == 2


def test_trim_single_run_data_preserves_mappings(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=0, ts_end=2)
    assert result.cpp_name_to_id == srd_basic.cpp_name_to_id
    assert result.id_to_cpp_name == srd_basic.id_to_cpp_name


def test_trim_single_run_data_does_not_mutate_original(srd_basic):
    original_len = len(srd_basic["cpp.var_a"])
    trim_single_run_data(srd_basic, ts_start=1, ts_end=2)
    assert len(srd_basic["cpp.var_a"]) == original_len


def test_trim_single_run_data_total_data_points_updated(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=1, ts_end=1)
    expected = sum(len(di.value_np) for di in result.id_to_instance.values())
    assert result.total_data_points == expected


def test_trim_single_run_data_resets_concat_boundaries(srd_basic):
    result = trim_single_run_data(srd_basic, ts_start=0, ts_end=2)
    assert result.concat_boundaries == []


# ---------------------------------------------------------------------------
# split_single_run_data
# ---------------------------------------------------------------------------


def test_split_single_run_data_returns_correct_segment_count(srd_basic):
    segments = split_single_run_data(srd_basic, [0, 1, 2])
    assert len(segments) == 2


def test_split_single_run_data_two_boundaries_one_segment(srd_basic):
    segments = split_single_run_data(srd_basic, [0, 2])
    assert len(segments) == 1


def test_split_single_run_data_segments_have_disjoint_ranges(srd_basic):
    segments = split_single_run_data(srd_basic, [0, 1, 2])
    assert segments[0].data_end_time <= segments[1].data_start_time


def test_split_single_run_data_raises_on_single_boundary(srd_basic):
    with pytest.raises(ValueError):
        split_single_run_data(srd_basic, [0])


def test_split_single_run_data_raises_on_empty_boundaries(srd_basic):
    with pytest.raises(ValueError):
        split_single_run_data(srd_basic, [])


@pytest.mark.parametrize(
    "boundaries",
    [
        pytest.param([0, 1, 2, 3], id="four_boundaries"),
        pytest.param([0, 0.5, 1, 1.5, 2], id="five_boundaries"),
    ],
)
def test_split_single_run_data_segment_count_matches_boundary_pairs(
    srd_basic, boundaries
):
    segments = split_single_run_data(srd_basic, boundaries)
    assert len(segments) == len(boundaries) - 1
