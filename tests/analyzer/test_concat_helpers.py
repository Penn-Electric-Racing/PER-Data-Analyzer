import numpy as np
import pytest

from perda.analyzer.concat_helpers import _upscale_to_us, concat_single_run_data
from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.semantic_embedding import build_semantic_index
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale
from perda.utils.search import search


def _make_srd(names_vals, ts_unit=Timescale.MS, start=0, end=None, descripts={}):
    """Build a minimal SingleRunData from {cpp_name: (ts_list, val_list)} dict."""
    instances = {}
    cpp_name_to_id = {}
    id_to_cpp_name = {}
    id_to_descript = {}
    all_ts = []

    for var_id, (name, (ts_list, val_list)) in enumerate(names_vals.items(), start=1):
        ts = np.array(ts_list, dtype=np.int64)
        vals = np.array(val_list, dtype=np.float64)
        all_ts.extend(ts_list)
        instances[var_id] = DataInstance(
            timestamp_np=ts, value_np=vals, label=name, var_id=var_id, cpp_name=name
        )
        cpp_name_to_id[name] = var_id
        id_to_cpp_name[var_id] = name
        id_to_descript[var_id] = descripts.get(name, "")

    data_end = end if end is not None else (max(all_ts) if all_ts else 0)
    return SingleRunData(
        id_to_instance=instances,
        cpp_name_to_id=cpp_name_to_id,
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        total_data_points=sum(len(v) for v in instances.values()),
        data_start_time=start,
        data_end_time=data_end,
        timestamp_unit=ts_unit,
    )


def test_upscale_to_us_multiplies_timestamps_by_1000():
    srd = _make_srd({"a.b": ([0, 1, 2], [1.0, 2.0, 3.0])}, ts_unit=Timescale.MS)
    result = _upscale_to_us(srd)
    np.testing.assert_array_equal(result["a.b"].timestamp_np, [0, 1000, 2000])


def test_upscale_to_us_sets_unit():
    srd = _make_srd({"x": ([0, 10], [0.0, 1.0])}, ts_unit=Timescale.MS)
    result = _upscale_to_us(srd)
    assert result.timestamp_unit == Timescale.US


def test_upscale_to_us_scales_start_end_time():
    srd = _make_srd({"x": ([5, 10], [0.0, 1.0])}, ts_unit=Timescale.MS, start=5, end=10)
    result = _upscale_to_us(srd)
    assert result.data_start_time == 5000
    assert result.data_end_time == 10000


def test_upscale_to_us_preserves_values():
    srd = _make_srd({"v": ([0, 1], [42.0, 43.0])}, ts_unit=Timescale.MS)
    result = _upscale_to_us(srd)
    np.testing.assert_allclose(result["v"].value_np, [42.0, 43.0])


def test_concat_second_timestamps_shifted_after_first():
    first = _make_srd({"a": ([0, 1, 2], [1.0, 2.0, 3.0])})
    second = _make_srd({"a": ([0, 1, 2], [4.0, 5.0, 6.0])})
    result = concat_single_run_data(first, second, gap=1)
    ts = result["a"].timestamp_np
    # First run ends at 2; second should start at 3+ (gap=1)
    assert ts[3] == 3


def test_concat_result_contains_all_values():
    first = _make_srd({"a": ([0, 1], [1.0, 2.0])})
    second = _make_srd({"a": ([0, 1], [3.0, 4.0])})
    result = concat_single_run_data(first, second)
    assert len(result["a"]) == 4


def test_concat_data_start_time_from_first():
    first = _make_srd({"a": ([10, 20], [1.0, 2.0])}, start=10, end=20)
    second = _make_srd({"a": ([0, 10], [3.0, 4.0])})
    result = concat_single_run_data(first, second)
    assert result.data_start_time == first.data_start_time


def test_concat_variables_union_of_both_runs():
    first = _make_srd({"a": ([0, 1], [1.0, 2.0])})
    second = _make_srd({"b": ([0, 1], [3.0, 4.0])})
    result = concat_single_run_data(first, second)
    assert "a" in result
    assert "b" in result


def test_concat_variable_only_in_first_kept():
    first = _make_srd({"only_first": ([0, 1], [9.0, 9.0]), "shared": ([0, 1], [1.0, 2.0])})
    second = _make_srd({"shared": ([0, 1], [3.0, 4.0])})
    result = concat_single_run_data(first, second)
    assert "only_first" in result
    assert len(result["only_first"]) == 2


def test_concat_variable_only_in_second_kept():
    first = _make_srd({"shared": ([0, 1], [1.0, 2.0])})
    second = _make_srd({"shared": ([0, 1], [3.0, 4.0]), "only_second": ([0, 1], [7.0, 8.0])})
    result = concat_single_run_data(first, second)
    assert "only_second" in result


def test_concat_adds_boundary_timestamp():
    first = _make_srd({"a": ([0, 5], [1.0, 2.0])}, end=5)
    second = _make_srd({"a": ([0, 5], [3.0, 4.0])})
    result = concat_single_run_data(first, second, gap=1)
    assert len(result.concat_boundaries) == 1


def test_concat_carries_existing_boundaries():
    first = _make_srd({"a": ([0, 5], [1.0, 2.0])}, end=5)
    first = first.model_copy(update={"concat_boundaries": [3]})
    second = _make_srd({"a": ([0, 5], [3.0, 4.0])})
    result = concat_single_run_data(first, second, gap=1)
    assert 3 in result.concat_boundaries


def test_concat_ms_and_us_result_is_us():
    first = _make_srd({"x": ([0, 1], [1.0, 2.0])}, ts_unit=Timescale.MS)
    second = _make_srd({"x": ([0, 1000], [3.0, 4.0])}, ts_unit=Timescale.US)
    result = concat_single_run_data(first, second)
    assert result.timestamp_unit == Timescale.US


def test_concat_us_and_ms_result_is_us():
    first = _make_srd({"x": ([0, 1000], [1.0, 2.0])}, ts_unit=Timescale.US)
    second = _make_srd({"x": ([0, 1], [3.0, 4.0])}, ts_unit=Timescale.MS)
    result = concat_single_run_data(first, second)
    assert result.timestamp_unit == Timescale.US


def test_concat_same_unit_no_upscale():
    first = _make_srd({"x": ([0, 1], [1.0, 2.0])}, ts_unit=Timescale.MS)
    second = _make_srd({"x": ([0, 1], [3.0, 4.0])}, ts_unit=Timescale.MS)
    result = concat_single_run_data(first, second)
    assert result.timestamp_unit == Timescale.MS


@pytest.mark.parametrize(
    "first_indexed, second_indexed, expected_embedded",
    [
        pytest.param(True, True, [], id="both_indexed"),
        pytest.param(True, False, [["rear speed"]], id="first_indexed"),
        pytest.param(False, True, [["front speed", "shared signal"]], id="second_indexed"),
    ],
)
def test_concat_merges_semantic_indexes(
    fake_encoder, first_indexed, second_indexed, expected_embedded
):
    first = _make_srd(
        {"a": ([0], [1.0]), "shared": ([0], [1.0])},
        descripts={"a": "front speed", "shared": "shared signal"},
    )
    second = _make_srd(
        {"shared": ([0], [1.0]), "b": ([0], [1.0])},
        descripts={"shared": "shared signal", "b": "rear speed"},
    )
    if first_indexed:
        first.semantic_index = build_semantic_index(first.id_to_descript, verbose=0)
    if second_indexed:
        second.semantic_index = build_semantic_index(second.id_to_descript, verbose=0)
    batches_before = len(fake_encoder)

    result = concat_single_run_data(first, second)

    assert fake_encoder[batches_before:] == expected_embedded
    assert result.semantic_index.id_to_descript == result.id_to_descript
    assert search(result, "rear speed", top_n=1)[0].cpp_name == "b"


def test_concat_without_semantic_indexes_has_none():
    first = _make_srd({"a": ([0], [1.0])})
    second = _make_srd({"b": ([0], [1.0])})
    assert concat_single_run_data(first, second).semantic_index is None
