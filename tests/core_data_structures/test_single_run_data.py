import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance


@pytest.mark.parametrize(
    "key, expected_values",
    [
        pytest.param(1, [1.0, 2.0, 3.0], id="int_id_first"),
        pytest.param(2, [4.0, 5.0, 6.0], id="int_id_second"),
        pytest.param("cpp.var_a", [1.0, 2.0, 3.0], id="string_name_first"),
        pytest.param("cpp.var_b", [4.0, 5.0, 6.0], id="string_name_second"),
    ],
)
def test_getitem_valid(srd_basic, key, expected_values):
    result = srd_basic[key]
    assert isinstance(result, DataInstance)
    np.testing.assert_array_equal(result.value_np, expected_values)


def test_getitem_by_data_instance_passthrough(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([99.0, 100.0]),
    )
    result = srd_basic[di]
    assert result is di


@pytest.mark.parametrize(
    "bad_key, exc_type",
    [
        pytest.param(999, KeyError, id="unknown_int"),
        pytest.param("no_such_var", KeyError, id="unknown_string"),
        pytest.param(3.14, (ValueError, TypeError), id="wrong_type"),
    ],
)
def test_getitem_invalid_raises(srd_basic, bad_key, exc_type):
    with pytest.raises(exc_type):
        srd_basic[bad_key]


@pytest.mark.parametrize(
    "key, expected_names",
    [
        pytest.param("cpp.*", ["cpp.var_a", "cpp.var_b"], id="trailing_wildcard"),
        pytest.param("*_b", ["cpp.var_b"], id="leading_wildcard_single_match"),
        pytest.param("cpp.var_*", ["cpp.var_a", "cpp.var_b"], id="partial_segment"),
        pytest.param(
            ("cpp.var_b", 1), ["cpp.var_b", "cpp.var_a"], id="tuple_keeps_order"
        ),
        pytest.param(
            ("*_b", "cpp.var_a"), ["cpp.var_b", "cpp.var_a"], id="tuple_with_pattern"
        ),
    ],
)
def test_getitem_multiple_returns_tuple(srd_basic, key, expected_names):
    result = srd_basic[key]
    assert isinstance(result, tuple)
    assert [di.cpp_name for di in result] == expected_names


@pytest.mark.parametrize("key", ["ghost.*", ("cpp.var_a", "ghost")])
def test_getitem_multiple_missing_raises(srd_basic, key):
    with pytest.raises(KeyError):
        srd_basic[key]


def test_pattern_does_not_treat_dot_as_regex(srd_basic):
    with pytest.raises(KeyError):
        srd_basic["cpp?var_*"]
    assert "cppXvar_*" not in srd_basic


@pytest.mark.parametrize("key", [1, 2, "cpp.var_a", "cpp.var_b", "cpp.*"])
def test_contains_present(srd_basic, key):
    assert key in srd_basic


@pytest.mark.parametrize("key", [999, "ghost", "", 42, "ghost.*"])
def test_contains_absent(srd_basic, key):
    assert key not in srd_basic


def test_contains_does_not_raise_on_missing(srd_basic):
    result = 42 in srd_basic
    assert result is False


def test_getitem_int_and_string_return_same_instance(srd_basic):
    by_id = srd_basic[1]
    by_name = srd_basic["cpp.var_a"]
    assert by_id is by_name


def test_cpp_name_to_id_consistent(srd_basic):
    for name, var_id in srd_basic.cpp_name_to_id.items():
        assert srd_basic.id_to_cpp_name[var_id] == name


def test_id_to_cpp_name_consistent(srd_basic):
    for var_id, name in srd_basic.id_to_cpp_name.items():
        assert srd_basic.cpp_name_to_id[name] == var_id


def test_all_ids_have_instances(srd_basic):
    for var_id in srd_basic.id_to_cpp_name:
        assert var_id in srd_basic.id_to_instance


def test_trim_clips_every_variable_and_updates_metadata(srd_basic):
    result = srd_basic.trim(ts_start=1, ts_end=2)
    for di in result.id_to_instance.values():
        np.testing.assert_array_equal(di.timestamp_np, [1, 2])
    assert (result.data_start_time, result.data_end_time) == (1, 2)
    assert result.total_data_points == 4
    assert result.cpp_name_to_id == srd_basic.cpp_name_to_id
    assert len(srd_basic["cpp.var_a"]) == 3


@pytest.mark.parametrize(
    "split_timestamps, expected_timestamps",
    [
        pytest.param([], [[0, 1, 2]], id="no_split"),
        pytest.param([1], [[0], [1, 2]], id="point_on_split_goes_later"),
        pytest.param([0.5, 1.5], [[0], [1], [2]], id="between_samples"),
        pytest.param([1, 1], [[0], [], [1, 2]], id="repeated_split"),
    ],
)
def test_split_gives_one_more_run_than_split_points(
    srd_basic, split_timestamps, expected_timestamps
):
    runs = srd_basic.split(split_timestamps)
    assert [
        run["cpp.var_a"].timestamp_np.tolist() for run in runs
    ] == expected_timestamps


def test_split_rejects_unsorted_points(srd_basic):
    with pytest.raises(ValueError):
        srd_basic.split([2, 1])
