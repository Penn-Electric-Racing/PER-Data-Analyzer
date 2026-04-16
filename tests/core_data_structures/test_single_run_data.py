import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData

# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [1, 2, "cpp.var_a", "cpp.var_b"])
def test_contains_present(srd_basic, key):
    assert key in srd_basic


@pytest.mark.parametrize("key", [999, "ghost", "", 42])
def test_contains_absent(srd_basic, key):
    assert key not in srd_basic


def test_contains_does_not_raise_on_missing(srd_basic):
    result = 42 in srd_basic
    assert result is False


# ---------------------------------------------------------------------------
# Consistency between lookups
# ---------------------------------------------------------------------------


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
