import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale

# ---------------------------------------------------------------------------
# __setitem__ / add / replace dispatch
# ---------------------------------------------------------------------------


def test_setitem_add_inserts_new_variable(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([9.0, 10.0], dtype=np.float64),
        label="new",
        var_id=1,
        cpp_name="cpp.new_var",
    )
    srd_basic["cpp.new_var"] = di
    assert "cpp.new_var" in srd_basic


def test_setitem_replace_overwrites_existing_variable(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([99.0, 99.0, 99.0], dtype=np.float64),
        label="var_a",
        var_id=1,
        cpp_name="cpp.var_a",
    )
    srd_basic["cpp.var_a"] = di
    np.testing.assert_allclose(srd_basic["cpp.var_a"].value_np, [99.0, 99.0, 99.0])


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_assigns_synthetic_negative_id(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([5.0, 6.0], dtype=np.float64),
        label="extra",
        var_id=1,
        cpp_name="cpp.extra",
    )
    srd_basic.add("cpp.extra", di)
    new_id = srd_basic.cpp_name_to_id["cpp.extra"]
    assert new_id < 0


def test_add_raises_if_already_exists(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([5.0, 6.0], dtype=np.float64),
        label="var_a",
        var_id=1,
        cpp_name="cpp.var_a",
    )
    with pytest.raises(KeyError):
        srd_basic.add("cpp.var_a", di)


def test_add_makes_variable_retrievable(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([7.0, 8.0], dtype=np.float64),
        label="added",
        var_id=1,
        cpp_name="cpp.added",
    )
    srd_basic.add("cpp.added", di)
    result = srd_basic["cpp.added"]
    np.testing.assert_allclose(result.value_np, [7.0, 8.0])


def test_add_populates_all_dicts(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64),
        value_np=np.array([1.0, 2.0], dtype=np.float64),
        label="new",
        var_id=1,
        cpp_name="cpp.new",
    )
    srd_basic.add("cpp.new", di)
    new_id = srd_basic.cpp_name_to_id["cpp.new"]
    assert srd_basic.id_to_cpp_name[new_id] == "cpp.new"
    assert new_id in srd_basic.id_to_instance
    assert new_id in srd_basic.id_to_descript


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------


def test_replace_overwrites_values(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        label="var_a",
        var_id=1,
        cpp_name="cpp.var_a",
    )
    srd_basic.replace("cpp.var_a", di)
    np.testing.assert_allclose(srd_basic["cpp.var_a"].value_np, [10.0, 20.0, 30.0])


def test_replace_preserves_var_id(srd_basic):
    original_id = srd_basic.cpp_name_to_id["cpp.var_a"]
    di = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        label="var_a",
        var_id=999,
        cpp_name="cpp.var_a",
    )
    srd_basic.replace("cpp.var_a", di)
    assert srd_basic.cpp_name_to_id["cpp.var_a"] == original_id


def test_replace_raises_if_not_found(srd_basic):
    di = DataInstance(
        timestamp_np=np.array([0], dtype=np.int64),
        value_np=np.array([0.0], dtype=np.float64),
        label="ghost",
        var_id=1,
        cpp_name="cpp.ghost",
    )
    with pytest.raises(KeyError):
        srd_basic.replace("cpp.ghost", di)
