import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance


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


@pytest.mark.parametrize(
    "incoming_var_id, expected_var_id",
    [
        pytest.param(None, 3, id="no_id_gets_fresh"),
        pytest.param(50, 50, id="free_id_kept"),
        pytest.param(1, 3, id="taken_id_gets_fresh"),
    ],
)
def test_setitem_add_var_id(srd_basic, incoming_var_id, expected_var_id):
    srd_basic["cpp.new"] = DataInstance(
        timestamp_np=np.array([0], dtype=np.int64),
        value_np=np.array([1.0]),
        var_id=incoming_var_id,
    )
    assert srd_basic.cpp_name_to_id["cpp.new"] == expected_var_id
    assert srd_basic[expected_var_id].cpp_name == "cpp.new"


def test_setitem_add_derived_from_existing_variable(srd_basic):
    srd_basic["cpp.doubled"] = srd_basic["cpp.var_a"] * 2
    np.testing.assert_allclose(srd_basic["cpp.doubled"].value_np, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(srd_basic["cpp.var_a"].value_np, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "new_label, expected_label",
    [
        pytest.param("renamed", "renamed", id="new_label_honored"),
        pytest.param(None, "var_a", id="missing_label_kept"),
    ],
)
def test_setitem_replace_label(srd_basic, new_label, expected_label):
    srd_basic["cpp.var_a"] = DataInstance(
        timestamp_np=np.array([0], dtype=np.int64),
        value_np=np.array([0.0]),
        label=new_label,
    )
    assert srd_basic["cpp.var_a"].label == expected_label
    assert srd_basic.id_to_descript[1] == expected_label


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"var_id": 2}, id="different_var_id"),
        pytest.param({"cpp_name": "cpp.other"}, id="different_cpp_name"),
    ],
)
def test_setitem_replace_rejects_metadata_change(srd_basic, metadata):
    with pytest.raises(ValueError):
        srd_basic["cpp.var_a"] = DataInstance(
            timestamp_np=np.array([0], dtype=np.int64),
            value_np=np.array([0.0]),
            **metadata,
        )


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(("cpp.var_a", "cpp.new"), id="names"),
        pytest.param((1, "cpp.new"), id="id_and_name"),
    ],
)
def test_setitem_multiple_keys(srd_basic, key):
    srd_basic[key] = (
        DataInstance(timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([7.0])),
        DataInstance(timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([8.0])),
    )
    np.testing.assert_allclose(srd_basic["cpp.var_a"].value_np, [7.0])
    np.testing.assert_allclose(srd_basic["cpp.new"].value_np, [8.0])


def test_setitem_by_id_replaces_existing_variable(srd_basic):
    srd_basic[2] = DataInstance(
        timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([5.0])
    )
    np.testing.assert_allclose(srd_basic["cpp.var_b"].value_np, [5.0])
    assert srd_basic.cpp_name_to_id["cpp.var_b"] == 2


def test_setitem_pattern_assigns_in_sorted_order(srd_basic):
    srd_basic["cpp.var_*"] = (
        DataInstance(timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([10.0])),
        DataInstance(timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([20.0])),
    )
    np.testing.assert_allclose(srd_basic["cpp.var_a"].value_np, [10.0])
    np.testing.assert_allclose(srd_basic["cpp.var_b"].value_np, [20.0])


@pytest.mark.parametrize(
    "key, value_count, exc_type",
    [
        pytest.param("cpp.var_*", 1, ValueError, id="pattern_count_mismatch"),
        pytest.param(("cpp.x", "cpp.y"), 3, ValueError, id="tuple_count_mismatch"),
        pytest.param("ghost.*", 1, KeyError, id="pattern_no_match"),
        pytest.param(999, 1, KeyError, id="unknown_id"),
    ],
)
def test_setitem_multiple_invalid_raises(srd_basic, key, value_count, exc_type):
    with pytest.raises(exc_type):
        srd_basic[key] = tuple(
            DataInstance(timestamp_np=np.array([0], dtype=np.int64), value_np=np.array([0.0]))
            for _ in range(value_count)
        )


@pytest.mark.parametrize(
    "key, expected_total",
    [
        pytest.param("cpp.new", 8, id="add"),
        pytest.param("cpp.var_a", 5, id="replace"),
    ],
)
def test_setitem_updates_total_data_points(srd_basic, key, expected_total):
    srd_basic[key] = DataInstance(
        timestamp_np=np.array([0, 1], dtype=np.int64), value_np=np.array([0.0, 1.0])
    )
    assert srd_basic.total_data_points == expected_total


@pytest.mark.parametrize(
    "key, labels, expected_text, expected_batches",
    [
        pytest.param(
            "cpp.new",
            ["Brake pressure"],
            {3: "cpp new brake pressure"},
            [["Brake pressure"]],
            id="add",
        ),
        pytest.param(
            "cpp.var_a",
            ["Brake pressure"],
            {1: "cpp var_a brake pressure"},
            [["Brake pressure"]],
            id="relabel",
        ),
        pytest.param(
            ("cpp.new_a", "cpp.new_b"),
            ["a", "b"],
            {3: "cpp new_a a", 4: "cpp new_b b"},
            [["a", "b"]],
            id="multiple_in_one_batch",
        ),
    ],
)
def test_setitem_updates_search_indexes(
    srd_semantic, fake_encoder, key, labels, expected_text, expected_batches
):
    batches_before = len(fake_encoder)
    srd_semantic[key] = tuple(
        DataInstance(timestamp_np=np.array([0]), value_np=np.array([1.0]), label=label)
        for label in labels
    )

    keyword_text = srd_semantic.keyword_index.id_to_normalized_text
    assert {var_id: keyword_text[var_id] for var_id in expected_text} == expected_text
    assert fake_encoder[batches_before:] == expected_batches
    assert srd_semantic.semantic_index.id_to_descript == srd_semantic.id_to_descript
    assert srd_semantic.semantic_index.faiss_index.ntotal == len(srd_semantic.id_to_cpp_name)


def test_setitem_skips_embedding_when_description_unchanged(srd_semantic, fake_encoder):
    srd_semantic["cpp.var_a"] = DataInstance(
        timestamp_np=np.array([0]), value_np=np.array([1.0]), label="Brake pressure"
    )
    batches_before = len(fake_encoder)
    srd_semantic["cpp.var_a"] = DataInstance(
        timestamp_np=np.array([0, 1]), value_np=np.array([1.0, 2.0])
    )
    assert len(fake_encoder) == batches_before
