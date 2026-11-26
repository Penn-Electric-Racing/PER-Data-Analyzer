import numpy as np

from perda.core_data_structures.semantic_embedding import (
    build_semantic_index,
    clone_semantic_index,
    copy_semantic_vectors,
    empty_semantic_index,
    update_semantic_index,
)


def test_build_semantic_index_returns_none_without_sentence_transformers(
    no_sentence_transformers,
):
    assert build_semantic_index({1: "voltage"}, verbose=0) is None


def test_build_semantic_index_returns_none_for_empty_mapping():
    assert build_semantic_index({}, verbose=0) is None


def test_build_semantic_index_returns_none_when_model_fails(failing_encoder):
    assert build_semantic_index({1: "voltage"}, verbose=0) is None


def test_copy_semantic_vectors_renames_ids(fake_encoder):
    source = build_semantic_index({1: "voltage", 2: "speed"}, verbose=0)
    target = empty_semantic_index()

    copy_semantic_vectors(source, target, {1: 10, 5: 11})

    assert target.id_to_descript == {10: "voltage"}
    np.testing.assert_array_equal(
        target.faiss_index.reconstruct(10), source.faiss_index.reconstruct(1)
    )


def test_clone_semantic_index_is_independent(fake_encoder):
    index = build_semantic_index({1: "voltage", 2: "speed"}, verbose=0)
    clone = clone_semantic_index(index)
    update_semantic_index(clone, {3: "brake"})

    assert index.faiss_index.ntotal == 2
    assert clone.faiss_index.ntotal == 3
    np.testing.assert_array_equal(
        clone.faiss_index.reconstruct(1), index.faiss_index.reconstruct(1)
    )
