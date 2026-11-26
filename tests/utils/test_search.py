import numpy as np
import pytest

from perda.analyzer.csv import parse_csv
from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.search_indexes import normalize_search_text
from perda.utils.search import keyword_score, search


def test_parse_degrades_gracefully_without_sentence_transformers(
    search_csv, no_sentence_transformers
):
    srd = parse_csv(search_csv(), verbose=0, build_search_index=True)
    assert srd.semantic_index is None

    results = search(srd, "voltage")
    assert results[0].cpp_name == "ams.pack.voltage"


@pytest.mark.parametrize(
    "query, top_n, message",
    [
        ("voltage", 0, "top_n must be a positive integer."),
        ("voltage", -1, "top_n must be a positive integer."),
        ("   ", 10, "Search query cannot be empty."),
        ("!!!", 10, "Search query must contain letters or numbers."),
    ],
)
def test_search_rejects_invalid_input(search_csv, query, top_n, message):
    srd = parse_csv(search_csv(), verbose=0)
    with pytest.raises(ValueError, match=message):
        search(srd, query, top_n=top_n)


def test_search_respects_top_n(search_csv):
    srd = parse_csv(search_csv(), verbose=0)
    results = search(srd, "speed", top_n=1)
    assert len(results) == 1


def test_search_ranks_are_sequential(search_csv):
    srd = parse_csv(search_csv(), verbose=0)
    results = search(srd, "wheel speed")
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


@pytest.mark.parametrize(
    "terms, normalized_text, expected",
    [
        (["voltage"], "ams pack voltage voltage", 1.0),
        (["wheel"], "pcm wheel speeds front right", 1.0),
        (["zzzz"], "ams pack voltage voltage", 0.0),
    ],
)
def test_keyword_score(terms, normalized_text, expected):
    assert keyword_score(terms, normalized_text) == pytest.approx(expected, abs=0.25)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ams.pack.voltage", "ams pack voltage"),
        ("pcm.wheelSpeeds.frontRight", "pcm wheel speeds front right"),
        ("  Mixed.CASE  ", "mixed case"),
    ],
)
def test_normalize_search_text(raw, expected):
    assert normalize_search_text(raw) == expected


def test_keyword_index_covers_all_variables(search_csv):
    srd = parse_csv(search_csv(), verbose=0)
    assert list(srd.keyword_index.id_to_normalized_text) == list(srd.id_to_cpp_name)


@pytest.mark.parametrize(
    "build_search_index", [False, True], ids=["keyword", "semantic"]
)
@pytest.mark.parametrize(
    "cpp_name, label",
    [
        pytest.param("test.newVar", "brake pressure", id="add"),
        pytest.param("ams.pack.voltage", "accumulator potential", id="relabel"),
    ],
)
def test_search_finds_assigned_variable(
    search_csv, fake_encoder, build_search_index, cpp_name, label
):
    srd = parse_csv(search_csv(), verbose=0, build_search_index=build_search_index)
    srd[cpp_name] = DataInstance(
        timestamp_np=np.array([0]), value_np=np.array([1.0]), label=label
    )

    assert search(srd, label, top_n=1)[0].cpp_name == cpp_name
