import textwrap
import numpy as np
import pytest
from unittest.mock import patch

from perda.analyzer.csv import parse_csv
from perda.utils.search import search
from perda.core_data_structures.data_instance import DataInstance

def test_search_semantic_fallback(tmp_path):
    content = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 10:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
    """
    )
    p = tmp_path / "test_search.csv"
    p.write_text(content)
    srd = parse_csv(str(p), verbose=0)

    # Mock SentenceTransformer constructor to fail
    with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Mocked transformer loading failure")):
        # Semantic search should fail gracefully, issue warning, and fall back to keyword search
        results = search(srd, "voltage", semantic=True)
        assert len(results) > 0
        assert results[0].cpp_name == "ams.pack.voltage"


def test_search_cache_invalidation(tmp_path):
    content = textwrap.dedent(
        """\
        PER Log: Thu Jun 11 10:00:00 2026 v2.0
        Value voltage (ams.pack.voltage): 1
        0,1,12.5
    """
    )
    p = tmp_path / "test_cache.csv"
    p.write_text(content)
    srd = parse_csv(str(p), verbose=0)

    # Initial search populates the search deck cache
    search(srd, "voltage")
    assert srd._search_deck is not None

    # Mutate SingleRunData by adding a new variable
    new_di = DataInstance(timestamp_np=np.array([0]), value_np=np.array([1.0]), label="New custom variable")
    srd.add("test.new_var", new_di)

    # Assert caches are cleared
    assert srd._search_deck is None
    assert srd._search_embeddings is None

    # Verify new variable is searchable
    results = search(srd, "custom")
    assert len(results) > 0
    assert results[0].cpp_name == "test.new_var"
