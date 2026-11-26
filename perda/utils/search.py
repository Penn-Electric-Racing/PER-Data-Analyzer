from __future__ import annotations

import re

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from ..constants import SECTION_DIVIDER, title_block
from ..core_data_structures.search_indexes import (
    VariableSemanticIndex,
    normalize_search_text,
)
from ..core_data_structures.semantic_embedding import embed_texts
from ..core_data_structures.single_run_data import SingleRunData

SCORE_COLUMN_WIDTH = 7
ID_COLUMN_WIDTH = 4
NAME_COLUMN_WIDTH = 40


class SearchResult(BaseModel):
    rank: int = Field(
        description="1-based position in the result list (1 = best match)."
    )
    score: float = Field(description="Relevance score (higher is better).")
    var_id: int = Field(description="Internal variable ID.")
    cpp_name: str = Field(description="C++ variable name used for data access.")
    descript: str = Field(description="Human-readable variable description.")

    def __str__(self) -> str:
        return (
            f"{self.score:<{SCORE_COLUMN_WIDTH}.2f}  "
            f"{self.var_id:<{ID_COLUMN_WIDTH}}  "
            f"{self.cpp_name:<{NAME_COLUMN_WIDTH}}  "
            f"{self.descript}"
        )


def search(data: SingleRunData, query: str, top_n: int = 10) -> list[SearchResult]:
    """
    Search telemetry variables, print the top matches, and return them.

    Parameters
    ----------
    data : SingleRunData
        Parsed CSV telemetry data.
    query : str
        Free-text search query (e.g. "front wheel speed").
    top_n : int, optional
        Maximum number of results to return and display. Default is 10.

    Returns
    -------
    list[SearchResult]
        Top matches in descending relevance order (at most ``top_n`` entries).

    Notes
    -----
    Uses semantic vector search when ``data.semantic_index`` was built at construction
    time, and fuzzy keyword matching otherwise.

    Examples
    --------
    >>> results = search(aly.data, "front wheel speed")
    >>> names = [r.cpp_name for r in results]
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    if not re.search(r"[a-z0-9]", query.lower()):
        raise ValueError("Search query must contain letters or numbers.")

    if not data.id_to_cpp_name:
        return []

    semantic_index = data.semantic_index
    if semantic_index is not None:
        results = _semantic_search(data, semantic_index, query, top_n)
    else:
        results = _keyword_search(data, query, top_n)

    _print_search_results(query, results)

    return results


def _semantic_search(
    data: SingleRunData,
    index: VariableSemanticIndex,
    query: str,
    top_n: int,
) -> list[SearchResult]:
    num_results = min(top_n, index.faiss_index.ntotal)
    scores, var_ids = index.faiss_index.search(embed_texts([query]), num_results)

    return [
        SearchResult(
            rank=rank + 1,
            score=float(score),
            var_id=int(var_id),
            cpp_name=data.id_to_cpp_name[int(var_id)],
            descript=data.id_to_descript[int(var_id)],
        )
        for rank, (score, var_id) in enumerate(zip(scores[0], var_ids[0]))
        if var_id != -1
    ]


def _keyword_search(data: SingleRunData, query: str, top_n: int) -> list[SearchResult]:
    """Rank variables by fuzzy matching query terms against normalized text."""
    query_terms = normalize_search_text(query).split()

    ranked = sorted(
        (
            (keyword_score(query_terms, text), var_id)
            for var_id, text in data.keyword_index.id_to_normalized_text.items()
        ),
        key=lambda scored: scored[0],
        reverse=True,
    )

    return [
        SearchResult(
            rank=rank + 1,
            score=float(score),
            var_id=var_id,
            cpp_name=data.id_to_cpp_name[var_id],
            descript=data.id_to_descript[var_id],
        )
        for rank, (score, var_id) in enumerate(ranked[:top_n])
    ]


def keyword_score(query_terms: list[str], normalized_text: str) -> float:
    """
    Score normalized variable text against query terms using fuzzy partial matching.

    Parameters
    ----------
    query_terms : list[str]
        Normalized, whitespace-split query terms.
    normalized_text : str
        A variable's normalized name and description, from a VariableKeywordIndex.

    Returns
    -------
    float
        Mean fuzzy match score in [0, 1].

    Notes
    -----
    Scoring each term separately makes matching order-independent, so "front wheel
    speed" still matches a variable stored as "wheel speeds front right".
    """
    return sum(
        fuzz.partial_ratio(term, normalized_text) / 100.0 for term in query_terms
    ) / len(query_terms)


def _print_search_results(query: str, results: list[SearchResult]) -> None:
    print(
        f"{title_block('Search Results')}\n"
        f"Query: {query}\n"
        f"{'Score':<{SCORE_COLUMN_WIDTH}}  "
        f"{'ID':<{ID_COLUMN_WIDTH}}  "
        f"{'C++ Name':<{NAME_COLUMN_WIDTH}}  "
        f"{'Description'}\n"
        f"{SECTION_DIVIDER}"
    )
    for result in results:
        print(result)
