from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

import faiss
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from ..constants import DELIMITER, title_block
from ..core_data_structures.search_indexes import (
    VariableSemanticIndex,
    normalize_search_text,
)
from ..core_data_structures.single_run_data import SingleRunData

HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

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


@lru_cache(maxsize=1)
def _load_encoder(model_id: str) -> Callable[[list[str]], NDArray]:
    """
    Load a sentence-transformers model and return its bound encode callable. Cached so
    the model is loaded at most once per process and shared by every run.

    Parameters
    ----------
    model_id : str
        HuggingFace identifier of the model to load.

    Notes
    -----
    Raises ImportError when the ``semantic`` extra is absent; callers are expected to
    catch that and fall back to keyword search. Callers coerce the returned vectors to
    float32, so the encoder's own dtype is not relied on.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id).encode


def build_semantic_index(
    id_to_descript: dict[int, str], verbose: int = 1
) -> VariableSemanticIndex | None:
    """
    Encode every variable description into a L2-normalized and FAISS inner-product index.
    Cosine similarity scores range [-1, 1].

    Parameters
    ----------
    id_to_descript : dict[int, str]
        Mapping from variable ID to its human-readable description.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 or higher for status and warnings. Default is 1.

    Returns
    -------
    VariableSemanticIndex | None
        The built index, or None if the optional dependencies or model are unavailable.
    """
    if not id_to_descript:
        return None

    try:
        if verbose >= 1:
            print("Building semantic search index...")

        encode = _load_encoder(HF_MODEL_ID)
        descriptions = list(id_to_descript.values())
        vectors = np.asarray(encode(descriptions), dtype=np.float32)

        faiss.normalize_L2(vectors)
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        faiss_index.add(vectors)

        return VariableSemanticIndex(
            faiss_index=faiss_index,
            row_to_var_id=list(id_to_descript.keys()),
        )
    except ImportError:
        if verbose >= 1:
            print(
                "Warning: sentence-transformers is not installed. "
                "Falling back to keyword-only search."
            )
        return None
    except Exception as e:
        if verbose >= 1:
            print(
                f"Warning: semantic index unavailable ({e}). "
                "Falling back to keyword-only search."
            )
        return None


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
    encode = _load_encoder(HF_MODEL_ID)
    query_vector = np.asarray(encode([query]), dtype=np.float32)
    faiss.normalize_L2(query_vector)

    num_results = min(top_n, index.faiss_index.ntotal)
    scores, rows = index.faiss_index.search(query_vector, num_results)

    return [
        SearchResult(
            rank=rank + 1,
            score=float(score),
            var_id=index.row_to_var_id[row],
            cpp_name=data.id_to_cpp_name[index.row_to_var_id[row]],
            descript=data.id_to_descript[index.row_to_var_id[row]],
        )
        for rank, (score, row) in enumerate(zip(scores[0], rows[0]))
        if row != -1
    ]


def _keyword_search(data: SingleRunData, query: str, top_n: int) -> list[SearchResult]:
    """Rank variables by fuzzy matching query terms against normalized text."""
    query_terms = normalize_search_text(query).split()
    index = data.keyword_index

    ranked = sorted(
        (
            (keyword_score(query_terms, text), index.row_to_var_id[row])
            for row, text in enumerate(index.normalized_text)
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
        f"{DELIMITER}"
    )
    for result in results:
        print(result)
