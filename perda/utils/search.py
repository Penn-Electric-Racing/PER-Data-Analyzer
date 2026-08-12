import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from ..constants import DELIMITER, title_block
from ..core_data_structures.single_run_data import SingleRunData

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
_HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# SentenceTransformer instance when loaded, else None
_model: Any = None

ABBREVIATIONS: dict[str, str] = {
    "pcm": "powertrain control module",
    "pdu": "power distribution unit",
    "ams": "accumulator management system",
    "bms": "battery management system",
    "lvbms": "low voltage battery management system",
    "dash": "dashboard",
    "ludwig": "data acquisition dashboard",
    "daqdash": "data acquisition dashboard",
    "moc": "motor controller",
    "nav": "vectornav",
    "vnav": "vectornav",
    "ins": "inertial navigation system",
    "bat": "battery",
    "bspd": "brake system plausibility device",
    "rtds": "ready to drive sound",
    "imd": "insulation monitoring device",
    "flt": "fault",
    "smo": "sliding mode observer",
    "mma": "minimum maximum average",
    "aerorake": "aero rakes",
    "shockpot": "shock potentiometer",
    "regen": "regenerative braking",
}


class SearchEntry(BaseModel):
    """One entry in the search deck, holding raw variable data alongside its search card."""

    var_id: int = Field(description="Internal variable ID.")
    cpp_name: str = Field(description="C++ variable name used for data access.")
    descript: str = Field(description="Human-readable variable description.")
    card: str = Field(description="Space-separated search card text for scoring.")


class SearchResult(BaseModel):
    """A single ranked result returned by :func:`search`."""

    rank: int = Field(
        description="1-based position in the result list (1 = best match)."
    )
    score: float = Field(description="Relevance score (higher is better).")
    var_id: int = Field(description="Internal variable ID.")
    cpp_name: str = Field(description="C++ variable name used for data access.")
    descript: str = Field(description="Human-readable variable description.")

    def __str__(self) -> str:
        col_score, col_id, col_name, col_desc = 7, 4, 40, 60
        name = (
            self.cpp_name
            if len(self.cpp_name) <= col_name
            else self.cpp_name[: col_name - 1] + "…"
        )
        desc = (
            self.descript
            if len(self.descript) <= col_desc
            else self.descript[: col_desc - 1] + "…"
        )
        return f"{self.score:<{col_score}.2f}  {self.var_id:<{col_id}}  {name:<{col_name}}  {desc:<{col_desc}}"


def install_encoder() -> bool:
    """Download and save the SentenceTransformer model for semantic search.

    Returns
    -------
    bool
        True if the model loaded successfully, False otherwise.
    """
    global _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Warning: sentence-transformers is not installed. Falling back to keyword-only search."
        )
        return False

    try:
        if not _MODEL_DIR.exists():
            print("Downloading SentenceTransformer model (one-time setup)...")
            _model = SentenceTransformer(_HF_MODEL_ID)
            _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            _model.save(str(_MODEL_DIR))
            print(f"Model saved to: {_MODEL_DIR}")
        else:
            _model = SentenceTransformer(str(_MODEL_DIR))
        return True
    except Exception as e:
        print(
            f"Warning: SentenceTransformer model unavailable ({e}). Falling back to keyword-only search."
        )
        _model = None
        return False


def search(
    data: SingleRunData, query: str, top_n: int = 10, semantic: bool = False
) -> list[SearchResult]:
    """Search telemetry variables, print the top matches, and return them.

    Parameters
    ----------
    data : SingleRunData
        Parsed CSV telemetry data.
    query : str
        Free-text search query (e.g. "bat wheel").
    top_n : int
        Maximum number of results to return and display (default 10).
    semantic : bool
        If True, runs a hybrid search with sentence-transformer embeddings.
        If False, uses fast keyword-only rapidfuzz search (default).

    Returns
    -------
    list[SearchResult]
        Top matches in descending relevance order (at most ``top_n`` entries).
        Each entry exposes ``rank``, ``score``, ``var_id``, ``cpp_name``,
        and ``descript`` for programmatic access.
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    keyword_query = re.findall(r"[a-z0-9]+", query.lower())
    if not keyword_query:
        raise ValueError("Search query must contain letters or numbers.")

    # Lazy-build search deck
    if getattr(data, "_search_deck", None) is None:
        data._search_deck = build_search_deck(data)
    deck = data._search_deck

    if not deck:
        return []

    num_terms = len(keyword_query)
    semantic_ready = install_encoder() if semantic else False

    if semantic_ready and _model is not None:
        try:
            # Lazy-build card embeddings
            if getattr(data, "_search_embeddings", None) is None:
                cards = [e.card for e in deck]
                card_embeddings = _model.encode(cards, convert_to_numpy=True)
                norms = np.linalg.norm(card_embeddings, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                data._search_embeddings = card_embeddings / norms

            embeddings = data._search_embeddings

            # Encode query and project
            semantic_query = preprocess_query(query)
            query_emb = _model.encode(semantic_query, convert_to_numpy=True)
            q_norm = np.linalg.norm(query_emb)
            if q_norm > 0:
                query_emb = query_emb / q_norm
            else:
                query_emb = np.zeros_like(query_emb)

            # Dot-product similarity scores
            semantic_scores_arr = np.dot(embeddings, query_emb)
            semantic_scores = {
                idx: float(score) for idx, score in enumerate(semantic_scores_arr)
            }

            ranked = sorted(
                (
                    (
                        combine_scores(
                            semantic_scores.get(idx, 0.0),
                            keyword_score(keyword_query, entry),
                            num_terms,
                        ),
                        idx,
                    )
                    for idx, entry in enumerate(deck)
                ),
                reverse=True,
            )
        except Exception as e:
            print(
                f"Warning: Semantic search failed ({e}). Falling back to keyword search."
            )
            semantic_ready = False

    if not semantic_ready or _model is None:
        ranked = sorted(
            (
                (keyword_score(keyword_query, entry), idx)
                for idx, entry in enumerate(deck)
            ),
            reverse=True,
        )

    top = ranked[:top_n]

    results = [
        SearchResult(
            rank=i + 1,
            score=score,
            var_id=deck[idx].var_id,
            cpp_name=deck[idx].cpp_name,
            descript=deck[idx].descript,
        )
        for i, (score, idx) in enumerate(top)
    ]

    _print_search_results(query, results)

    return results


def _print_search_results(query: str, results: list[SearchResult]) -> None:
    """Print search results as a compact 4-column table.

    Parameters
    ----------
    query : str
        The original search query string.
    results : list[SearchResult]
        Ordered list of search results to display.
    """
    col_score, col_id, col_name, col_desc = 7, 4, 40, 60

    print(title_block("Search Results"))
    print(f"Query: {query}\n")
    print(
        f"{'Score':<{col_score}}  {'ID':<{col_id}}  {'C++ Name':<{col_name}}  {'Description':<{col_desc}}"
    )
    print(DELIMITER)
    for result in results:
        print(result)


def preprocess_query(query: str) -> str:
    """Expand domain abbreviations in a search query for semantic ranking.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    str
        Query with known abbreviations expanded and duplicate tokens removed.
    """
    terms: list[str] = []
    for term in re.findall(r"[a-z0-9]+", query.lower()):
        terms.append(term)
        if term in ABBREVIATIONS:
            terms.extend(ABBREVIATIONS[term].split())
    return " ".join(dict.fromkeys(terms))


def build_search_deck(data: SingleRunData) -> list[SearchEntry]:
    """Build the search deck from all variables in a run.

    Parameters
    ----------
    data : SingleRunData
        Parsed CSV telemetry data.

    Returns
    -------
    list[SearchEntry]
        One entry per variable, containing its ID, names, description, and search card.
    """
    return [
        SearchEntry(
            var_id=var_id,
            cpp_name=cpp_name,
            descript=data.id_to_descript[var_id],
            card=build_search_card(cpp_name, data.id_to_descript[var_id]),
        )
        for var_id, cpp_name in data.id_to_cpp_name.items()
    ]


def build_search_card(cpp_name: str, descript: str) -> str:
    """Build a search card for one variable.

    Splits the C++ identifier on separators and camelCase boundaries, expands
    known abbreviations inline, and appends the description.

    Parameters
    ----------
    cpp_name : str
        C++ variable name (e.g. "pcm.requestedTorque").
    descript : str
        Human-readable variable description.

    Returns
    -------
    str
        Space-separated card text ready for the cross-encoder and keyword scorer.
    """
    tokens: list[str] = []
    for segment in re.split(r"[._]", cpp_name):
        for part in re.sub(r"([a-z])([A-Z])", r"\1 \2", segment).split():
            lowered = part.lower()
            tokens.append(
                ABBREVIATIONS[lowered] if lowered in ABBREVIATIONS else lowered
            )
    return " ".join(dict.fromkeys(tokens)) + " " + descript.lower()


def keyword_score(query_terms: list[str], entry: SearchEntry) -> float:
    """Score a card against query terms using fuzzy partial matching.

    Uses rapidfuzz.fuzz.partial_ratio per term then averages. Handles
    prefixes, substrings, and minor typos naturally.

    Parameters
    ----------
    query_terms : list[str]
        Tokenized query terms.
    entry : SearchEntry
        The search entry to score.

    Returns
    -------
    float
        Mean fuzzy match score in [0, 1].
    """
    raw_text = entry.cpp_name + " " + entry.descript
    search_text = " ".join(
        re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text).split()
    ).lower()

    return sum(
        fuzz.partial_ratio(term, search_text) / 100.0 for term in query_terms
    ) / len(query_terms)


def combine_scores(
    semantic_score: float, keyword_score: float, num_terms: int
) -> float:
    """Combine semantic and keyword scores using a weighted blend.

    Short queries (fewer terms) get more keyword weight; longer queries lean
    on semantic relevance.

    Parameters
    ----------
    semantic_score : float
        Relevance score from the cross-encoder.
    keyword_score : float
        Relevance score from fuzzy keyword matching.
    num_terms : int
        Number of terms in the original query.

    Returns
    -------
    float
        Combined score
    """
    kw_weight = max(0.3, 0.6 - 0.05 * (num_terms - 1))
    combined = kw_weight * keyword_score + (1 - kw_weight) * semantic_score
    return combined
