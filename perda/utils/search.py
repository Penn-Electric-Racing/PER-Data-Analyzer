"""Semantic + keyword search.

Each variable is represented by a search card combining its expanded C++
identifier tokens with its description. Results are ranked by a weighted blend
of cross-encoder semantic score and rapidfuzz keyword score. Short queries lean
on keyword matching; longer queries lean on semantic ranking.
"""

import re
from pathlib import Path

from pydantic import BaseModel
from rapidfuzz import fuzz
from sentence_transformers.cross_encoder import CrossEncoder

from ..analyzer.single_run_data import SingleRunData
from ..constants import DELIMITER, title_block

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "stsb-cross-encoder"
if not _MODEL_DIR.exists():
    raise RuntimeError(
        "Search model not found. Please restart your kernel with an internet "
        "connection so the model can be downloaded on import."
    )
_model = CrossEncoder(str(_MODEL_DIR))


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

    var_id: int
    cpp_name: str
    descript: str
    card: str


def search(data: SingleRunData, query: str) -> None:
    """Search telemetry variables and print the top matches.

    Parameters
    ----------
    data : SingleRunData
        Parsed CSV telemetry data.
    query : str
        Free-text search query (e.g. ``"bat wheel"``).

    Raises
    ------
    ValueError
        If the query is empty or has no alphanumeric terms.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    keyword_query = re.findall(r"[a-z0-9]+", query.lower())
    if not keyword_query:
        raise ValueError("Search query must contain letters or numbers.")

    semantic_query = preprocess_query(query)

    deck = build_search_deck(data)

    # rank() returns dicts with "corpus_id" (index into deck) and "score"
    semantic_scores = {
        int(r["corpus_id"]): float(r["score"])
        for r in _model.rank(semantic_query, [e.card for e in deck])
    }

    # Combine semantic and keyword scores
    num_terms = len(keyword_query)
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

    # Print top results
    print(title_block("Search Results"))
    print("Query: ", query)
    for score, idx in ranked[:10]:
        print(DELIMITER)
        print_result(deck[idx], score=score)


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
        C++ variable name (e.g. ``"pcm.requestedTorque"``).
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

    Uses ``rapidfuzz.fuzz.partial_ratio`` per term then averages. Handles
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
    on semantic relevance. The blend weight is computed by ``_keyword_weight``.

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


def print_result(entry: SearchEntry, score: float) -> None:
    """Print a single ranked search result.

    Parameters
    ----------
    entry : SearchEntry
        The search entry to display.
    score : float
        Combined relevance score.
    """
    print(f"Score: {score:.2f}")
    print(f"Variable ID: {entry.var_id}")
    print(f"C++ Name: {entry.cpp_name}")
    print(f"Description: {entry.descript}")
