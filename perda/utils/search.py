import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz

from ..constants import DELIMITER, title_block
from ..core_data_structures.single_run_data import SingleRunData

try:
    from sentence_transformers.cross_encoder import CrossEncoder

    _SEMANTIC_AVAILABLE: bool = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "stsb-cross-encoder"
_HF_MODEL_ID = "cross-encoder/stsb-distilroberta-base"

# CrossEncoder instance when loaded, else None
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

    var_id: int
    cpp_name: str
    descript: str
    card: str


def install_encoder() -> bool:
    """Download and save the cross-encoder model for semantic search.

    Returns
    -------
    bool
        True if the model loaded successfully, False otherwise.

    Notes
    -----
    Returns False immediately if ``sentence-transformers`` is not installed
    (i.e. ``perda[semantic]`` extra was not requested).
    Any download or filesystem error is caught and printed; the function
    returns False so callers fall back to keyword-only search.
    """
    global _model

    if not _SEMANTIC_AVAILABLE:
        return False

    try:
        if not _MODEL_DIR.exists():
            print("Downloading cross-encoder model (one-time setup)...")
            _model = CrossEncoder(_HF_MODEL_ID)
            _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            _model.save(str(_MODEL_DIR))
            print(f"Model saved to: {_MODEL_DIR}")
        else:
            _model = CrossEncoder(str(_MODEL_DIR))
        return True
    except Exception as e:
        print(
            f"Warning: cross-encoder model unavailable ({e}). Falling back to keyword-only search."
        )
        _model = None
        return False


def search(data: SingleRunData, query: str) -> None:
    """Search telemetry variables and print the top matches.

    Parameters
    ----------
    data : SingleRunData
        Parsed CSV telemetry data.
    query : str
        Free-text search query (e.g. "bat wheel").

    Notes
    -----
    When ``perda[semantic]`` is installed and the cross-encoder model loads
    successfully, results are ranked by a weighted blend of semantic score and
    rapidfuzz keyword score. Otherwise falls back to keyword-only scoring with
    no error raised.

    Short queries lean on keyword matching; longer queries lean on semantic
    ranking when the model is available.
    """
    semantic_ready = install_encoder()

    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    keyword_query = re.findall(r"[a-z0-9]+", query.lower())
    if not keyword_query:
        raise ValueError("Search query must contain letters or numbers.")

    deck = build_search_deck(data)
    num_terms = len(keyword_query)

    if semantic_ready and _model is not None:
        semantic_query = preprocess_query(query)
        # rank() returns dicts with "corpus_id" (index into deck) and "score"
        semantic_scores = {
            int(r["corpus_id"]): float(r["score"])
            for r in _model.rank(semantic_query, [e.card for e in deck])
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
    else:
        ranked = sorted(
            (
                (keyword_score(keyword_query, entry), idx)
                for idx, entry in enumerate(deck)
            ),
            reverse=True,
        )

    _print_search_results(
        query, [deck[idx] for _, idx in ranked[:10]], [s for s, _ in ranked[:10]]
    )


def _print_search_results(
    query: str, entries: list[SearchEntry], scores: list[float]
) -> None:
    """Print search results as a compact 4-column table.

    Parameters
    ----------
    query : str
        The original search query string.
    entries : list[SearchEntry]
        Ordered list of search entries to display.
    scores : list[float]
        Relevance scores corresponding to each entry.
    """
    col_score = 7
    col_id = 4
    col_name = 40
    col_desc = 60

    print(title_block("Search Results"))
    print(f"Query: {query}\n")
    print(
        f"{'Score':<{col_score}}  {'ID':<{col_id}}  {'C++ Name':<{col_name}}  {'Description':<{col_desc}}"
    )
    print(DELIMITER)
    for entry, score in zip(entries, scores):
        name = (
            entry.cpp_name
            if len(entry.cpp_name) <= col_name
            else entry.cpp_name[: col_name - 1] + "…"
        )
        desc = (
            entry.descript
            if len(entry.descript) <= col_desc
            else entry.descript[: col_desc - 1] + "…"
        )
        print(
            f"{score:<{col_score}.2f}  {entry.var_id:<{col_id}}  {name:<{col_name}}  {desc:<{col_desc}}"
        )


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
