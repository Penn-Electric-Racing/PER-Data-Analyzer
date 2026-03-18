"""Semantic + keyword search. 

Each variable is represented by a search card: its C++ name +
description. Results is combined from semantic cross-encoder score with 
keyword matching. Short queries rely more on keyword
matching, while longer queries rely more on semantic ranking.
"""

import re
from functools import lru_cache
from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder

from ..analyzer.single_run_data import SingleRunData

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

_PACKAGED_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "stsb-cross-encoder"
_REPO_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "stsb-cross-encoder"
_HF_MODEL_ID = "cross-encoder/stsb-distilroberta-base"


def _resolve_model_dir() -> Path:
    """Return the first existing local model directory, or the packaged path."""
    for candidate in (_PACKAGED_MODEL_DIR, _REPO_MODEL_DIR):
        if candidate.exists():
            return candidate
    return _PACKAGED_MODEL_DIR


_LOCAL_MODEL_DIR = _resolve_model_dir()

# ---------------------------------------------------------------------------
# Domain abbreviations
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    "pcm": "powertrain control module",
    "pdu": "power distribution unit",
    "ams": "accumulator management system",
    "bms": "battery management system",
    "dash": "dashboard",
    "moc": "motor controller",
    "nav": "navigation",
    "bat": "battery",
    "bspd": "brake system plausibility device",
    "rtds": "ready to drive sound",
    "imd": "insulation monitoring device",
    "flt": "fault",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    query = _normalize(query)
    if not query:
        raise ValueError("Search query cannot be empty.")

    query_terms = _tokenize(query)
    if not query_terms:
        raise ValueError("Search query must contain letters or numbers.")

    model = _load_model()

    semantic_query = _preprocess_query(query)

    corpus: list[str] = []
    corpus_meta: list[tuple[int, str, str]] = []

    for var_id, cpp_name in data.id_to_cpp_name.items():
        descript = data.id_to_descript[var_id]

        if query == _normalize(cpp_name):
            print("Exact match found!")
            _print_result(var_id, cpp_name, descript, score=1.0)
            return

        corpus.append(_build_card(cpp_name, descript))
        corpus_meta.append((var_id, cpp_name, descript))

    if not corpus:
        print("No variables available to search.")
        return

    semantic_scores = {
        int(r["corpus_id"]): float(r["score"])
        for r in model.rank(semantic_query, corpus)
    }

    kw_weight = _keyword_weight(query_terms)
    ranked = []
    for idx, card in enumerate(corpus):
        _, cpp_name, descript = corpus_meta[idx]
        sem = semantic_scores.get(idx, 0.0)
        kw = _keyword_score(query_terms, cpp_name, descript, card)
        ranked.append((kw_weight * kw + (1.0 - kw_weight) * sem, idx))

    ranked.sort(key=lambda x: x[0], reverse=True)

    print("==== Search Results ====")
    print("Query: ", query)
    for score, idx in ranked[:10]:
        print("----------------------------")
        _print_result(*corpus_meta[idx], score=score)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().strip().split())


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from *text*."""
    return re.findall(r"[a-z0-9]+", _normalize(text))


def _tokenize_identifier(identifier: str) -> list[str]:
    """Split a telemetry identifier into searchable tokens.

    Splits on separators and camelCase boundaries, then expands known
    abbreviations.
    """
    tokens: list[str] = []
    for segment in re.split(r"[._]", identifier):
        if not segment:
            continue
        # Split camelCase: "requestedTorque" -> ["requested", "Torque"]
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", segment).split()
        for part in parts:
            lowered = part.lower()
            if lowered in ABBREVIATIONS:
                tokens.append(f"{ABBREVIATIONS[lowered]} ({part})")
            else:
                tokens.append(lowered)
    return tokens


def _preprocess_query(query: str) -> str:
    """Expand abbreviations and deduplicate tokens for semantic ranking."""
    terms: list[str] = []
    for term in _tokenize(query):
        terms.append(term)
        if term in ABBREVIATIONS:
            terms.extend(_tokenize(ABBREVIATIONS[term]))
    return " ".join(dict.fromkeys(terms))


# ---------------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------------


def _build_card(cpp_name: str, descript: str) -> str:
    """Build one search card from the variable name and description."""
    id_tokens = _tokenize_identifier(cpp_name)
    context = " ".join(dict.fromkeys(id_tokens))
    return f"{context} | {_normalize(descript)}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _keyword_weight(query_terms: list[str]) -> float:
    """Compute the blend weight between keyword and semantic scores.

    Short queries get more keyword weight; longer queries lean semantic.
    """
    if not query_terms:
        return 0.5

    n = len(query_terms)
    avg_len = sum(len(t) for t in query_terms) / n

    # Linear interpolation: 0.60 at n=1, 0.45 at n=2, 0.35 floor at n>=3
    base = max(0.35, 0.60 - 0.15 * (n - 1))

    # Abbreviation boost for short single-token queries
    if n == 1:
        ramp = max(0.0, min(1.0, 4.0 - avg_len))
        base += 0.20 * ramp

    return min(base, 1.0)


def _term_match_strength(
    term: str,
    card_tokens: list[str],
    card_text: str,
) -> float:
    """Return how strongly one query term matches a search card.

    Exact token matches score highest, followed by prefix and substring matches.
    """
    best = 0.0

    for token in card_tokens:
        if token == term:
            return 1.0

        coverage = len(term) / max(len(token), 1)
        if token.startswith(term):
            best = max(best, 0.7 + 0.3 * coverage)
        elif term in token:
            best = max(best, 0.4 + 0.4 * coverage)

    if best == 0.0 and term in card_text:
        best = 0.4 + 0.2 * min(len(term) / 10.0, 1.0)

    return min(best, 1.0)


def _keyword_score(
    query_terms: list[str],
    cpp_name: str,
    descript: str,
    card: str,
) -> float:
    """Aggregate keyword relevance for all query terms into ``[0, 1]``."""
    if not query_terms:
        return 0.0

    card_text = _normalize(f"{cpp_name} {descript} {card}")
    card_tokens = _tokenize(card_text)

    strengths = [_term_match_strength(t, card_tokens, card_text) for t in query_terms]

    raw = sum(strengths)
    coverage = sum(1 for s in strengths if s > 0.0) / len(query_terms)
    raw += 0.5 * coverage ** 2

    return min(raw / (len(query_terms) + 0.5), 1.0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    """Load and cache the cross-encoder, preferring a local copy."""
    if _LOCAL_MODEL_DIR.exists():
        return CrossEncoder(str(_LOCAL_MODEL_DIR))
    return CrossEncoder(_HF_MODEL_ID)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_result(
    var_id: int,
    cpp_name: str,
    descript: str,
    score: float,
) -> None:
    """Print a single ranked result."""
    print(f"Score: {score:.2f}")
    print(f"Variable ID: {var_id}")
    print(f"C++ Name: {cpp_name}")
    print(f"Description: {descript}")