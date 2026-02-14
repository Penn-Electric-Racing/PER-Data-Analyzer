import re

from sentence_transformers.cross_encoder import CrossEncoder

from ..analyzer.single_run_data import SingleRunData


ABBREVIATIONS = {
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


def search(
    data: SingleRunData,
    search: str,
) -> None:
    """
    Natural language search for available variables in SingleRunData.

    Parameters
    ----------
    data : SingleRunData
        Data structure containing CSV file data
    search : str
        Query
    """
    search = normalize_search_query(search)
    if not search:
        raise ValueError("Search query cannot be empty.")

    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    semantic_query = _expand_query(search)
    query_terms = _extract_terms(search)

    corpus = []
    corpus_meta = []

    for var_id in data.id_to_cpp_name.keys():
        descript = data.id_to_descript[var_id]
        cpp_name = data.id_to_cpp_name[var_id]

        if search == normalize_search_query(cpp_name):

            print("Exact match found!")
            print_single_result(data, cpp_name, descript, score=1.0)
            return

        corpus.append(create_card(cpp_name, descript))
        corpus_meta.append((cpp_name, descript))

    ranks = model.rank(semantic_query, corpus)

    semantic_scores = {
        int(rank["corpus_id"]): float(rank["score"])
        for rank in ranks
    }

    keyword_weight = _keyword_weight(query_terms)
    combined_ranks = []
    for idx, card in enumerate(corpus):
        cpp_name, descript = corpus_meta[idx]
        semantic_score = semantic_scores.get(idx, 0.0)
        keyword_score = _keyword_score(query_terms, cpp_name, descript, card)
        combined_score = (
            keyword_weight * keyword_score + (1.0 - keyword_weight) * semantic_score
        )
        combined_ranks.append((combined_score, idx))

    combined_ranks.sort(key=lambda x: x[0], reverse=True)

    print("==== Search Results ====")
    print("Query: ", search)
    for score, corpus_id in combined_ranks[:10]:
        print("----------------------------")
        cpp_name, descript = corpus_meta[corpus_id]

        print_single_result(data, cpp_name, descript, score)

def normalize_search_query(query: str) -> str:
    """
    Normalize the search query by converting to lowercase, removing extra whitespace

    Parameters
    ----------
    query : str
        The search query to normalize

    Returns
    -------
    str
        The normalized search query
    """
    basic_normalized = ' '.join(query.lower().strip().split())

    return basic_normalized

def print_single_result(
    data: SingleRunData,
    cpp_name: str,
    descript: str,
    score: float,
) -> None:
    var_id = data.cpp_name_to_id[cpp_name]

    print(f"Score: {score:.2f}")
    print(f"Variable ID: {var_id}")
    print(f"C++ Name: {cpp_name}")
    print(f"Description: {descript}")


def create_card(cpp_name: str, descript: str) -> str:
    """
    Create a card string for the search corpus.

    Parameters
    ----------
    cpp_name : str
        The C++ variable name
    descript : str
        The description of the variable

    Returns
    -------
    str
        A combined string of cpp_name and descript for the search corpus
    """

    tokens = []
    for segment in re.split(r"[._]", cpp_name):
        for token in advanced_split(segment):
            lowered = token.lower()
            if lowered in ABBREVIATIONS:
                lowered = ABBREVIATIONS[lowered] + " (" + token + ")"
            tokens.append(lowered)

    normalized_descript = normalize_search_query(descript)
    expanded_context = " ".join(dict.fromkeys(tokens))
    return f"{expanded_context} | {normalized_descript}"


def _extract_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_search_query(text))


def _expand_query(query: str) -> str:
    expanded_terms: list[str] = []
    for term in _extract_terms(query):
        expanded_terms.append(term)
        if term in ABBREVIATIONS:
            expanded_terms.extend(_extract_terms(ABBREVIATIONS[term]))
    return " ".join(dict.fromkeys(expanded_terms))


def _keyword_weight(query_terms: list[str]) -> float:
    if not query_terms:
        return 0.5

    if len(query_terms) == 1 and len(query_terms[0]) <= 3:
        return 0.8
    if len(query_terms) == 1:
        return 0.6
    if len(query_terms) == 2:
        return 0.45

    return 0.35


def _keyword_score(
    query_terms: list[str],
    cpp_name: str,
    descript: str,
    card: str,
) -> float:
    if not query_terms:
        return 0.0

    searchable_text = normalize_search_query(f"{cpp_name} {descript} {card}")
    searchable_tokens = _extract_terms(searchable_text)

    raw_score = 0.0
    matched_terms = 0
    for term in query_terms:
        if re.search(rf"\b{re.escape(term)}\b", searchable_text):
            raw_score += 1.0
            matched_terms += 1
        elif any(token.startswith(term) for token in searchable_tokens):
            raw_score += 0.7
            matched_terms += 1
        elif term in searchable_text:
            raw_score += 0.4
            matched_terms += 1

    if matched_terms == len(query_terms):
        raw_score += 0.5

    max_possible = len(query_terms) + 0.5
    return min(raw_score / max_possible, 1.0)
            

def advanced_split(camel_case_string: str) -> list[str]:
    """Splits a camelCase string into a list of words using regex."""
    # Inserts a period between a lowercase letter and an uppercase letter
    s1 = re.sub(r"([a-z])([A-Z])", r"\1.\2", camel_case_string)
    # Splits the resulting string by period
    return s1.split('.')