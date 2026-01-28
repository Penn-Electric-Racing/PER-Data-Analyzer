from ..analyzer.single_run_data import SingleRunData


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
    search = search.strip()
    if not search:
        raise ValueError("Search query cannot be empty.")
    search_list = search.lower().split(" ")

    query_hits = []
    for var_id in data.id_to_cpp_name.keys():
        descript = data.id_to_descript[var_id]
        cpp_name = data.id_to_cpp_name[var_id]

        score = _determine_query_hit(search_list, cpp_name, descript)
        if score:
            query_hits.append((score, var_id))

    # Sort by score descending
    query_hits.sort(reverse=True, key=lambda x: x[0])

    print("==== Search Results ====")
    for score, var_id in query_hits:
        descript = data.id_to_descript[var_id]
        cpp_name = data.id_to_cpp_name[var_id]

        print(f"Variable: {descript}")
        print(f"ID: {var_id}")
        print(f"C++ Name: {cpp_name}")
        print("-----------------------")


def _determine_query_hit(
    search_list: list[str],
    cpp_name: str,
    descript: str,
) -> int:
    """
    Determine if a variable matches the search query.

    Parameters
    ----------
    search_list : list[str]
        List of search terms
    cpp_name : str
        Variable name
    descript : str
        Variable description

    Returns
    -------
    int
        Integer representing how good the match is (larger means better).
        0 indicates no match
    """
    # TODO: IMPROVE CRITERION FOR MATCH QUALITY

    match = False

    match |= any(term in cpp_name.lower() for term in search_list)
    match |= any(term in descript.lower() for term in search_list)

    return 1 if match else 0
