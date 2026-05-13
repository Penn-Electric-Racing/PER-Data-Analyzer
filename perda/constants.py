DELIMITER = "=" * 40
TITLE_DELIM = "=" * 5

R_EARTH_M = 6_371_000.0


def title_block(title: str) -> str:
    """
    Format a title surrounded by title delimiters.

    Parameters
    ----------
    title : str
        The title text

    Returns
    -------
    str
        Title formatted as ``===== Title =====``
    """
    return f"{TITLE_DELIM} {title} {TITLE_DELIM}"
