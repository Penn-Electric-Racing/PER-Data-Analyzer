import re
from collections.abc import Iterable

WILDCARD = "*"


def is_pattern(name: str) -> bool:
    """Return True if ``name`` contains the ``*`` wildcard."""
    return WILDCARD in name


def match_pattern(pattern: str, names: Iterable[str]) -> list[str]:
    """
    Find every name that matches a wildcard pattern.

    Parameters
    ----------
    pattern : str
        Variable name where each ``*`` matches any run of characters (including none),
        e.g. ``"pcm.vnav.*"`` or ``"*.steeringWheel.angle"``. All other characters match literally.
    names : Iterable[str]
        Candidate variable names.

    Returns
    -------
    list[str]
        Matching names in sorted order.

    Examples
    --------
    >>> match_pattern("pcm.wheelSpeeds.front*", data.cpp_name_to_id)
    ['pcm.wheelSpeeds.frontLeft', 'pcm.wheelSpeeds.frontRight']
    """
    literal_parts = [re.escape(part) for part in pattern.split(WILDCARD)]
    regex = re.compile(".*".join(literal_parts))
    return sorted(name for name in names if regex.fullmatch(name))
