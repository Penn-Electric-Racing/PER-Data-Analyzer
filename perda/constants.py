SECTION_DIVIDER = "=" * 40
TITLE_DELIMITER = "=" * 5

EARTH_RADIUS_METERS = 6_371_000.0

SENTENCE_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_DIMENSION = 384


def title_block(title: str) -> str:
    """Format a title as ``===== Title =====``"""
    return f"{TITLE_DELIMITER} {title} {TITLE_DELIMITER}"
