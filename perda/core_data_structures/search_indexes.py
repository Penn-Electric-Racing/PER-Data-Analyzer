import re

import faiss
from pydantic import BaseModel, ConfigDict, Field


def normalize_search_text(text: str) -> str:
    """
    Lower-case text and split camelCase and period boundaries into separate words.

    Parameters
    ----------
    text : str
        Raw text, such as a C++ variable name joined with its description.

    Returns
    -------
    str
        Normalized text, e.g. "pcm.wheelSpeeds.frontRight" becomes
        "pcm wheel speeds front right".
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace(".", " ")
    return " ".join(spaced.split()).lower()


class VariableKeywordIndex(BaseModel):
    """Normalized searchable text per variable, built once at parse time."""

    row_to_var_id: list[int] = Field(
        description="Maps index row position to variable ID"
    )
    normalized_text: list[str] = Field(
        description="Lower-cased text of each variable's name and description, with "
        "camelCase and period boundaries split into separate words"
    )


class VariableSemanticIndex(BaseModel):
    """Vector index over variable description embeddings, built once at parse time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    faiss_index: faiss.IndexFlatIP = Field(
        description="Inner-product index holding L2-normalized description vectors"
    )
    row_to_var_id: list[int] = Field(
        description="Maps index row position to variable ID"
    )


def variable_keyword_index_from_csv_mappings(
    id_to_cpp_name: dict[int, str], id_to_descript: dict[int, str]
) -> VariableKeywordIndex:
    """
    Normalize every variable's name and description into searchable text.

    Parameters
    ----------
    id_to_cpp_name : dict[int, str]
        Mapping from variable ID to its C++ variable name.
    id_to_descript : dict[int, str]
        Mapping from variable ID to its human-readable description.

    Returns
    -------
    VariableKeywordIndex
        Normalized text for each variable, aligned with ``row_to_var_id``.
    """
    var_ids = list(id_to_cpp_name.keys())

    return VariableKeywordIndex(
        row_to_var_id=var_ids,
        normalized_text=[
            normalize_search_text(
                f"{id_to_cpp_name[var_id]} {id_to_descript.get(var_id, '')}"
            )
            for var_id in var_ids
        ],
    )
