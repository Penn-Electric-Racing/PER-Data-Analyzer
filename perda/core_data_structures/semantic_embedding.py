from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable

import faiss
import numpy as np
from numpy.typing import NDArray

from ..constants import SENTENCE_EMBEDDING_DIMENSION, SENTENCE_EMBEDDING_MODEL_ID
from .search_indexes import VariableSemanticIndex


@lru_cache(maxsize=1)
def load_encoder(model_id: str) -> Callable[[list[str]], NDArray]:
    """
    Load a sentence-transformers model and return its bound encode callable. Cached so
    the model is loaded at most once per process and shared by every run.

    Parameters
    ----------
    model_id : str
        HuggingFace identifier of the model to load.

    Notes
    -----
    Raises ImportError when the ``semantic`` extra is absent; callers are expected to
    catch that and fall back to keyword search. Callers coerce the returned vectors to
    float32, so the encoder's own dtype is not relied on.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id).encode


def _empty_faiss_index() -> faiss.IndexIDMap2:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(SENTENCE_EMBEDDING_DIMENSION))


def _ids_array(var_ids: Iterable[int]) -> NDArray[np.int64]:
    return np.fromiter(var_ids, dtype=np.int64)


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Encode texts into L2-normalized vectors, one row per text."""
    encode = load_encoder(SENTENCE_EMBEDDING_MODEL_ID)
    vectors = np.asarray(encode(texts), dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def update_semantic_index(
    index: VariableSemanticIndex, id_to_descript: dict[int, str]
) -> None:
    """
    Embed and store the given descriptions in place, replacing any older vector per ID.

    Parameters
    ----------
    index : VariableSemanticIndex
        Index to update.
    id_to_descript : dict[int, str]
        Descriptions to index, keyed by variable ID. IDs whose description is already
        indexed are skipped, so no embedding work is done for them.
    """
    changed = {
        var_id: descript
        for var_id, descript in id_to_descript.items()
        if index.id_to_descript.get(var_id) != descript
    }
    if not changed:
        return

    var_ids = _ids_array(changed)
    index.faiss_index.remove_ids(faiss.IDSelectorBatch(var_ids))
    index.faiss_index.add_with_ids(embed_texts(list(changed.values())), var_ids)
    index.id_to_descript.update(changed)


def clone_semantic_index(index: VariableSemanticIndex) -> VariableSemanticIndex:
    """Independent copy of a semantic index, so updating one never changes the other."""
    var_ids = _ids_array(index.id_to_descript)
    faiss_index = _empty_faiss_index()
    faiss_index.add_with_ids(index.faiss_index.reconstruct_batch(var_ids), var_ids)
    return VariableSemanticIndex(
        faiss_index=faiss_index, id_to_descript=dict(index.id_to_descript)
    )


def build_semantic_index(
    id_to_descript: dict[int, str], verbose: int = 1
) -> VariableSemanticIndex | None:
    """
    Encode every variable description into a L2-normalized and FAISS inner-product index.
    Cosine similarity scores range [-1, 1].

    Parameters
    ----------
    id_to_descript : dict[int, str]
        Mapping from variable ID to its human-readable description.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 or higher for status and warnings. Default is 1.

    Returns
    -------
    VariableSemanticIndex | None
        The built index, or None if the optional dependencies or model are unavailable.
    """
    if not id_to_descript:
        return None

    try:
        if verbose >= 1:
            print("Building semantic search index...")

        index = VariableSemanticIndex(
            faiss_index=_empty_faiss_index(), id_to_descript={}
        )
        update_semantic_index(index, id_to_descript)
        return index
    except ImportError:
        if verbose >= 1:
            print(
                "Warning: sentence-transformers is not installed. "
                "Falling back to keyword-only search."
            )
        return None
    except Exception as e:
        if verbose >= 1:
            print(
                f"Warning: semantic index unavailable ({e}). "
                "Falling back to keyword-only search."
            )
        return None
