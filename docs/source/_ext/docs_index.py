from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sphinx.application import Sphinx

from perda.constants import SENTENCE_EMBEDDING_MODEL_ID


class DocsIndexEntry(BaseModel):
    name: str = Field(description="Fully qualified name of the documented object")
    kind: str = Field(description="Autodoc object type, e.g. class or method")
    docstring: str = Field(description="The object's complete docstring")


class DocsIndex(BaseModel):
    model: str = Field(description="Sentence-transformers model behind the vectors")
    entries: list[DocsIndexEntry] = Field(description="Every documented object")
    row_to_entry: list[int] = Field(
        description="Maps each FAISS row to its position in entries"
    )


ENTRIES: dict[str, DocsIndexEntry] = {}


def record_docstring(
    app: Sphinx, what: str, name: str, obj: object, options: object, lines: list[str]
) -> None:
    """Store the docstring of each object autodoc documents."""
    docstring = "\n".join(lines).strip()
    if docstring:
        ENTRIES[name] = DocsIndexEntry(name=name, kind=what, docstring=docstring)


def write_index(app: Sphinx, exception: Exception | None) -> None:
    """
    Embed every docstring into a FAISS index and write it next to the JSON entries.

    Notes
    -----
    The model reads at most 256 tokens, so each docstring paragraph gets its own vector.
    ``row_to_entry`` maps every vector back to its whole entry.
    """
    if exception is not None:
        return

    entries = sorted(ENTRIES.values(), key=lambda entry: entry.name)
    texts: list[str] = []
    row_to_entry: list[int] = []
    for position, entry in enumerate(entries):
        for paragraph in entry.docstring.split("\n\n"):
            texts.append(f"{entry.name}: {paragraph}")
            row_to_entry.append(position)

    vectors = (
        SentenceTransformer(SENTENCE_EMBEDDING_MODEL_ID)
        .encode(texts)
        .astype(np.float32)
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    docs_index = DocsIndex(
        model=SENTENCE_EMBEDDING_MODEL_ID,
        entries=entries,
        row_to_entry=row_to_entry,
    )

    # Outside build/ so the index isn't deployed to GitHub Pages
    outdir = Path(app.confdir).parent / "build_index"
    outdir.mkdir(exist_ok=True)
    faiss.write_index(index, str(outdir / "docs_index.faiss"))
    (outdir / "docs_index.json").write_text(
        docs_index.model_dump_json(indent=2), encoding="utf-8"
    )


def setup(app: Sphinx) -> None:
    # Run before napoleon and typehints (priority 500) rewrite the docstring into rST
    app.connect("autodoc-process-docstring", record_docstring, priority=400)
    app.connect("build-finished", write_index)
