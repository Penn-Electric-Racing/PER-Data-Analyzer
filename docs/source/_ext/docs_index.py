from __future__ import annotations

from pathlib import Path

from docutils import nodes
from docutils.frontend import get_default_settings
from docutils.io import StringOutput
from docutils.utils import new_document
from pydantic import BaseModel, Field
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.writers.text import TextWriter

# Object types kept inside their parent's entry, e.g. a pydantic model's fields
MEMBER_KINDS = {"attribute", "pydantic_field"}


class DocsIndexEntry(BaseModel):
    name: str = Field(description="Fully qualified name, or the docname of a page")
    kind: str = Field(description="Object type, e.g. class or method, or page")
    summary: str = Field(
        description="First paragraph of the docstring, or the page title"
    )
    text: str = Field(
        description="The object's documentation, as the text builder renders it"
    )


class DocsIndex(BaseModel):
    entries: list[DocsIndexEntry] = Field(
        description="Every documented object and page"
    )


# Filled page by page during the build, then written out by write_index
ENTRIES: list[DocsIndexEntry] = []


def render_text(app: Sphinx, nodes_to_render: list[nodes.Node]) -> str:
    """
    Render doctree nodes to plain text, as the text builder would render them on a page.

    Parameters
    ----------
    app : Sphinx
        The running Sphinx application. Its builder must be the text builder.
    nodes_to_render : list of nodes.Node
        Nodes to render, detached from their page (pass copies).

    Returns
    -------
    str
        The rendered text, stripped.

    Notes
    -----
    Once a page's doctree is final (the ``doctree-resolved`` event), this extension renders each
    ``desc`` node to text with the same writer the text builder uses for whole pages.
    The writer renders whole documents, so the nodes go into a new, empty one. It needs the
    text writer's default settings, plus the build environment that Sphinx's translator reads.
    """
    writer = TextWriter(app.builder)
    settings = get_default_settings(writer)
    settings.env = app.env
    document = new_document("", settings)
    document.extend(nodes_to_render)
    writer.write(document, StringOutput(encoding="unicode"))
    return writer.output.strip()


def record_entries(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """
    Add an entry to ``ENTRIES`` for each object documented on a page.

    Called by Sphinx for every page once its doctree is final.

    Parameters
    ----------
    app : Sphinx
        The running Sphinx application.
    doctree : nodes.document
        The page's doctree.
    docname : str
        The page's path in the source tree without its extension, e.g. ``getting_started/index``.

    Notes
    -----
    Sphinx parses every page into a doctree, a tree of nodes. For each object it documents,
    autodoc adds one ``desc`` node with two children: a ``desc_signature`` (the signature line,
    whose ``ids`` hold the fully qualified name) and a ``desc_content`` (the rendered docstring).
    A class's ``desc_content`` holds its members' ``desc`` nodes in turn.

    - **Object entries:** a class's entry keeps its fields but drops its methods and nested
      classes, which get entries of their own. That keeps each entry about one object.
    - **Summary:** the first paragraph of the docstring. Autodoc's ``Bases: ...`` line is
      skipped, and so are objects with no docstring, which get an empty summary.
    - **Pages without objects:** a hand-written page such as Getting Started becomes one
      ``page`` entry. Autosummary's generated package pages are skipped, since they only list
      their modules.
    """
    # Under any other builder, TextWriter would get that builder's translator instead
    if app.builder.name != "text":
        return

    descs = [
        desc
        for desc in doctree.findall(addnodes.desc)
        if desc["objtype"] not in MEMBER_KINDS
    ]
    if not descs:
        if docname.startswith("_autosummary/"):
            return
        title = next(doctree.findall(nodes.title), None)
        text = render_text(app, [child.deepcopy() for child in doctree.children])
        if title is not None and text:
            ENTRIES.append(
                DocsIndexEntry(
                    name=docname, kind="page", summary=title.astext(), text=text
                )
            )
        return

    for desc in descs:
        # Remove nested methods and classes from a copy, leaving the page itself intact
        node = desc.deepcopy()
        for nested in list(node.findall(addnodes.desc))[1:]:
            if nested["objtype"] not in MEMBER_KINDS and nested.parent is not None:
                nested.parent.remove(nested)
        content = next(node.findall(addnodes.desc_content))
        summary = next(
            (
                child.astext()
                for child in content.children
                if isinstance(child, nodes.paragraph)
                and not child.astext().startswith("Bases: ")
            ),
            "",
        )
        ENTRIES.append(
            DocsIndexEntry(
                name=next(desc.findall(addnodes.desc_signature))["ids"][0],
                kind=desc["objtype"],
                summary=" ".join(summary.split()),
                text=render_text(app, [node]),
            )
        )


def write_index(app: Sphinx, exception: Exception | None) -> None:
    """
    Write every recorded entry, sorted by name, to ``docs_index.json`` in the build directory.

    Called by Sphinx once the build ends. Sorting by name puts each class just above its methods.
    Writes nothing if the build failed or wasn't a text build.
    """
    if exception is not None or app.builder.name != "text":
        return
    docs_index = DocsIndex(entries=sorted(ENTRIES, key=lambda entry: entry.name))
    (Path(app.outdir) / "docs_index.json").write_text(
        docs_index.model_dump_json(indent=2), encoding="utf-8"
    )


def setup(app: Sphinx) -> None:
    """
    Register the extension's handlers with Sphinx.
    The extension saves one entry per object, and writes them all to ``docs_index.json`` once the build finishes.
    """
    app.connect("doctree-resolved", record_entries)
    app.connect("build-finished", write_index)
