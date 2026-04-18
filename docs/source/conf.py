#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PERDA"
copyright = "2025, Penn Electric Racing"
author = "Penn Electric Racing"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# -- Autodoc configuration ---------------------------------------------------
# Automatically generate stub pages for autosummary
autosummary_generate = True
autosummary_imported_members = False

# Add the parent directory to the path so we can import the package
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Autodoc Pydantic configuration ------------------------------------------
autodoc_pydantic_model_members = True
autodoc_pydantic_model_undoc_members = True

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_show_validator_members = True
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_model_member_order = "bysource"


extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.napoleon",  # For numpy/google style docstrings
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # Before autodoc_pydantic to avoid conflicts
    "sphinxcontrib.autodoc_pydantic",  # Must come after autodoc to override it
    "sphinx.ext.mathjax",  # For math rendering
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx_copybutton",  # For copy buttons on code blocks
    "sphinx_design",  # For better CSS
    "sphinx_markdown_builder"  # For MCP Markdown file
]

# Configure autodoc-typehints to work with pydantic
autodoc_typehints = "description"
typehints_use_signature = False

templates_path = ["_templates"]
exclude_patterns: list[str] = []
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "PERDA"
html_short_title = "PERDA"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 1,
    "collapse_navigation": False,
    "navbar_align": "content",
}
