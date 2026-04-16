# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PERDA (PER-Data-Analyzer) is Penn Electric Racing's in-house Python library for parsing and analyzing log files produced by the car. The primary user-facing API is the `Analyzer` class; the Jupyter notebook `notebooks/Tutorial.ipynb` serves as both documentation and integration demo.

## Architecture

The library has four layers:

1. **Analyzer** (`perda/analyzer/`) — the API for generic analysis tasks. Users typically interact with this class
   - Reduce complexity in this class by black-boxing functionality into the layers below

2. **Core data structures** (`perda/core_data_structures`) — supporting classes and data structures for the `Analyzer` to function. Users may occasionally interact with these directly for more advanced use cases.
   - `DataInstance` — a single time-series variable (Pydantic model wrapping NumPy arrays for timestamps + values). Supports arithmetic operators and join operations. Timestamps must be monotonically non-decreasing float64.
   - `SingleRunData` — dictionary-like container for all variables in one run, supporting lookup by variable ID or name.
   - `joins.py` — timestamp-based alignment via left/inner/outer join with interpolation.

3. **Utilities** (`perda/utils/`) — helper functions that reduce complexity in the main Analyzer by black-boxing some functionality.
   - `search.py` — keyword scoring against variable names/descriptions for natural language variable lookup
   - `diff.py` — comparison of `SingleRunData` object against a backup `SingleRunData` object created from backup server data, reporting variable and value mismatches
   - `integrate.py` — time-series integration
   - `data_summary.py` — summary statistics
   - `frequency_analysis.py` — frequency analysis of a data instance, to validate sampling rates
   - `gps_analysis.py` — GPS-specific logic: timestamp alignment, velocity-based trimming, and graph creation using function in `perda/plotting/`

4. **Plotting** (`perda/plotting/`):
   - `data_instance_plotter.py` — Plotly-based interactive visualization, supports single and dual y-axis.
   - `parametric_plot.py` — Generic 2D parametric curve plotting (neither axis needs to be a function of the other)
      - `plot_parametric_curve` — Static Plotly figure of a 2D parametric curve
      - `plot_parametric_curve_square` — Same, but with equal axes and a square aspect ratio
      - `plot_parametric_trimmer` — Interactive trimmer widget for a parametric curve, with optional timestamp labels
   - `plotting_constants.py` — All config objects that can be used to configure our plotting functions, as well as sensible defaults

5. **Text Encoder Models** — Language models used in this codebase are installed lazily, upon calling `search()`, and packaged under: `perda/models/`.

## Code Conventions

- Provide non-null default arguments directly in the function signature when possible, to avoid null checks in the function body

- Any complex, multi-field datatype should be made into a Pydantic model for type safety and validation. This includes all configuration objects for plotting.

- Graphs should always be generated using Plotly

- Graphing APIs should expose arguments that allow customization of important parts of the graph's appearance

- Reduce variables to graphing APIs by grouping related arguments into custom strongly typed configuration objects, defined in `plotting_constants.py`.

- Whenever possible, provide sensible defaults through default versions of the config objects to minimize required user configuration

- Separate graphs and logic. Try to make graphing utilities generic, reusable, and modularized in `plotting`

- Functions in `plotting` can have more arguments but should not have any arguments that are specific to a particular analysis or use case. Business logic should be responsible for calling these graphing utilities and reducing complexity by reducing the number of arguments exposed.

- Package-wide defaults should be defined in `constants.py`

- Units and conversion factors should be defined in `units.py`. No magic numbers in code

- Unit tests should not be bundled in classes. Write individual functions.

- If specific state is needed across multiple tests, use pytest fixtures to set up that state in a consistent way.

## Style Guide

- To run any commands or code, a Python virtual environment, setup in the root of the repository, must be activated. All new dependencies should be added to `requirements.txt` and installed in the virtual environment.

- Functions and methods must have concise numpy-style docstrings with
   - Parameters section: name, type, and description on the next line (indented)
   - Returns section: type and description on the next line (indented)
   - Optional Notes section only used in very special cases with important implementation details or caveats
   - Optional Examples section for user-facing API functions, with simple code snippets demonstrating usage

- Code style should be checked with 
   - `pre-commit run --all-files` will automatically format and lint
   - Pre-commit will raise `mypy` errors, but they need to be fixed manually
   - All numpy typing must be precise (e.g. `np.ndarray` is not allowed; use `NDArray[Float64]` instead). Use `numpy.typing` for this purpose.

- DO NOT use scoped imports, only use imports at the top of the file. When you have many imports, use asterisk to avoid long import lists

- Whenever possible, control print() statement spacing with python format specifiers and alignment options. For large groups of print statements, refactor spacing constants out into a local variable, for easier readability and maintainability. 

- Whenever possible, create and leverage __str__ representations of objects instead of manually writing a string representation, to avoid duplicate code.