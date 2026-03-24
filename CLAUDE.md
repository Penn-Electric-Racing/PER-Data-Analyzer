# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PERDA (PER-Data-Analyzer) is Penn Electric Racing's Python library for parsing and analyzing CAN bus telemetry log files. The primary user-facing API is the `Analyzer` class; the Jupyter notebook `notebooks/Tutorial.ipynb` serves as both documentation and integration demo.

## Architecture

The library has four layers:

1. **Analyzer** (`perda/analyzer/analyzer.py`) — the API for generic analysis tasks. Users typically interact with this class
   - `__init__()` — initializes the analyzer with default configuration
   - `plot()` — creates visualizations of the analyzed data
   - `search()` — searches for variables based on keywords
   - `diff()` — compares two datasets and highlights differences
   - `integrate()` — TODO
   - `summarize()` — TODO
   - `analyze_frequency()` — TODO
   - `__str__()` — string representation of the analyzer, equivalent to `summarize()`

2. **Core data models** (`perda/analyzer/`) - supporting classes and data structures for the `Analyzer` to function. Users may occasionally interact with these directly for more advanced use cases.
   - `DataInstance` — a single time-series variable (Pydantic model wrapping NumPy arrays for timestamps + values). Supports arithmetic operators and join operations. Timestamps must be monotonically non-decreasing float64.
   - `SingleRunData` — dictionary-like container for all variables in one run, supporting lookup by variable ID or name.
   - `joins.py` — timestamp-based alignment via left/inner/outer join with interpolation.

3. **Utilities** (`perda/utils/`) - helper functions that reduce complexity in the main Analyzer by black-boxing some functionality:
   - `search.py` — keyword scoring against variable names/descriptions for natural language variable lookup.
   - `diff.py` — comparison of `SingleRunData` object against a backup `SingleRunData` object created from backup server data, reporting variable and value mismatches.
   - `integrate.py` — time-series integration.
   - `data_summary.py` — summary statistics.

4. **Plotting** (`perda/plotting/`):
   - `data_instance_plotter.py` — Plotly-based interactive visualization, supports single and dual y-axis.
   - `gps_visualization.py` — GPS-specific visualization
      - `plot_gps_trimmer` — Function to create an interactive GPS trimmer
      - `create_representative_gps_image` — Function to create a representative static image of a GPS track colored by velocity
   - `plotting_constants.py` — All config objects that can be used to configure our plotting functions, as well as sensible defaults for these config objects.
      - `LayoutConfig` — (titles, height/width, etc.)
      - `FontConfig` — font sizes
      - `DiffPlotConfig` — all config related to the diff bar chart, including colors and bucket size.
   - `diff_plotter.py` — Overlaid bar chart plots for visualizing diff results.

## CSV Format

Input CSV files follow the CAN bus log format:

```
# Header lines: "Value [description] (cpp.name): id"
timestamp, variable_id, value
```

Parsing is in `perda/analyzer/csv.py`. It tolerates unordered timestamps and allows configurable error limits via `parsing_errors_limit`.

## Coding Standards and Styles

- Functions and methods must have concise numpy-style docstrings with
   - Parameters section: name, type, and description on the next line (indented)
   - Returns section: type and description on the next line (indented)
   - Optional Notes section only used in very special cases with important implementation details or caveats
   - Optional Examples section for user-facing API functions, with simple code snippets demonstrating usage

- Code style should be checked with 
   - `pre-commit run --all-files` will automatically format and lint
   - Pre-commit will raise `mypy` errors, but they need to be fixed manually
   - All numpy typing must be precise (e.g. `np.ndarray` is not allowed; use `NDArray[Float64]` instead). Use `numpy.typing` for this purpose.
- DO NOT use scoped imports, only use imports at the top of the file. When you have many imports, feel free to use asterisk to avoid long import lists
- Provide non-null default arguments directly in the function signature when possible, to avoid null checks in the function body

- Any complex, multi-field datatype should be made into a Pydantic model for type safety and validation. This includes all configuration objects for plotting.

- Graphs should always be generated using Plotly
- Graphing APIs should allow configurability by exposing arguments for custom strongly typed configuration objects, defined in `plotting_constants.py`
- Whenever possible, provide sensible defaults through default versions of the config objects to minimize required user configuration
