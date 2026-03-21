# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PERDA (PER-Data-Analyzer) is Penn Electric Racing's Python library for parsing and analyzing CAN bus telemetry log files. The primary user-facing API is the `Analyzer` class; the Jupyter notebook `notebooks/Tutorial.ipynb` serves as both documentation and integration demo.

## Architecture

The library has four layers:

1. **Analyzer** (`perda/analyzer/analyzer.py`) — the public API. Exposes `parse_csv()`, `plot()`, `search()`, and `diff()`. This is the only class users interact with directly.

2. **Core data models** (`perda/analyzer/`):
   - `DataInstance` — a single time-series variable (Pydantic model wrapping NumPy arrays for timestamps + values). Supports arithmetic operators and join operations. Timestamps must be monotonically non-decreasing float64.
   - `SingleRunData` — dictionary-like container for all variables in one run, supporting lookup by variable ID or name.
   - `joins.py` — timestamp-based alignment via left/inner/outer join with interpolation.

3. **Utilities** (`perda/utils/`):
   - `search.py` — keyword scoring against variable names/descriptions for natural language variable lookup.
   - `diff.py` — comparison of two `SingleRunData` objects, reporting variable and value mismatches.
   - `integrate.py` — time-series integration.
   - `data_summary.py` — summary statistics.

4. **Plotting** (`perda/plotting/`):
   - `data_instance_plotter.py` — Plotly-based interactive visualization, supports single and dual y-axis.
   - `gps_trimmer.py` — GPS-specific plot trimming.
   - `plotting_constants.py` — All config objects that can be used to configure our plotting functions
   - `diff_plotter.py` — Overlaid bar chart plots for visualizing diff results.

## CSV Format

Input CSV files follow the CAN bus log format:

```
# Header lines: "Value [description] (cpp.name): id"
timestamp, variable_id, value
```

Parsing is in `perda/analyzer/csv.py`. It tolerates unordered timestamps and allows configurable error limits via `parsing_errors_limit`.

## Code Conventions

- Strict type annotations throughout; mypy is enforced via pre-commit.
- Black formatting and isort import sorting are enforced via pre-commit.
- Pydantic is used for data validation in `DataInstance` and related models.
- Graphs are generated using Plotly for interactivity, with a consistent color scheme and layout defined in `plotting_constants.py`. All plots should be configured using custom strongly typed config objects. Defaults are defined in `plotting_defaults.py`. Whenever possible, provide sensible defaults through default versions of the config objects to minimize required user configuration.
