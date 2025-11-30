"""Unified data fetcher for both CSV and computed variables.

This module provides utilities to fetch data from either the CSV file
(via perda) or from the computed variables registry, with a unified interface.
"""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Union, Optional

from perda.csv_parser import CSVParser
from perda.utils import get_data_slice_by_timestamp

from .computed_variables import get_computed_variable, ComputedVariable


# CSV path from environment variable
CSV_PATH = os.getenv("ACTIVE_CSV_PATH", "temp/16thMay13-52.csv")

# Cache for parsed CSV data to avoid re-parsing
_csv_cache = None
_csv_cache_path = None


def get_variable_data(variable_name: str, start_time: int = 0, end_time: int = -1):
    """
    Get variable data from either computed variables or CSV.

    This function first checks if the variable exists in the computed variables
    registry. If not found, it attempts to load it from the CSV file.

    Args:
        variable_name: Name or path of the variable
        start_time: Start time in milliseconds (default: 0)
        end_time: End time in milliseconds (default: -1 for end of data)

    Returns:
        DataInstance or ComputedVariable with the requested data, sliced to time range

    Raises:
        KeyError: If variable not found in either computed variables or CSV
        FileNotFoundError: If CSV file doesn't exist
        Exception: For other errors during data loading
    """
    # First, check computed variables
    computed_var = get_computed_variable(variable_name)
    if computed_var is not None:
        # Apply time slicing to computed variable
        return _slice_computed_variable(computed_var, start_time, end_time)

    # If not in computed variables, try CSV
    return _get_csv_variable(variable_name, start_time, end_time)


def _slice_computed_variable(var: ComputedVariable, start_time: int, end_time: int) -> ComputedVariable:
    """Slice a computed variable by timestamp."""
    if start_time == 0 and end_time == -1:
        return var

    # Handle negative end_time (means end of data)
    actual_end = var.timestamps[-1] if end_time < 0 else end_time

    # Find indices within time range
    mask = (var.timestamps >= start_time) & (var.timestamps <= actual_end)
    sliced_timestamps = var.timestamps[mask]
    sliced_values = var.values[mask]

    # Create new ComputedVariable with sliced data
    return ComputedVariable(
        name=var.name,
        label=var.label,
        timestamps=sliced_timestamps,
        values=sliced_values,
        metadata=var.metadata
    )


def _get_csv_variable(variable_name: str, start_time: int, end_time: int):
    """Get variable from CSV file using perda."""
    global _csv_cache, _csv_cache_path

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    # Use cached CSV data if available and path hasn't changed
    if _csv_cache is None or _csv_cache_path != CSV_PATH:
        parser = CSVParser()
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _csv_cache = parser(CSV_PATH)
                _csv_cache_path = CSV_PATH
        except Exception as e:
            raise Exception(f"Error parsing CSV: {e}")

    # Get the variable from cached data
    try:
        data_instance = _csv_cache[variable_name]
    except KeyError:
        raise KeyError(f"Variable '{variable_name}' not found in CSV or computed variables")

    # Apply time slicing
    if start_time == 0 and end_time == -1:
        return data_instance

    return get_data_slice_by_timestamp(data_instance, start_time, end_time)


def clear_csv_cache():
    """Clear the CSV cache to force re-parsing."""
    global _csv_cache, _csv_cache_path
    _csv_cache = None
    _csv_cache_path = None


def variable_exists(variable_name: str) -> bool:
    """
    Check if a variable exists in either computed variables or CSV.

    Args:
        variable_name: Name or path of the variable

    Returns:
        True if variable exists, False otherwise
    """
    # Check computed variables first
    if get_computed_variable(variable_name) is not None:
        return True

    # Check CSV
    try:
        _get_csv_variable(variable_name, 0, -1)
        return True
    except (KeyError, FileNotFoundError, Exception):
        return False
