"""
CAN Variable Search using perda's pretty print functions.

Simple wrapper around perda's built-in search functionality.
"""

import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from perda.csv_parser import CSVParser
from perda.pretty_print_data import (
    pretty_print_single_run_variables,
    pretty_print_data_instance_info,
    pretty_print_single_run_info,
)


class CANSearch:
    """Simple CAN search using perda's functionality."""

    def __init__(self, csv_path: str = None):
        """
        Initialize with a CSV file path.

        Args:
            csv_path: Path to CSV file with CAN data
        """
        self.csv_path = csv_path
        self.data = None

        if csv_path and os.path.exists(csv_path):
            self._load_data(csv_path)

    def _load_data(self, csv_path: str):
        """Load CSV data using perda parser."""
        parser = CSVParser()
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.data = parser(csv_path)

    def search_variables(self, query: str, strict: bool = False) -> str:
        """
        Search for CAN variables.

        Args:
            query: Search terms (space-separated)
            strict: If True, ALL terms must match. If False, ANY term matches.

        Returns:
            Formatted string with search results
        """
        if not self.data:
            return "No data loaded. Please load a CSV file first."

        # Capture the output from perda's pretty print function
        output = StringIO()
        with redirect_stdout(output):
            pretty_print_single_run_variables(
                self.data,
                search=query if query else None,
                strict_search=strict,
                sort_by="name"
            )

        return output.getvalue()

    def get_variable_info(self, variable_path: str, time_unit: str = "s") -> str:
        """
        Get detailed information about a specific variable.

        Args:
            variable_path: Full path like "ams.pack.soc"
            time_unit: "s" or "ms"

        Returns:
            Formatted string with variable statistics
        """
        if not self.data:
            return "No data loaded. Please load a CSV file first."

        try:
            data_instance = self.data[variable_path]
        except KeyError:
            return f"Variable '{variable_path}' not found in dataset."

        # Capture the output from perda's pretty print function
        output = StringIO()
        with redirect_stdout(output):
            pretty_print_data_instance_info(data_instance, time_unit=time_unit)

        return output.getvalue()

    def get_dataset_info(self, time_unit: str = "s") -> str:
        """
        Get overall dataset information.

        Args:
            time_unit: "s" or "ms"

        Returns:
            Formatted string with dataset info
        """
        if not self.data:
            return "No data loaded. Please load a CSV file first."

        # Capture the output from perda's pretty print function
        output = StringIO()
        with redirect_stdout(output):
            pretty_print_single_run_info(self.data, time_unit=time_unit)

        return output.getvalue()

    def list_all_variables(self) -> str:
        """
        List all variables in the dataset.

        Returns:
            Formatted string with all variables
        """
        return self.search_variables(query=None, strict=False)
