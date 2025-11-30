"""Computed variables registry for user-generated data.

This module provides a global registry for storing computed variables created
through arithmetic operations or other transformations. These variables can be
used in graphs, statistics, and further computations just like CSV variables.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ComputedVariable:
    """Represents a computed variable created from operations on other variables."""

    name: str
    label: str
    timestamps: np.ndarray
    values: np.ndarray
    metadata: dict

    def __len__(self):
        """Return number of data points."""
        return len(self.timestamps)

    @property
    def timestamp_np(self):
        """Alias for compatibility with perda DataInstance."""
        return self.timestamps

    @property
    def value_np(self):
        """Alias for compatibility with perda DataInstance."""
        return self.values


class ComputedVariablesRegistry:
    """Global registry for user-generated computed variables."""

    def __init__(self):
        self._variables: Dict[str, ComputedVariable] = {}
        self._counter = 0

    def register(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        label: str,
        metadata: Optional[dict] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Register a new computed variable.

        Args:
            timestamps: Array of timestamps in milliseconds
            values: Array of values
            label: Human-readable label for the variable
            metadata: Optional metadata dictionary
            name: Optional custom name (auto-generated if not provided)

        Returns:
            Name of the registered variable
        """
        if name is None:
            name = f"computed_{self._counter}"
            self._counter += 1

        if metadata is None:
            metadata = {}

        computed_var = ComputedVariable(
            name=name,
            label=label,
            timestamps=timestamps,
            values=values,
            metadata=metadata
        )

        self._variables[name] = computed_var
        return name

    def get(self, name: str) -> Optional[ComputedVariable]:
        """Get a computed variable by name."""
        return self._variables.get(name)

    def exists(self, name: str) -> bool:
        """Check if a computed variable exists."""
        return name in self._variables

    def list_all(self) -> list[str]:
        """List all registered computed variable names."""
        return list(self._variables.keys())

    def clear(self):
        """Clear all computed variables."""
        self._variables.clear()
        self._counter = 0

    def get_info(self, name: str) -> Optional[dict]:
        """Get information about a computed variable."""
        var = self.get(name)
        if var is None:
            return None

        return {
            "name": var.name,
            "label": var.label,
            "num_points": len(var),
            "time_range": {
                "start_ms": float(var.timestamps[0]) if len(var) > 0 else None,
                "end_ms": float(var.timestamps[-1]) if len(var) > 0 else None,
            },
            "value_range": {
                "min": float(np.min(var.values)) if len(var) > 0 else None,
                "max": float(np.max(var.values)) if len(var) > 0 else None,
            },
            "metadata": var.metadata
        }


# Global registry instance
_registry = ComputedVariablesRegistry()


def get_registry() -> ComputedVariablesRegistry:
    """Get the global computed variables registry."""
    return _registry


def register_computed_variable(
    timestamps: np.ndarray,
    values: np.ndarray,
    label: str,
    metadata: Optional[dict] = None,
    name: Optional[str] = None
) -> str:
    """
    Convenience function to register a computed variable.

    Returns:
        Name of the registered variable
    """
    return _registry.register(timestamps, values, label, metadata, name)


def get_computed_variable(name: str) -> Optional[ComputedVariable]:
    """Convenience function to get a computed variable."""
    return _registry.get(name)
