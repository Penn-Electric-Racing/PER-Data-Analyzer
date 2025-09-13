from .analyzer import analyzer
from .datainstance import DataInstance

__all__ = [
    "create",
    "DataInstance",  # Make DataInstance available for direct import
]


def create():
    return analyzer()
