# peralyzer/__init__.py
from .analyzer import analyzer  # Import Analyzer from core.py

__all__ = ["create"]  # Defines what gets imported with `from peralyzer import *`

def create():
    return analyzer()