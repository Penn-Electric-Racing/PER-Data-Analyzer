# peralyzer/__init__.py
from .analyzer import analyzer  # Import Analyzer from core.py

__all__ = ["analyzer", "init", "readCsv", "plot"]  # Defines what gets imported with `from peralyzer import *`

# Create a global instance of `analyzer`
analyzer_instance = None

def init():
    global analyzer_instance 
    analyzer_instance = analyzer()

def read_csv(csvPath):
    if analyzer_instance is None:
        raise RuntimeError("Call init() before using readCsv()")
    analyzer_instance.read_csv(csvPath)

def plot(variables, same_graph=False):
    if analyzer_instance is None:
        raise RuntimeError("Call init() before using plot()")
    analyzer_instance.plot(variables, same_graph)