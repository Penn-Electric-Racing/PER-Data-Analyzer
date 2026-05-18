from ..core_data_structures.single_run_data import SingleRunData
from .analyzer import Analyzer


def from_single_run_data(single_run_data: SingleRunData) -> Analyzer:
    """
    Create an Analyzer instance from a SingleRunData object.

    Parameters
    ----------
    single_run_data : SingleRunData
        The SingleRunData object containing the data to analyze.

    Returns
    -------
    Analyzer
        An instance of the Analyzer class initialized with the provided data.
    """
    aly = Analyzer.__new__(Analyzer)
    aly.data = single_run_data
    return aly
