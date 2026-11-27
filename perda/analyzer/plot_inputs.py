from collections.abc import Sequence

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData

VariableInput = str | int | DataInstance
PlotVariables = VariableInput | Sequence[VariableInput | Sequence[VariableInput | None] | None]
AxisTitles = str | Sequence[str | None] | None


def group_variables_by_subplot(variables: PlotVariables) -> list[PlotVariables | None]:
    """
    Split plot variables into one group per subplot, based on their nesting.

    Parameters
    ----------
    variables : PlotVariables
        A single variable or a flat list of variables (one subplot), or a list where at
        least one entry is an inner list or ``None`` (one subplot per entry).

    Returns
    -------
    list[PlotVariables | None]
        One variable group per subplot, top to bottom.

    Notes
    -----
    In the nested form, entries that are not lists are treated as lists of length one,
    and ``None`` entries give a subplot with nothing on that axis, e.g.
    ``[None, "steering"]`` puts ``"steering"`` on the second subplot only.
    """
    if isinstance(variables, (list, tuple)) and any(
        item is None or isinstance(item, (list, tuple)) for item in variables
    ):
        return list(variables)
    return [variables]


def lookup_subplot_data_instances(
    data: SingleRunData,
    subplot_variables: PlotVariables | None,
    ts_start_raw: float | None,
    ts_end_raw: float | None,
) -> list[DataInstance]:
    """
    Look up every DataInstance for one subplot axis, trimmed to a time window.

    Parameters
    ----------
    data : SingleRunData
        Run data to look variables up in.
    subplot_variables : PlotVariables | None
        Variables for one subplot axis, as returned by :func:`group_variables_by_subplot`.
        Nested lists are flattened, wildcard patterns are expanded, and ``None`` entries are
        skipped.
    ts_start_raw : float | None
        Start of the time window in raw timestamp units, or None for no lower bound.
    ts_end_raw : float | None
        End of the time window in raw timestamp units, or None for no upper bound.

    Returns
    -------
    list[DataInstance]
        DataInstances in the order given, empty if there are none.
    """
    if subplot_variables is None:
        return []
    if not isinstance(subplot_variables, (str, int, DataInstance)):
        return [
            di
            for item in subplot_variables
            for di in lookup_subplot_data_instances(data, item, ts_start_raw, ts_end_raw)
        ]
    matches = data[(subplot_variables,)]
    if ts_start_raw is None and ts_end_raw is None:
        return list(matches)
    return [di.trim(ts_start_raw, ts_end_raw) for di in matches]


def axis_titles_per_subplot(titles: AxisTitles, n_subplots: int) -> list[str | None] | None:
    """Repeat a single axis title across every subplot; lists and None pass through."""
    if isinstance(titles, str):
        return [titles for _ in range(n_subplots)]
    return None if titles is None else list(titles)
