import numpy as np

from .data_instance import DataInstance


def deduplicate(data: DataInstance) -> DataInstance:
    """
    Remove consecutive duplicate values from a DataInstance.

    Parameters
    ----------
    data : DataInstance

    Returns
    -------
    DataInstance

    Examples
    --------
    >>> gps_deduped = deduplicate(analyzer.data["gps.lat"])
    """
    src_v = data.value_np
    keep = np.concatenate(([True], src_v[1:] != src_v[:-1]))
    return DataInstance(
        timestamp_np=data.timestamp_np[keep],
        value_np=src_v[keep],
        label=data.label,
        var_id=data.var_id,
        cpp_name=data.cpp_name,
    )
