import numpy as np
from numpy.typing import NDArray


def _find_contiguous_slices(var_ids: NDArray[np.int64]) -> dict[int, tuple[int, int]]:
    """Find contiguous slice boundary index pairs for a sorted 1D array.

    Avoids ``np.unique`` sorting overhead by performing a single-pass O(N) diff.

    Parameters
    ----------
    var_ids : NDArray[np.int64]
        Sorted 1D array of variable identifiers.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from each unique variable ID to ``(start, end)`` slice bounds.
    """
    if len(var_ids) == 0:
        return {}

    diff_mask = var_ids[:-1] != var_ids[1:]
    change_indices = np.flatnonzero(diff_mask) + 1

    start_indices = np.empty(len(change_indices) + 1, dtype=np.int64)
    start_indices[0] = 0
    start_indices[1:] = change_indices

    end_indices = np.append(start_indices[1:], len(var_ids))
    unique_ids = var_ids[start_indices]

    return {
        int(uid): (start, end)
        for uid, start, end in zip(unique_ids, start_indices, end_indices)
    }
