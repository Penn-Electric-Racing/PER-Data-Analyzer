from pydantic import BaseModel
import numpy as np
from numpy.typing import NDArray


class ContiguousSlice(BaseModel):
    """Represents a contiguous segment of variable IDs in a sorted array."""

    var_id: int
    start: int
    end: int


class ContiguousSliceResult(BaseModel):
    """Mapping of variable IDs to contiguous slice boundaries."""

    slices: dict[int, ContiguousSlice]


def _find_contiguous_slices(var_ids: NDArray[np.int64]) -> ContiguousSliceResult:
    """Find contiguous slice boundary index pairs for a sorted 1D array.

    Avoids ``np.unique`` sorting overhead by performing a single-pass O(N) diff.

    Parameters
    ----------
    var_ids : NDArray[np.int64]
        Sorted 1D array of variable identifiers.

    Returns
    -------
    ContiguousSliceResult
        Pydantic model containing each unique variable ID mapped to its ``(start, end)`` bounds.
    """
    if len(var_ids) == 0:
        return ContiguousSliceResult(slices={})

    diff_mask = var_ids[:-1] != var_ids[1:]
    change_indices = np.flatnonzero(diff_mask) + 1

    start_indices = np.empty(len(change_indices) + 1, dtype=np.int64)
    start_indices[0] = 0
    start_indices[1:] = change_indices

    end_indices = np.append(start_indices[1:], len(var_ids))
    unique_ids = var_ids[start_indices]

    slices = {
        int(uid): ContiguousSlice(var_id=int(uid), start=int(start), end=int(end))
        for uid, start, end in zip(unique_ids, start_indices, end_indices)
    }
    return ContiguousSliceResult(slices=slices)
