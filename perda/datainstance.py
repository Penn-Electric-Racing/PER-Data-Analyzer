from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class DataInstance:
    timestamp_np: NDArray[np.integer]
    val_np: NDArray[np.number]

    def __post_init__(self):
        if self.timestamp_np.ndim != 1 or self.val_np.ndim != 1:
            raise ValueError("timestamp_np and val_np must be 1-dimensional arrays.")
        if self.timestamp_np.shape[0] != self.val_np.shape[0]:
            raise ValueError("timestamp_np and val_np must have the same length.")
        if not np.issubdtype(self.timestamp_np.dtype, np.integer):
            raise TypeError("timestamp_np must be of integer type.")
        if not np.issubdtype(self.val_np.dtype, np.number):
            raise TypeError("val_np must be of numeric type.")
        if not np.all(np.diff(self.timestamp_np) > 0):
            raise ValueError("timestamp_np must be strictly increasing.")
        if not self.timestamp_np[0] >= 0:
            raise ValueError("timestamp_np must be non-negative.")

    # ---------- numeric protocol ----------
    def __add__(self, other):
        from helper import add
        return add(self, other)


    # scalars on the right (e.g., d1 / 2, d1 ** 2)

    
    # unary
    def __neg__(self):
        return self

    def __pos__(self):
        return self
