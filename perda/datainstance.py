from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DataInstance:
    # Make empty instances by default no input
    timestamp_np: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    value_np: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    label: str = ""

    # desired canonical dtypes
    TS_DTYPE = np.int64
    VAL_DTYPE = np.float64

    def __post_init__(self):
        # coerce to arrays in the dtypes you want
        self.timestamp_np = np.asarray(self.timestamp_np, dtype=self.TS_DTYPE)
        self.value_np = np.asarray(self.value_np, dtype=self.VAL_DTYPE)

        # validate the arrays
        if self.timestamp_np.ndim != 1 or self.value_np.ndim != 1:
            raise ValueError("timestamp_np and value_np must be 1-dimensional arrays.")
        if self.timestamp_np.shape[0] != self.value_np.shape[0]:
            raise ValueError("timestamp_np and value_np must have the same length.")
        if not np.all(np.diff(self.timestamp_np) >= 0):
            raise ValueError("timestamp_np cannot be decreasing.")
        if len(self.timestamp_np) and self.timestamp_np[0] < 0:
            raise ValueError("timestamp_np must be non-negative.")

    # ---------- editing (mutable) ----------
    def append(self, ts, v):
        """Append a single (ts, v). ts must be > last ts (or list empty)."""
        ts = int(ts)
        v = float(v)
        if len(self.timestamp_np) and ts < int(self.timestamp_np[-1]):
            raise ValueError("append requires ts >= last timestamp.")
        self.timestamp_np = np.append(self.timestamp_np, np.int64(ts))
        self.value_np = np.append(self.value_np, np.float64(v))

    def set_label(self, label: str):
        self.label = label

    # ---------- getters ----------

    def get_range(self, start_time: int = 0, end_time: int = -1) -> "DataInstance":
        """
        Get a new DataInstance with data in [start_time, end_time).
        If end_time == -1, means till end.
        """
        if end_time == -1:
            mask = self.timestamp_np >= start_time
        else:
            mask = (self.timestamp_np >= start_time) & (self.timestamp_np < end_time)
        return DataInstance(self.timestamp_np[mask], self.value_np[mask], self.label)

    # ---------- info ----------
    def __len__(self) -> int:
        return self.timestamp_np.shape[0]

    # ---------- numeric protocol ----------
    def __add__(self, other):
        from .datahelper import add

        return add(self, other)

    def __sub__(self, other):
        from .datahelper import sub

        return sub(self, other)

    def __mul__(self, other):
        from .datahelper import mul

        return mul(self, other)

    def __truediv__(self, other):
        from .datahelper import truediv

        return truediv(self, other)

    def __pow__(self, other):
        from .datahelper import pow

        return pow(self, other)

    # scalars on the right (e.g., d1 / 2, d1 ** 2)

    # # unary
    # def __neg__(self):
    #     return self

    # def __pos__(self):
    #     return self
