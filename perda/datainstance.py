from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .helper import *


@dataclass(slots=True)
class DataInstance:
    # Make empty instances by default no input
    timestamp_np: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    value_np: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    label: str = ""
    canid: int = -1

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

    def set_canid(self, canid: int):
        self.canid = canid

    # ---------- getters ----------

    def get_range(self, start_time: int = 0, end_time: int = -1):
        """
        Get a new DataInstance with data in [start_time, end_time).
        If end_time == -1, means till end.
        """
        if end_time < 0:
            mask = self.timestamp_np >= start_time
        else:
            mask = (self.timestamp_np >= start_time) & (self.timestamp_np < end_time)
        return DataInstance(
            self.timestamp_np[mask], self.value_np[mask], self.label, self.canid
        )

    def get_min(self):
        """Get min timestamp and value."""
        if len(self.timestamp_np) == 0:
            return None, None
        return int(self.timestamp_np.min()), float(self.value_np.min())

    def get_max(self):
        """Get max timestamp and value."""
        if len(self.timestamp_np) == 0:
            return None, None
        return int(self.timestamp_np.max()), float(self.value_np.max())

    def get_integral(self, time_unit: str = "ms"):
        """Get integral of the value over time using trapezoidal rule."""
        if len(self.timestamp_np) < 2:
            return 0.0
        ts = self.timestamp_np.astype(np.float64)
        # Convert timestamps from ms to s for integration
        if time_unit == "s":
            ts /= 1e3
        integral = np.trapz(self.value_np, ts)
        return float(integral)

    def get_average(self):
        """Get average using integral over time."""
        if len(self.timestamp_np) < 2:
            return 0.0
        total_time = self.timestamp_np[-1] - self.timestamp_np[0]
        if total_time == 0:
            return 0.0
        integral = self.get_integral()
        average = integral / total_time
        return float(average)

    # ---------- info ----------
    def __len__(self):
        return self.timestamp_np.shape[0]

    def print_info(self, time_unit: str = "s"):
        print(f"DataInstance for | {self.label} | canid={self.canid}")
        if len(self) > 0:
            min_ts, min_val = self.get_min()
            max_ts, max_val = self.get_max()
            first_ts = self.timestamp_np[0]
            last_ts = self.timestamp_np[-1]
            if time_unit == "s":
                first_ts = float(first_ts) / 1e3
                last_ts = float(last_ts) / 1e3
                min_ts = float(min_ts) / 1e3
                max_ts = float(max_ts) / 1e3
            avg_val = self.get_average()
            integral = self.get_integral(time_unit=time_unit)
            # Set width for alignment
            w = 10
            print(f"  Data points:  {len(self):>{w}}")
            print(
                f"  Time range:   {first_ts:>{w}.4f} to {last_ts:>{w}.4f} ({time_unit})"
            )
            print(
                f"  Min value:    {min_val:>{w}.4f} at {min_ts:>{w}.4f} ({time_unit})"
            )
            print(
                f"  Max value:    {max_val:>{w}.4f} at {max_ts:>{w}.4f} ({time_unit})"
            )
            print(f"  Integral:     {integral:>{w}.4f}")
            print(f"  Average value:{avg_val:>{w}.4f}")
        else:
            print("  Empty DataInstance.")

    # ---------- numeric protocol ----------
    def __add__(self, other):
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.add,
            )
            return DataInstance(ts, val)
        if np.isscalar(other):
            return DataInstance(self.timestamp_np, np.add(self.value_np, other))
        raise TypeError("add expects (DataInstance, DataInstance|scalar) in any order")

    def __sub__(self, other):
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.subtract,
            )
            return DataInstance(ts, val)
        if np.isscalar(other):
            return DataInstance(self.timestamp_np, np.subtract(self.value_np, other))
        raise TypeError("sub expects (DataInstance, DataInstance|scalar)")

    def __mul__(self, other):
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.multiply,
            )
            return DataInstance(ts, val)
        if np.isscalar(other):
            return DataInstance(self.timestamp_np, np.multiply(self.value_np, other))
        raise TypeError("mul expects (DataInstance, DataInstance|scalar)")

    def __truediv__(self, other):
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.true_divide,
            )
            return DataInstance(ts, val)
        if np.isscalar(other):
            return DataInstance(self.timestamp_np, np.true_divide(self.value_np, other))
        raise TypeError("div expects (DataInstance, DataInstance|scalar)")

    def __pow__(self, other):
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.power,
            )
            return DataInstance(ts, val)
        if np.isscalar(other):
            return DataInstance(self.timestamp_np, np.power(self.value_np, other))
        raise TypeError("pow_ expects (DataInstance, DataInstance|scalar)")

    def __neg__(self):
        return DataInstance(self.timestamp_np, np.negative(self.value_np))
