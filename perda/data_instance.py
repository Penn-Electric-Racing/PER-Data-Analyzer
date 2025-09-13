from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .helper import *


class DataInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp_np: NDArray
    value_np: NDArray
    label: str
    canid: int

    @field_validator("timestamp_np")
    @classmethod
    def validate_timestamp(cls, v: Any) -> NDArray:
        v = np.asarray(v, dtype=np.int64)
        if v.ndim != 1:
            raise ValueError("timestamp_np must be 1-dimensional array.")
        if not np.all(np.diff(v) >= 0):
            raise ValueError("timestamp_np cannot be decreasing.")
        if len(v) and v[0] < 0:
            raise ValueError("timestamp_np must be non-negative.")
        return v

    @field_validator("value_np")
    @classmethod
    def validate_value(cls, v: Any) -> NDArray:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("value_np must be 1-dimensional array.")
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.timestamp_np.shape[0] != self.value_np.shape[0]:
            raise ValueError("timestamp_np and value_np must have the same length.")

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
            timestamp_np=self.timestamp_np[mask],
            value_np=self.value_np[mask],
            label=self.label,
            canid=self.canid,
        )

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
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, canid=self.canid
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.add(self.value_np, other),
                label=self.label,
                canid=self.canid,
            )
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
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, canid=self.canid
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.subtract(self.value_np, other),
                label=self.label,
                canid=self.canid,
            )
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
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, canid=self.canid
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.multiply(self.value_np, other),
                label=self.label,
                canid=self.canid,
            )
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
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, canid=self.canid
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.true_divide(self.value_np, other),
                label=self.label,
                canid=self.canid,
            )
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
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, canid=self.canid
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.power(self.value_np, other),
                label=self.label,
                canid=self.canid,
            )
        raise TypeError("pow_ expects (DataInstance, DataInstance|scalar)")

    def __neg__(self):
        return DataInstance(
            timestamp_np=self.timestamp_np,
            value_np=np.negative(self.value_np),
            label=self.label,
            canid=self.canid,
        )
