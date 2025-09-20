from typing import Any, Dict, Union

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


class SingleRunData(BaseModel):
    """Pydantic model to store parsed CSV data with dictionary-like lookup."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core data storage
    tv_map: Dict[int, DataInstance] = {}  # canid -> DataInstance
    id_map: Dict[int, str] = {}  # canid -> name
    name_map: Dict[str, int] = {}  # name -> canid

    # Metadata
    total_data_points: int
    data_start_time: int
    data_end_time: int

    def __getitem__(
        self, input_canid_name: Union[str, int, DataInstance]
    ) -> DataInstance:
        """Dictionary-like access to DataInstance by CAN ID or variable name."""
        # Dummy return for plotting function for convenience
        if isinstance(input_canid_name, DataInstance):
            return input_canid_name

        # If input is a CAN ID
        if isinstance(input_canid_name, int):
            if input_canid_name not in self.tv_map:
                raise KeyError(f"Cannot find CAN ID: {input_canid_name}")
            canid = input_canid_name

        # If input is CAN variable name
        elif isinstance(input_canid_name, str):
            canid = None
            for long_name in self.name_map:
                if name_matches(input_canid_name, long_name):
                    canid = self.name_map[long_name]
                    break
            if canid is None:
                raise KeyError(f"Cannot find CAN name: {input_canid_name}")
        else:
            raise ValueError("Input must be a string, int, or DataInstance.")

        # Return DataInstance
        return self.tv_map[canid]

    def __contains__(self, input_canid_name: Union[str, int]) -> bool:
        """Check if CAN ID or variable name exists in the data."""
        try:
            self[input_canid_name]
            return True
        except (KeyError, AttributeError):
            return False
