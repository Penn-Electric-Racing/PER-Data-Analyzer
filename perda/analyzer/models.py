from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .joins import *


class DataInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp_np: NDArray
    value_np: NDArray
    label: str
    var_id: int

    @field_validator("timestamp_np")
    @classmethod
    def validate_timestamp(cls, v: Any) -> NDArray:
        """
        Validate timestamp array for DataInstance.

        Parameters
        ----------
        v : Any
            Input value to validate and convert to timestamp array

        Returns
        -------
        NDArray
            Validated 1D array of int64 timestamps

        Raises
        ------
        ValueError
            If array is not 1-dimensional, contains decreasing values, or has negative values
        """
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
        """
        Validate value array for DataInstance.

        Parameters
        ----------
        v : Any
            Input value to validate and convert to value array

        Returns
        -------
        NDArray
            Validated 1D array of float64 values

        Raises
        ------
        ValueError
            If array is not 1-dimensional
        """
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("value_np must be 1-dimensional array.")
        return v

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization validation for DataInstance.
        """
        if self.timestamp_np.shape[0] != self.value_np.shape[0]:
            raise ValueError("timestamp_np and value_np must have the same length.")

    def __len__(self) -> int:
        """
        Get number of data points in this DataInstance.
        """
        return self.timestamp_np.shape[0]

    def __add__(self, other: Union["DataInstance", int, float]) -> "DataInstance":
        """
        Add two DataInstances or add a scalar to a DataInstance.

        Parameters
        ----------
        other : Union[DataInstance, int, float]
            DataInstance or scalar value to add

        Returns
        -------
        DataInstance
            New DataInstance with added values
        """
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.add,
            )
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, var_id=self.var_id
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.add(self.value_np, other),
                label=self.label,
                var_id=self.var_id,
            )
        raise TypeError("add expects (DataInstance, DataInstance|scalar) in any order")

    def __sub__(self, other: Union["DataInstance", int, float]) -> "DataInstance":
        """
        Subtract a DataInstance or scalar from this DataInstance.

        Parameters
        ----------
        other : Union[DataInstance, int, float]
            DataInstance or scalar value to subtract

        Returns
        -------
        DataInstance
            New DataInstance with subtracted values
        """
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.subtract,
            )
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, var_id=self.var_id
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.subtract(self.value_np, other),
                label=self.label,
                var_id=self.var_id,
            )
        raise TypeError("sub expects (DataInstance, DataInstance|scalar)")

    def __mul__(self, other: Union["DataInstance", int, float]) -> "DataInstance":
        """
        Multiply two DataInstances or multiply a DataInstance by a scalar.

        Parameters
        ----------
        other : Union[DataInstance, int, float]
            DataInstance or scalar value to multiply

        Returns
        -------
        DataInstance
            New DataInstance with multiplied values
        """
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.multiply,
            )
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, var_id=self.var_id
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.multiply(self.value_np, other),
                label=self.label,
                var_id=self.var_id,
            )
        raise TypeError("mul expects (DataInstance, DataInstance|scalar)")

    def __truediv__(self, other: Union["DataInstance", int, float]) -> "DataInstance":
        """
        Divide this DataInstance by another DataInstance or scalar.

        Parameters
        ----------
        other : Union[DataInstance, int, float]
            DataInstance or scalar value to divide by

        Returns
        -------
        DataInstance
            New DataInstance with divided values
        """
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.true_divide,
            )
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, var_id=self.var_id
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.true_divide(self.value_np, other),
                label=self.label,
                var_id=self.var_id,
            )
        raise TypeError("div expects (DataInstance, DataInstance|scalar)")

    def __pow__(self, other: Union["DataInstance", int, float]) -> "DataInstance":
        """
        Raise this DataInstance to the power of another DataInstance or scalar.

        Parameters
        ----------
        other : Union[DataInstance, int, float]
            DataInstance or scalar exponent

        Returns
        -------
        DataInstance
            New DataInstance with values raised to the power
        """
        if isinstance(other, DataInstance):
            ts, val = combine(
                self.timestamp_np,
                self.value_np,
                other.timestamp_np,
                other.value_np,
                np.power,
            )
            return DataInstance(
                timestamp_np=ts, value_np=val, label=self.label, var_id=self.var_id
            )
        if np.isscalar(other):
            return DataInstance(
                timestamp_np=self.timestamp_np,
                value_np=np.power(self.value_np, other),
                label=self.label,
                var_id=self.var_id,
            )
        raise TypeError("pow_ expects (DataInstance, DataInstance|scalar)")

    def __neg__(self) -> "DataInstance":
        """
        Negate all values in this DataInstance.

        Returns
        -------
        DataInstance
            New DataInstance with negated values
        """
        return DataInstance(
            timestamp_np=self.timestamp_np,
            value_np=np.negative(self.value_np),
            label=self.label,
            var_id=self.var_id,
        )


class SingleRunData(BaseModel):
    """Pydantic model to store parsed CSV data with dictionary-like lookup."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core data storage
    data_instance_map: Dict[int, DataInstance] = Field(
        default_factory=dict, description="Mapping from variable ID to DataInstance"
    )
    var_name_map: Dict[int, str] = Field(
        default_factory=dict, description="Mapping from variable ID to variable name"
    )
    var_id_map: Dict[str, int] = Field(
        default_factory=dict, description="Mapping from variable name to variable ID"
    )

    # Metadata
    total_data_points: int = Field(
        description="Total number of data points across all variables"
    )
    data_start_time: int = Field(description="Start timestamp in milliseconds")
    data_end_time: int = Field(description="End timestamp in milliseconds")

    def __getitem__(
        self, input_var_id_name: Union[str, int, DataInstance]
    ) -> DataInstance:
        """
        Dictionary-like access to DataInstance by variable ID or variable name.

        Parameters
        ----------
        input_var_id_name : Union[str, int, DataInstance]
            Variable ID (int), variable name (str), or DataInstance to retrieve

        Returns
        -------
        DataInstance
            DataInstance corresponding to the input
        """
        # Dummy return for plotting function for convenience
        if isinstance(input_var_id_name, DataInstance):
            return input_var_id_name

        # If input is a variable ID
        if isinstance(input_var_id_name, int):
            if input_var_id_name not in self.data_instance_map:
                raise KeyError(f"Cannot find variable ID: {input_var_id_name}")
            var_id = input_var_id_name

        # If input is variable name
        elif isinstance(input_var_id_name, str):
            var_id = None
            for long_name in self.var_id_map:
                if name_matches(input_var_id_name, long_name):
                    var_id = self.var_id_map[long_name]
                    break
            if var_id is None:
                raise KeyError(f"Cannot find variable name: {input_var_id_name}")
        else:
            raise ValueError("Input must be a string, int, or DataInstance.")

        # Return DataInstance
        assert var_id is not None  # This should never be None due to the checks above
        return self.data_instance_map[var_id]

    def __contains__(self, input_var_id_name: Union[str, int]) -> bool:
        """
        Check if variable ID or variable name exists in the data.

        Parameters
        ----------
        input_var_id_name : Union[str, int]
            Variable ID or variable name to check

        Returns
        -------
        bool
            True if the variable ID or variable name exists in the data
        """
        try:
            self[input_var_id_name]
            return True
        except (KeyError, AttributeError):
            return False
