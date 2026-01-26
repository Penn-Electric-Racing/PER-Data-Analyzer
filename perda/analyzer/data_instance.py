from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .joins import inner_join, left_join, outer_join


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
        Validate that timestamp array is 1-dimensional, positive, and strictly increasing.
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
        Validate that value array is 1-dimensional
        """
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("value_np must be 1-dimensional array.")
        return v

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization validation that timestamp and value arrays have the same length.
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
        """
        if isinstance(other, DataInstance):
            return apply_ufunc_outer_join(self, other, np.add)
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
        """
        if isinstance(other, DataInstance):
            return apply_ufunc_outer_join(self, other, np.subtract)
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
        """
        if isinstance(other, DataInstance):
            return apply_ufunc_outer_join(self, other, np.multiply)
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
        """
        if isinstance(other, DataInstance):
            return apply_ufunc_outer_join(self, other, np.true_divide)
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
        """
        if isinstance(other, DataInstance):
            return apply_ufunc_outer_join(self, other, np.power)
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
        """
        return DataInstance(
            timestamp_np=self.timestamp_np,
            value_np=np.negative(self.value_np),
            label=self.label,
            var_id=self.var_id,
        )


def left_join_data_instances(
    left: DataInstance,
    right: DataInstance,
) -> Tuple[DataInstance, DataInstance]:
    """
    Left join two DataInstances: keep all left timestamps, interpolate right values.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance (all timestamps are kept)
    right : DataInstance
        Right DataInstance (values are matched/interpolated to left)

    Returns
    -------
    left_result : DataInstance
        Left DataInstance with aligned timestamps
    right_result : DataInstance
        Right DataInstance with values interpolated to left timestamps
    """
    ts, left_val, right_val = left_join(
        left.timestamp_np, left.value_np, right.timestamp_np, right.value_np
    )

    left_result = DataInstance(
        timestamp_np=ts, value_np=left_val, label=left.label, var_id=left.var_id
    )
    right_result = DataInstance(
        timestamp_np=ts, value_np=right_val, label=right.label, var_id=right.var_id
    )

    return left_result, right_result


def outer_join_data_instances(
    left: DataInstance,
    right: DataInstance,
    *,
    drop_nan: bool = True,
    fill: float = 0.0,
) -> Tuple[DataInstance, DataInstance]:
    """
    Outer join two DataInstances: union of timestamps with interpolation.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance
    right : DataInstance
        Right DataInstance
    drop_nan : bool, optional
        If True, drop rows where either series has NaN after interpolation.
        Default is True.
    fill : float, optional
        Fill value for NaNs when drop_nan is False. Default is 0.0.

    Returns
    -------
    left_result : DataInstance
        Left DataInstance with values interpolated to union timestamps
    right_result : DataInstance
        Right DataInstance with values interpolated to union timestamps
    """
    ts, left_val, right_val = outer_join(
        left.timestamp_np,
        left.value_np,
        right.timestamp_np,
        right.value_np,
        drop_nan=drop_nan,
        fill=fill,
    )

    left_result = DataInstance(
        timestamp_np=ts, value_np=left_val, label=left.label, var_id=left.var_id
    )
    right_result = DataInstance(
        timestamp_np=ts, value_np=right_val, label=right.label, var_id=right.var_id
    )

    return left_result, right_result


def inner_join_data_instances(
    left: DataInstance,
    right: DataInstance,
    *,
    tolerance: float,
) -> Tuple[DataInstance, DataInstance]:
    """
    Inner join two DataInstances: keep only left timestamps with matching right timestamps.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance
    right : DataInstance
        Right DataInstance
    tolerance : float
        Maximum allowed distance between left and right timestamps for a match.
        Timestamps with distance > tolerance are dropped.

    Returns
    -------
    left_result : DataInstance
        Left DataInstance with only matched timestamps
    right_result : DataInstance
        Right DataInstance with only matched timestamps
    """
    ts, left_val, right_val = inner_join(
        left.timestamp_np,
        left.value_np,
        right.timestamp_np,
        right.value_np,
        tolerance=tolerance,
    )

    left_result = DataInstance(
        timestamp_np=ts, value_np=left_val, label=left.label, var_id=left.var_id
    )
    right_result = DataInstance(
        timestamp_np=ts, value_np=right_val, label=right.label, var_id=right.var_id
    )

    return left_result, right_result


def apply_ufunc_left_join(
    left: DataInstance,
    right: DataInstance,
    ufunc: Callable,
) -> DataInstance:
    """
    Apply a binary operation to two DataInstances using left join.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance (all timestamps are kept)
    right : DataInstance
        Right DataInstance (values interpolated to left)
    ufunc : Callable
        NumPy universal function to apply (e.g., np.add, np.subtract)

    Returns
    -------
    DataInstance
        New DataInstance with combined values
    """
    left_aligned, right_aligned = left_join_data_instances(left, right)
    result_val = ufunc(left_aligned.value_np, right_aligned.value_np)

    return DataInstance(
        timestamp_np=left_aligned.timestamp_np,
        value_np=result_val,
        label=left.label,
        var_id=left.var_id,
    )


def apply_ufunc_outer_join(
    left: DataInstance,
    right: DataInstance,
    ufunc: Callable,
    *,
    drop_nan: bool = True,
    fill: float = 0.0,
) -> DataInstance:
    """
    Apply a binary operation to two DataInstances using outer join.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance
    right : DataInstance
        Right DataInstance
    ufunc : Callable
        NumPy universal function to apply (e.g., np.add, np.subtract)
    drop_nan : bool, optional
        If True, drop rows where either series has NaN after interpolation.
        Default is True.
    fill : float, optional
        Fill value for NaNs when drop_nan is False. Default is 0.0.

    Returns
    -------
    DataInstance
        New DataInstance with combined values
    """
    left_aligned, right_aligned = outer_join_data_instances(
        left, right, drop_nan=drop_nan, fill=fill
    )
    result_val = ufunc(left_aligned.value_np, right_aligned.value_np)

    return DataInstance(
        timestamp_np=left_aligned.timestamp_np,
        value_np=result_val,
        label=left.label,
        var_id=left.var_id,
    )


def apply_ufunc_inner_join(
    left: DataInstance,
    right: DataInstance,
    ufunc: Callable,
    *,
    tolerance: float,
) -> DataInstance:
    """
    Apply a binary operation to two DataInstances using inner join.

    Parameters
    ----------
    left : DataInstance
        Left DataInstance
    right : DataInstance
        Right DataInstance
    ufunc : Callable
        NumPy universal function to apply (e.g., np.add, np.subtract)
    tolerance : float
        Maximum allowed distance between left and right timestamps for a match.

    Returns
    -------
    DataInstance
        New DataInstance with combined values
    """
    left_aligned, right_aligned = inner_join_data_instances(
        left, right, tolerance=tolerance
    )
    result_val = ufunc(left_aligned.value_np, right_aligned.value_np)

    return DataInstance(
        timestamp_np=left_aligned.timestamp_np,
        value_np=result_val,
        label=left.label,
        var_id=left.var_id,
    )
