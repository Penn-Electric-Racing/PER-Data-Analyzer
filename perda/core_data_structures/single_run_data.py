from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..units import Timescale
from .data_instance import DataInstance
from .search_indexes import (
    VariableKeywordIndex,
    VariableSemanticIndex,
    variable_keyword_index_from_csv_mappings,
)
from .summarize import summarize_run
from .variable_patterns import is_pattern, match_pattern

VariableKey = str | int | DataInstance


class SingleRunData(BaseModel):
    """
    Pydantic model to store parsed CSV data with dictionary-like lookup.

    Notes
    -----
    All timestamps (``data_start_time``, ``data_end_time`` and every DataInstance's ``timestamp_np``) are
    raw integers in ``timestamp_unit`` (milliseconds, microseconds, etc.). Helpers that take a ``source_time_unit``
    or ``timestamp_unit`` argument require it; pass ``data.timestamp_unit``.

    Examples
    --------
    >>> data = aly.data
    >>> print(data)   # run summary
    >>> di = data["pcm.wheelSpeeds.frontRight"]
    >>> "pcm.wheelSpeeds.frontRight" in data
    >>> data["my.new.var"] = DataInstance(timestamp_np=ts, value_np=vals, label="My var", cpp_name="my.new.var")
    >>> data["my.existing.var"] = updated_di
    >>> vel_x, vel_y, vel_z = data["pcm.vnav.bodyVelocity.*"]   # wildcard, sorted by name
    >>> left, right = data["pcm.wheelSpeeds.frontLeft", "pcm.wheelSpeeds.frontRight"]
    >>> data["pcm.vnav.bodyVelocity.*"] = vel_x * 2, vel_y * 2, vel_z * 2
    >>> data["derived.a", "derived.b"] = a_di, b_di
    >>> first_minute = data.trim(data.data_start_time, data.data_start_time + 60_000)
    >>> before, after = data.split([split_timestamp])
    >>> from perda.units import to_seconds
    >>> duration_s = to_seconds(data.data_end_time - data.data_start_time, data.timestamp_unit)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core data storage
    id_to_instance: dict[int, DataInstance] = Field(
        description="Mapping from variable ID to DataInstance"
    )
    cpp_name_to_id: dict[str, int] = Field(
        description="Mapping from variable name to variable ID"
    )
    id_to_cpp_name: dict[int, str] = Field(
        description="Mapping from variable ID to variable name"
    )
    id_to_descript: dict[int, str] = Field(
        description="Mapping from variable ID to variable description"
    )

    # Metadata
    creation_time: datetime | None = Field(
        default=None,
        description="Timestamp when this run log was recorded",
    )
    total_data_points: int = Field(
        description="Total number of data points across all variables"
    )
    data_start_time: int = Field(description="Start timestamp in log timestamp unit")
    data_end_time: int = Field(description="End timestamp in log timestamp unit")
    timestamp_unit: Timescale = Field(
        default=Timescale.MS,
        description="Timestamp logging unit for this run (ms/us)",
    )
    concat_boundaries: list[int] = Field(
        default_factory=list,
        description="Timestamps where concatenated runs begin (post-shift)",
    )
    semantic_index: VariableSemanticIndex | None = Field(
        default=None,
        description="Vector index over variable descriptions. None when semantic search is disabled or unavailable",
    )
    keyword_index: VariableKeywordIndex = Field(
        default_factory=lambda: VariableKeywordIndex(
            row_to_var_id=[], normalized_text=[]
        ),
        description="Normalized name and description text backing keyword search",
    )

    @model_validator(mode="after")
    def _build_keyword_index(self) -> SingleRunData:
        if not self.keyword_index.row_to_var_id:
            self.keyword_index = variable_keyword_index_from_csv_mappings(
                self.id_to_cpp_name, self.id_to_descript
            )
        return self

    @overload
    def __getitem__(self, key: VariableKey) -> DataInstance: ...

    @overload
    def __getitem__(self, key: tuple[VariableKey, ...]) -> tuple[DataInstance, ...]: ...

    def __getitem__(
        self, key: VariableKey | tuple[VariableKey, ...]
    ) -> DataInstance | tuple[DataInstance, ...]:
        """
        Dictionary-like access by variable name, variable ID, or ``*`` wildcard pattern.

        A plain name, ID, or DataInstance returns one DataInstance. A wildcard pattern or a
        tuple of keys returns a tuple of DataInstances, with each pattern expanded to its
        matches in sorted name order. A DataInstance key is returned unchanged.
        """
        if isinstance(key, tuple):
            return tuple(di for sub_key in key for di in self._lookup(sub_key))
        if isinstance(key, str) and is_pattern(key):
            return self._lookup(key)
        return self._lookup(key)[0]

    def __setitem__(
        self,
        key: str | int | tuple[str | int, ...],
        value: DataInstance | Sequence[DataInstance],
    ) -> None:
        """
        Add or replace one or more variables using dictionary-style assignment.

        ``key`` may be a name, an existing ID, a ``*`` wildcard pattern, or a tuple of these. Patterns
        expand to existing names in sorted order. A tuple of names in key are paired with the
        DataInstances in ``value`` by position, so the counts must match.
        """
        sub_keys = key if isinstance(key, tuple) else (key,)
        cpp_names = [name for sub_key in sub_keys for name in self._resolve(sub_key)]
        instances = [value] if isinstance(value, DataInstance) else list(value)
        if len(cpp_names) != len(instances):
            raise ValueError(
                f"Cannot assign {len(instances)} DataInstance(s) to {len(cpp_names)} variable(s): {cpp_names}"
            )
        for cpp_name, di in zip(cpp_names, instances):
            self._assign(cpp_name, di)

    def __str__(self) -> str:
        """Summary of the run: logging unit, time range in seconds, and variable and point counts."""
        return summarize_run(
            self.timestamp_unit,
            self.data_start_time,
            self.data_end_time,
            len(self.id_to_instance),
            self.total_data_points,
        )

    def trim(
        self,
        ts_start: float | None = None,
        ts_end: float | None = None,
    ) -> SingleRunData:
        """
        Return a new SingleRunData with every variable trimmed to a timestamp range.

        Parameters
        ----------
        ts_start : float | None, optional
            Lower bound in raw timestamp units (inclusive). Default is None (no lower bound).
        ts_end : float | None, optional
            Upper bound in raw timestamp units (inclusive). Default is None (no upper bound).

        Returns
        -------
        SingleRunData
            New run data; the original is not changed.
        """
        trimmed = {
            var_id: di.trim(ts_start, ts_end)
            for var_id, di in self.id_to_instance.items()
        }
        start = self.data_start_time if ts_start is None else int(ts_start)
        end = self.data_end_time if ts_end is None else int(ts_end)
        return SingleRunData(
            id_to_instance=trimmed,
            cpp_name_to_id=dict(self.cpp_name_to_id),
            id_to_cpp_name=dict(self.id_to_cpp_name),
            id_to_descript=dict(self.id_to_descript),
            creation_time=self.creation_time,
            total_data_points=sum(len(di) for di in trimmed.values()),
            data_start_time=start,
            data_end_time=end,
            timestamp_unit=self.timestamp_unit,
            semantic_index=self.semantic_index,
            keyword_index=self.keyword_index,
        )

    def split(self, split_timestamps: Sequence[float]) -> list[SingleRunData]:
        """
        Split into consecutive runs at the given timestamps/

        Parameters
        ----------
        split_timestamps : Sequence[float]
            Split points, interpreted in timestamp units of the current run.
            Must be strictly increasing. A sample exactly on a split point goes to the later run.

        Returns
        -------
        list[SingleRunData]
        """
        if any(
            later < earlier
            for earlier, later in zip(split_timestamps, split_timestamps[1:])
        ):
            raise ValueError(
                f"split_timestamps must be in increasing order: {list(split_timestamps)}"
            )
        # Timestamps are integers: ">= split" is ">= ceil(split)" and "< split" is "<= ceil(split) - 1".
        first_kept = [math.ceil(split) for split in split_timestamps]
        starts: list[int | None] = [None, *first_kept]
        ends: list[int | None] = [*(timestamp - 1 for timestamp in first_kept), None]
        return [self.trim(start, end) for start, end in zip(starts, ends)]

    def __contains__(self, key: str | int) -> bool:
        """Check if a variable ID, variable name, or any match of a wildcard pattern exists."""
        try:
            self[key]
            return True
        except (KeyError, AttributeError):
            return False

    def _lookup(self, key: VariableKey) -> tuple[DataInstance, ...]:
        if isinstance(key, DataInstance):
            return (key,)
        names = self._resolve(key)
        missing = [name for name in names if name not in self.cpp_name_to_id]
        if missing:
            raise KeyError(f"Cannot find variable name: {missing[0]}")
        return tuple(self.id_to_instance[self.cpp_name_to_id[name]] for name in names)

    def _resolve(self, key: str | int) -> list[str]:
        if isinstance(key, int):
            if key not in self.id_to_cpp_name:
                raise KeyError(f"Cannot find variable ID: {key}")
            return [self.id_to_cpp_name[key]]
        if not isinstance(key, str):
            raise ValueError("Key must be a string, int, or DataInstance.")
        if not is_pattern(key):
            return [key]
        names = match_pattern(key, self.cpp_name_to_id)
        if not names:
            raise KeyError(f"No variable names match pattern: {key}")
        return names

    def _assign(self, cpp_name: str, di: DataInstance) -> None:
        if cpp_name in self.cpp_name_to_id:
            var_id = self.cpp_name_to_id[cpp_name]
            if di.var_id is not None and di.var_id != var_id:
                raise ValueError(
                    f"Cannot change var_id of '{cpp_name}' from {var_id} to {di.var_id}"
                )
            if di.cpp_name is not None and di.cpp_name != cpp_name:
                raise ValueError(
                    f"Cannot store DataInstance named '{di.cpp_name}' under '{cpp_name}'"
                )
            label = di.label or self.id_to_instance[var_id].label
            self.id_to_instance[var_id] = DataInstance(
                timestamp_np=di.timestamp_np,
                value_np=di.value_np,
                label=label,
                var_id=var_id,
                cpp_name=cpp_name,
            )
            self.id_to_descript[var_id] = label or ""
            return

        var_id = self._free_var_id(di.var_id)
        self.id_to_instance[var_id] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=di.value_np,
            label=di.label,
            var_id=var_id,
            cpp_name=cpp_name,
        )
        self.cpp_name_to_id[cpp_name] = var_id
        self.id_to_cpp_name[var_id] = cpp_name
        self.id_to_descript[var_id] = di.label or ""

    def _free_var_id(self, preferred: int | None) -> int:
        if preferred is not None and preferred not in self.id_to_instance:
            return preferred
        var_id = len(self.id_to_instance) + 1
        while var_id in self.id_to_instance:
            var_id += 1
        return var_id
