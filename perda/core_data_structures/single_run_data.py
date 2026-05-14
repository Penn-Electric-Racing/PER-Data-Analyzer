from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from ..units import Timescale
from .data_instance import DataInstance


class SingleRunData(BaseModel):
    """Pydantic model to store parsed CSV data with dictionary-like lookup."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core data storage
    id_to_instance: Dict[int, DataInstance] = Field(
        description="Mapping from variable ID to DataInstance"
    )
    cpp_name_to_id: Dict[str, int] = Field(
        description="Mapping from variable name to variable ID"
    )
    id_to_cpp_name: Dict[int, str] = Field(
        description="Mapping from variable ID to variable name"
    )
    id_to_descript: Dict[int, str] = Field(
        description="Mapping from variable ID to variable description"
    )

    # Metadata
    total_data_points: int = Field(
        description="Total number of data points across all variables"
    )
    data_start_time: int = Field(description="Start timestamp in log timestamp unit")
    data_end_time: int = Field(description="End timestamp in log timestamp unit")
    timestamp_unit: Timescale = Field(
        default=Timescale.MS,
        description="Timestamp logging unit for this run (ms/us)",
    )
    concat_boundaries: List[int] = Field(
        default_factory=list,
        description="Timestamps where concatenated runs begin (post-shift)",
    )

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
            if input_var_id_name not in self.id_to_instance:
                raise KeyError(f"Cannot find variable ID: {input_var_id_name}")
            return self.id_to_instance[input_var_id_name]

        # If input is variable name
        elif isinstance(input_var_id_name, str):
            if input_var_id_name not in self.cpp_name_to_id:
                raise KeyError(f"Cannot find variable name: {input_var_id_name}")
            return self.id_to_instance[self.cpp_name_to_id[input_var_id_name]]

        else:
            raise ValueError("Input must be a string, int, or DataInstance.")

    def __setitem__(self, cpp_name: str, di: DataInstance) -> None:
        """
        Add or replace a variable using dictionary-style assignment.

        Dispatches to ``replace`` if ``cpp_name`` already exists, else ``add``.

        Parameters
        ----------
        cpp_name : str
            C++ variable name to insert or overwrite.
        di : DataInstance
            DataInstance to store. Must have non-None ``label`` and ``cpp_name``.

        Examples
        --------
        >>> data["my.new.var"] = DataInstance(timestamp_np=ts, value_np=vals, label="My var", cpp_name="my.new.var")
        >>> data["my.existing.var"] = updated_di
        """
        if cpp_name in self:
            self.replace(cpp_name, di)
        else:
            self.add(cpp_name, di)

    def add(self, cpp_name: str, di: DataInstance) -> None:
        """
        Insert a new derived DataInstance using a synthetic negative ID.

        Parameters
        ----------
        cpp_name : str
            C++ variable name key for the new variable.
        di : DataInstance
            DataInstance to insert.
        """
        if cpp_name in self:
            raise KeyError(f"'{cpp_name}' already exists; use replace() to overwrite.")

        if di.cpp_name != cpp_name:
            print(f"Warning: replacing DataInstance.cpp_name with {cpp_name}")

        synthetic_id = -(len(self.id_to_instance) + 1)
        if di.var_id != synthetic_id:
            print(f"Warning: replacing DataInstance.var_id with {synthetic_id}")

        stored = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=di.value_np,
            label=di.label,
            var_id=synthetic_id,
            cpp_name=cpp_name,
        )
        self.id_to_instance[synthetic_id] = stored
        self.cpp_name_to_id[cpp_name] = synthetic_id
        self.id_to_cpp_name[synthetic_id] = cpp_name
        self.id_to_descript[synthetic_id] = di.label or ""

    def replace(self, cpp_name: str, di: DataInstance) -> None:
        """
        Overwrite the values of an existing variable in-place.

        Parameters
        ----------
        cpp_name : str
            C++ variable name of the variable to replace.
        di : DataInstance
            DataInstance whose ``value_np`` that replaces the stored one.
        """
        if cpp_name not in self:
            raise KeyError(
                f"'{cpp_name}' not found; use add() to insert a new variable."
            )

        var_id = self.cpp_name_to_id[cpp_name]
        old = self.id_to_instance[var_id]

        if di.cpp_name != cpp_name:
            print(f"Warning: retaining old DataInstance.cpp_name {cpp_name}")
        if di.var_id != var_id:
            print(f"Warning: retaining old DataInstance.var_id {var_id}")
        if di.label != old.label:
            print(f"Warning: retaining old DataInstance.label {old.label}")

        self.id_to_instance[var_id] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=di.value_np,
            label=di.label,
            var_id=old.var_id,
            cpp_name=old.cpp_name,
        )

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
