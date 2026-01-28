"""SingleRunData model for storing parsed CSV data."""

from typing import Dict, Union

from pydantic import BaseModel, ConfigDict, Field

from .data_instance import DataInstance
from .joins import name_matches


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
