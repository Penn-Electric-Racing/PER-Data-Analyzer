"""SingleRunData model for storing parsed CSV data."""

from typing import Dict, Union

from pydantic import BaseModel, ConfigDict, Field

from .data_instance import DataInstance
from .joins import name_matches


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
