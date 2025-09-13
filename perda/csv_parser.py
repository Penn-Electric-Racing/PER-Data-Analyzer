from typing import Any, Dict, Optional, Union

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from .data_instance import DataInstance


class SingleRunData(BaseModel):
    """Pydantic model to store parsed CSV data with dictionary-like lookup."""

    # Core data storage
    tv_map: Dict[int, DataInstance] = {}  # canid -> DataInstance
    id_map: Dict[int, str] = {}  # canid -> name
    name_map: Dict[str, int] = {}  # name -> canid

    # Metadata
    total_data_points: int = 0
    data_start_time: Optional[int] = None
    data_end_time: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

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
                if CSVParser.name_matches(input_canid_name, long_name):
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


class CSVParser:
    """Callable CSV parser that returns SingleRunData model."""

    def __call__(self, file_path: str, bad_data_limit: int = 100) -> "SingleRunData":
        """
        Parse CSV file and return SingleRunData model.
        file_path: path of file we want to parse
        bad_data_limit: number of bad data before stopping (-1 for no limit)
        """
        # Initialize data containers
        tv_map = {}
        id_map = {}
        name_map = {}
        total_data_points = 0
        data_start_time = None
        data_end_time = None

        # Initialize bad data count
        bad_data = 0

        with open(file_path, "r") as log:
            # Skip and print first line (header)
            header = next(log)
            print(f"Header: {header}")
            # Start at second line for displaying error line
            line_num = 1
            for line in tqdm(log, desc="Reading CSV", unit=" lines", initial=1):
                line_num += 1
                # Stop read if too many bad lines
                if bad_data_limit > 0 and bad_data >= bad_data_limit:
                    raise Exception("Too many bad data lines encountered.")
                # Read canid and name before data
                if line.startswith("Value "):
                    canid_name = line[6:].strip().split(": ")
                    try:
                        canID = int(canid_name[1])
                        name = canid_name[0]
                        # Store everything in map
                        if canID in id_map:
                            print(
                                f"Warning: Duplicate CAN ID {canID} at line {line_num}. Overwriting previous name."
                            )
                        id_map[canID] = name
                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        bad_data += 1
                        continue
                # Read data lines
                else:
                    data = line.strip().split(",")
                    try:
                        id = int(data[1])
                        timestamp = int(data[0])
                        val = float(data[2])

                        # Record start time
                        if data_start_time is None:
                            data_start_time = timestamp

                        if id not in tv_map:
                            # Use can ID as key since there is somehow duplicate names sometimes :(
                            tv_map[id] = []
                        tv_map[id].append([timestamp, val])
                        total_data_points += 1

                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        bad_data += 1
                        continue

        # Convert lists to DataInstance
        for canid in tqdm(id_map, desc="Creating DataInstances"):
            name = id_map[canid]
            name_map[name] = canid
            if canid not in tv_map:
                tv_map[canid] = DataInstance(
                    timestamp_np=np.array([]),
                    value_np=np.array([]),
                    label=name,
                    canid=canid,
                )
            else:
                data_array = tv_map[canid]
                data_array = np.array(data_array)
                timestamps = data_array[:, 0]
                values = data_array[:, 1]
                tv_map[canid] = DataInstance(
                    timestamp_np=timestamps,
                    value_np=values,
                    label=name,
                    canid=canid,
                )

        # Record end time as last timestamp
        data_end_time = timestamp
        if bad_data != 0:
            print(f"Csv parsing complete with: {bad_data} bad lines.")
        else:
            print("Csv parsing complete.")

        # Create and return SingleRunData model
        return SingleRunData(
            tv_map=tv_map,
            id_map=id_map,
            name_map=name_map,
            total_data_points=total_data_points,
            data_start_time=data_start_time,
            data_end_time=data_end_time,
        )

    @staticmethod
    def name_matches(short_name, full_name):
        return f"({short_name})" in full_name or f"{short_name}" in full_name
