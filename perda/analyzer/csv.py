import re
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale


def parse_header_creation_time(header_line: str) -> datetime | None:
    """
    Extract the recording date from a log file's first line.

    Parameters
    ----------
    header_line : str
        First line of the log file, e.g. ``"PER Log: Thu Jun 11 17:06:37 2026 v2.0"``.

    Returns
    -------
    datetime | None
        Parsed recording date, or None if the header carries no recognizable date.
    """
    match = re.search(r"PER Log:\s*(.*?)(?:\s+v\d+\.\d+)?$", header_line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1).strip(), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None


def _find_start_end_indices_for_each_unique_value(
    var_ids: NDArray[np.int64],
) -> dict[int, tuple[int, int]]:
    """
    Find contiguous slice boundary index pairs for a sorted 1D array.

    Parameters
    ----------
    var_ids : NDArray[np.int64]
        Sorted 1D array of variable identifiers.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from each unique variable ID to its ``(start, end)`` slice bounds.
    """
    if len(var_ids) == 0:
        return {}

    diff_mask = var_ids[:-1] != var_ids[1:]
    change_indices = np.flatnonzero(diff_mask) + 1

    start_indices = np.empty(len(change_indices) + 1, dtype=np.int64)
    start_indices[0] = 0
    start_indices[1:] = change_indices

    end_indices = np.append(start_indices[1:], len(var_ids))
    unique_ids = var_ids[start_indices]

    return {
        int(uid): (int(start), int(end))
        for uid, start, end in zip(unique_ids, start_indices, end_indices)
    }


def parse_csv(
    file_path: str,
    ts_offset: int = 0,
    parsing_errors_limit: int = 100,
    verbose: int = 1,
) -> SingleRunData:
    """
    Parse CSV file and return SingleRunData model.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to parse
    parsing_errors_limit : int, optional
        Maximum number of parsing errors before stopping. -1 for no limit. Default is 100
    parse_unit : Timescale | str | None, optional
        Logging timestamp unit. If None, auto-detects using header suffix "v2.0" (us) or defaults to ms.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 for basic output, 2 for detailed output. Default is 1.

    Returns
    -------
    SingleRunData
        Parsed data structure containing all variables
    """
    # Maps variable ID to variable name
    id_to_cpp_name: dict[int, str] = {}
    id_to_descript: dict[int, str] = {}

    with open(file_path, "r") as f:
        # Parse and print first line (header)
        header_line = f.readline()
        parse_unit = (
            Timescale.US if header_line.rstrip().endswith("v2.0") else Timescale.MS
        )

        creation_time = parse_header_creation_time(header_line)

        if verbose >= 1:
            print(f"Header: {header_line.rstrip()}")
            print(f"Timestamp unit: {parse_unit.value}")
            if creation_time:
                print(f"Log recorded on: {creation_time}")

        # Block 1: Variable ID/Name pairs
        if verbose >= 2:
            pbar = tqdm(desc="Reading variable ID mappings", unit=" lines", initial=2)
        skip_rows = 1  # header line
        line = f.readline()
        while line and line.startswith("Value "):
            if verbose >= 2:
                pbar.update(1)
            skip_rows += 1

            # Remove "Value " prefix, separate into variable name and ID
            identifier = line[6:].strip().split(": ")

            try:
                var_id = int(identifier[1])
                name_part = identifier[0]

                # Check format: Value Desc (cpp.name): id | Value cpp.name: id
                if "(" in name_part and ")" in name_part:
                    open_idx = name_part.rfind("(")
                    close_idx = name_part.rfind(")")
                    if open_idx < close_idx:
                        cpp_name = name_part[open_idx + 1 : close_idx].strip()
                        descript = name_part[:open_idx].strip()
                    else:
                        cpp_name = name_part.strip()
                        descript = ""
                else:
                    cpp_name = name_part.strip()
                    descript = ""
                if not cpp_name:
                    raise ValueError(f"Empty cpp_name in mapping line: {line.strip()}")

                # Store variable ID to name mapping
                if var_id in id_to_cpp_name:
                    if verbose >= 1:
                        print(
                            f"Warning: Duplicate variable ID {var_id} at line {pbar.n}. Overwriting previous name."
                        )
                id_to_cpp_name[var_id] = cpp_name
                id_to_descript[var_id] = descript

            except Exception as e:
                if verbose >= 1:
                    print(f"Error parsing variable ID/Name pair at line {pbar.n}: {e}")

            line = f.readline()
        if verbose >= 2:
            pbar.close()

    # Block 2: Read data with Polars, Block 3: Sort — all in one step
    if verbose >= 1:
        print("Reading and sorting data...")
    df = pl.read_csv(
        file_path,
        skip_rows=skip_rows,
        has_header=False,
        new_columns=["timestamp", "var_id", "value"],
        schema={"column_1": pl.Int64, "column_2": pl.Int32, "column_3": pl.Float64},
        ignore_errors=True,
        glob=False,
    )

    parsing_errors = len(
        df.filter(
            df["timestamp"].is_null() | df["var_id"].is_null() | df["value"].is_null()
        )
    )
    if parsing_errors_limit > 0 and parsing_errors >= parsing_errors_limit:
        raise Exception("Too many data parsing errors encountered.")

    df = (
        df.drop_nulls()
        .with_columns((pl.col("timestamp") + ts_offset).alias("timestamp"))
        .sort(["var_id", "timestamp"])
    )

    if df.is_empty():
        raise Exception("No valid data points found after parsing.")

    total_data_points = len(df)
    data_start_time = int(cast(int, df["timestamp"].min()))
    data_end_time = int(cast(int, df["timestamp"].max()))

    # Extract sorted columns to main numpy arrays
    var_ids = df["var_id"].to_numpy()
    timestamps_all = df["timestamp"].to_numpy()
    values_all = df["value"].to_numpy()

    # Fast O(N) boundary scan using helper function
    slice_map = _find_start_end_indices_for_each_unique_value(var_ids)

    # Format data as DataInstances (zero-copy slicing views)
    id_to_instance: dict[int, DataInstance] = {}
    cpp_name_to_id: dict[str, int] = {}
    if verbose >= 2:
        di_pbar = tqdm(desc="Creating DataInstances", total=len(id_to_cpp_name))
    for var_id in id_to_cpp_name:
        name = id_to_cpp_name[var_id]
        descript = id_to_descript[var_id]
        cpp_name_to_id[name] = var_id

        if var_id in slice_map:
            start, end = slice_map[var_id]
            timestamps_np = timestamps_all[start:end]
            values_np = values_all[start:end]
        else:
            timestamps_np = np.array([], dtype=np.int64)
            values_np = np.array([], dtype=np.float64)

        id_to_instance[var_id] = DataInstance(
            timestamp_np=timestamps_np,
            value_np=values_np,
            label=descript,
            var_id=var_id,
            cpp_name=name,
        )
        if verbose >= 2:
            di_pbar.update(1)
    if verbose >= 2:
        di_pbar.close()

    # Create and return SingleRunData model
    if verbose >= 1:
        print(f"CSV parsing complete with {parsing_errors} parsing errors.")
    return SingleRunData(
        id_to_instance=id_to_instance,
        cpp_name_to_id=cpp_name_to_id,
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        creation_time=creation_time,
        total_data_points=total_data_points,
        data_start_time=data_start_time,
        data_end_time=data_end_time,
        timestamp_unit=parse_unit,
    )
