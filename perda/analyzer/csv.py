from __future__ import annotations

import re
from datetime import datetime
from typing import TextIO, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import Timescale
from ..utils.search import build_semantic_index

DATA_COLUMN_NAMES = ["timestamp", "var_id", "value"]
DATA_COLUMN_SCHEMA = {
    "column_1": pl.Int64,
    "column_2": pl.Int32,
    "column_3": pl.Float64,
}
VALUE_LINE_PREFIX = "Value "
MICROSECOND_HEADER_SUFFIX = "v2.0"
VARIABLE_LINE_PATTERN = re.compile(
    rf"""
    ^{re.escape(VALUE_LINE_PREFIX)}\s*      # prefix marking a variable declaration
    (?P<description>.*?)                    # free-form description
    \s*
    \(\s*(?P<cpp_name>[^()]+?)\s*\)         # parenthesised C++ name
    \s*:\s*                                 # separator before the variable ID
    (?P<var_id>\d+)\s*$                     # variable ID
    """,
    re.VERBOSE,
)


class ParsedVariableLine(BaseModel):
    cpp_name: str = Field(description="C++ name of the variable")
    description: str = Field(description="Human readable description of the variable")
    var_id: int = Field(description="Numeric ID the data rows refer to")


class VariableMappings(BaseModel):
    id_to_cpp_name: dict[int, str] = Field(
        description="Mapping from variable ID to variable name"
    )
    id_to_descript: dict[int, str] = Field(
        description="Mapping from variable ID to variable description"
    )
    skip_rows: int = Field(
        description="Number of leading rows before the numeric data section"
    )


class ParsedDataFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_frame: pl.DataFrame = Field(
        description="Data sorted by variable ID then timestamp"
    )
    parsing_errors: int = Field(description="Number of rows dropped as unparseable")


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


def parse_variable_mapping_line(line: str) -> ParsedVariableLine:
    """
    Parse one variable declaration line into its three fields.

    Parameters
    ----------
    line : str
        Declaration line, e.g. ``"Value pack voltage (ams.pack.voltage): 1"``.

    Returns
    -------
    ParsedVariableLine
        The variable's C++ name, description, and ID.
    """
    match = VARIABLE_LINE_PATTERN.match(line)
    if match is None:
        raise ValueError(f"Malformed variable declaration line: {line.strip()}")

    return ParsedVariableLine(
        cpp_name=match.group("cpp_name").strip(),
        description=match.group("description").strip(),
        var_id=int(match.group("var_id")),
    )


def parse_variable_id_mappings(
    file_handle: TextIO, verbose: int = 1
) -> VariableMappings:
    """
    Read the variable ID and name mapping lines that follow the file header.

    Parameters
    ----------
    file_handle : TextIO
        Open log file positioned just after the header line.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 for warnings, 2 for progress bars. Default is 1.

    Returns
    -------
    VariableMappings
        Parsed lookup tables and the row offset where numeric data begins.
    """
    id_to_cpp_name: dict[int, str] = {}
    id_to_descript: dict[int, str] = {}

    progress_bar = (
        tqdm(desc="Reading variable ID mappings", unit=" lines", initial=2)
        if verbose >= 2
        else None
    )

    skip_rows = 1  # header line
    line_number = 1
    line = file_handle.readline()

    while line and line.startswith(VALUE_LINE_PREFIX):
        if progress_bar is not None:
            progress_bar.update(1)
        skip_rows += 1
        line_number += 1

        try:
            parsed_line = parse_variable_mapping_line(line)

            if parsed_line.var_id in id_to_cpp_name and verbose >= 1:
                print(
                    f"Warning: Duplicate variable ID {parsed_line.var_id} at line "
                    f"{line_number}. Overwriting previous name."
                )

            id_to_cpp_name[parsed_line.var_id] = parsed_line.cpp_name
            id_to_descript[parsed_line.var_id] = parsed_line.description

        except ValueError as e:
            if verbose >= 1:
                print(f"Error parsing variable ID/Name pair at line {line_number}: {e}")

        line = file_handle.readline()

    if progress_bar is not None:
        progress_bar.close()

    return VariableMappings(
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        skip_rows=skip_rows,
    )


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


def read_and_sort_data(
    file_path: str,
    skip_rows: int,
    ts_offset: int = 0,
    parsing_errors_limit: int = 100,
    verbose: int = 1,
) -> ParsedDataFrame:
    """
    Read the numeric data section and sort it by variable ID then timestamp.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to read.
    skip_rows : int
        Number of leading rows to skip before the numeric data.
    ts_offset : int, optional
        Timestamp offset applied to all data points. Default is 0.
    parsing_errors_limit : int, optional
        Maximum number of malformed rows tolerated. -1 for no limit. Default is 100.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 or higher for status. Default is 1.

    Returns
    -------
    ParsedDataFrame
        Sorted data and the count of malformed rows dropped.
    """
    if verbose >= 1:
        print("Reading and sorting data...")

    df = pl.read_csv(
        file_path,
        skip_rows=skip_rows,
        has_header=False,
        new_columns=DATA_COLUMN_NAMES,
        schema=DATA_COLUMN_SCHEMA,
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

    return ParsedDataFrame(data_frame=df, parsing_errors=parsing_errors)


def build_data_instances(
    mappings: VariableMappings, data_frame: pl.DataFrame, verbose: int = 1
) -> dict[int, DataInstance]:
    """
    Slice the sorted data into one DataInstance per variable.

    Parameters
    ----------
    mappings : VariableMappings
        Variable ID lookup tables.
    data_frame : pl.DataFrame
        Data sorted by variable ID then timestamp.
    verbose : int, optional
        Verbosity level. 0 for no output, 2 for progress bars. Default is 1.

    Returns
    -------
    dict[int, DataInstance]
        Mapping from variable ID to its DataInstance.

    Notes
    -----
    Slicing the shared arrays yields zero-copy numpy views rather than duplicating data.
    Variables declared in the mapping lines but absent from the data get empty arrays.
    """
    var_ids = data_frame["var_id"].to_numpy()
    timestamps_all = data_frame["timestamp"].to_numpy()
    values_all = data_frame["value"].to_numpy()

    slice_map = _find_start_end_indices_for_each_unique_value(var_ids)

    id_to_instance: dict[int, DataInstance] = {}
    progress_bar = (
        tqdm(desc="Creating DataInstances", total=len(mappings.id_to_cpp_name))
        if verbose >= 2
        else None
    )

    for var_id, cpp_name in mappings.id_to_cpp_name.items():
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
            label=mappings.id_to_descript[var_id],
            var_id=var_id,
            cpp_name=cpp_name,
        )
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    return id_to_instance


def parse_csv(
    file_path: str,
    ts_offset: int = 0,
    parsing_errors_limit: int = 100,
    verbose: int = 1,
    build_search_index: bool = False,
) -> SingleRunData:
    """
    Parse CSV file and return SingleRunData model.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to parse.
    ts_offset : int, optional
        Timestamp offset applied to all data points. Default is 0.
    parsing_errors_limit : int, optional
        Maximum number of parsing errors before stopping. -1 for no limit. Default is 100.
    verbose : int, optional
        Verbosity level. 0 for no output, 1 for basic output, 2 for detailed output. Default is 1.
    build_search_index : bool, optional
        Whether to vectorize variable descriptions for semantic search. Requires the
        ``semantic`` extra and adds noticeable time to parsing. Default is False.

    Returns
    -------
    SingleRunData
        Parsed data structure containing all variables.

    Notes
    -----
    The timestamp unit is auto-detected from the header suffix: "v2.0" means
    microseconds, anything else means milliseconds.
    """
    with open(file_path, "r") as f:
        header_line = f.readline()
        parse_unit = (
            Timescale.US
            if header_line.rstrip().endswith(MICROSECOND_HEADER_SUFFIX)
            else Timescale.MS
        )
        creation_time = parse_header_creation_time(header_line)

        if verbose >= 1:
            print(f"Header: {header_line.rstrip()}")
            print(f"Timestamp unit: {parse_unit.value}")
            if creation_time:
                print(f"Log recorded on: {creation_time}")

        mappings = parse_variable_id_mappings(f, verbose=verbose)

    parsed = read_and_sort_data(
        file_path,
        mappings.skip_rows,
        ts_offset=ts_offset,
        parsing_errors_limit=parsing_errors_limit,
        verbose=verbose,
    )

    semantic_index = (
        build_semantic_index(mappings.id_to_descript, verbose=verbose)
        if build_search_index
        else None
    )

    id_to_instance = build_data_instances(mappings, parsed.data_frame, verbose=verbose)

    if verbose >= 1:
        print(f"CSV parsing complete with {parsed.parsing_errors} parsing errors.")

    return SingleRunData(
        id_to_instance=id_to_instance,
        cpp_name_to_id={
            cpp_name: var_id for var_id, cpp_name in mappings.id_to_cpp_name.items()
        },
        id_to_cpp_name=mappings.id_to_cpp_name,
        id_to_descript=mappings.id_to_descript,
        creation_time=creation_time,
        total_data_points=len(parsed.data_frame),
        data_start_time=int(cast(int, parsed.data_frame["timestamp"].min())),
        data_end_time=int(cast(int, parsed.data_frame["timestamp"].max())),
        timestamp_unit=parse_unit,
        semantic_index=semantic_index,
    )
