import polars as pl
from tqdm import tqdm

from .data_instance import DataInstance
from .single_run_data import SingleRunData
from ..utils.types import Timescale


def _resolve_parse_unit(
    header_line: str,
    parse_unit: Timescale | str | None,
) -> Timescale:
    if isinstance(parse_unit, str):
        parse_unit = parse_unit.strip().lower()
        if parse_unit not in (Timescale.MS.value, Timescale.US.value):
            raise ValueError(
                f"parse_unit must be 'ms' or 'us', got {parse_unit}"
            )
        parse_unit = Timescale(parse_unit)

    if parse_unit is not None and parse_unit not in (Timescale.MS, Timescale.US):
        raise ValueError(
            f"parse_unit must be Timescale.MS or Timescale.US, got {parse_unit}"
        )

    if parse_unit is not None:
        return parse_unit

    return Timescale.US if header_line.rstrip().endswith("v2.0") else Timescale.MS


def parse_csv(
    file_path: str,
    ts_offset: int = 0,
    parsing_errors_limit: int = 100,
    parse_unit: Timescale | str | None = None,
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
        parse_unit = _resolve_parse_unit(header_line, parse_unit)
        print(f"Header: {header_line.rstrip()}")
        print(f"Timestamp unit: {parse_unit.value}")

        # Block 1: Variable ID/Name pairs
        pbar = tqdm(desc="Reading variable ID mappings", unit=" lines", initial=2)
        skip_rows = 1  # header line
        line = f.readline()
        while line and line.startswith("Value "):
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
                    print(
                        f"Warning: Duplicate variable ID {var_id} at line {pbar.n}. Overwriting previous name."
                    )
                id_to_cpp_name[var_id] = cpp_name
                id_to_descript[var_id] = descript

            except Exception as e:
                print(f"Error parsing variable ID/Name pair at line {pbar.n}: {e}")

            line = f.readline()
        pbar.close()

    # Block 2: Read data with Polars, Block 3: Sort — all in one step
    print("Reading and sorting data...")
    df = pl.read_csv(
        file_path,
        skip_rows=skip_rows,
        has_header=False,
        new_columns=["timestamp", "var_id", "value"],
        schema={"column_1": pl.Int64, "column_2": pl.Int32, "column_3": pl.Float64},
        ignore_errors=True,
    )

    parsing_errors = len(df.filter(df["timestamp"].is_null() | df["var_id"].is_null() | df["value"].is_null()))
    if parsing_errors_limit > 0 and parsing_errors >= parsing_errors_limit:
        raise Exception("Too many data parsing errors encountered.")

    df = df.drop_nulls().with_columns(
        (pl.col("timestamp") + ts_offset).alias("timestamp")
    ).sort(["var_id", "timestamp"])

    total_data_points = len(df)
    data_start_time = int(df["timestamp"].min()) if total_data_points > 0 else 0
    data_end_time = int(df["timestamp"].max()) if total_data_points > 0 else 0

    # Build per-variable numpy arrays from grouped Polars data
    var_arrays: dict[int, tuple] = {}
    for (var_id,), group in df.group_by(["var_id"], maintain_order=True):
        var_arrays[int(var_id)] = (
            group["timestamp"].to_numpy(),
            group["value"].to_numpy(),
        )

    # Format data as DataInstances
    id_to_instance: dict[int, DataInstance] = {}
    cpp_name_to_id: dict[str, int] = {}
    for var_id in tqdm(id_to_cpp_name, desc="Creating DataInstances"):
        name = id_to_cpp_name[var_id]
        descript = id_to_descript[var_id]
        cpp_name_to_id[name] = var_id
        timestamps_np, values_np = var_arrays.get(var_id, ([], []))
        id_to_instance[var_id] = DataInstance(
            timestamp_np=timestamps_np,
            value_np=values_np,
            label=descript,
            var_id=var_id,
            cpp_name=name,
        )

    # Create and return SingleRunData model
    print(f"CSV parsing complete with {parsing_errors} parsing errors.")
    return SingleRunData(
        id_to_instance=id_to_instance,
        cpp_name_to_id=cpp_name_to_id,
        id_to_cpp_name=id_to_cpp_name,
        id_to_descript=id_to_descript,
        total_data_points=total_data_points,
        data_start_time=data_start_time,
        data_end_time=data_end_time,
        timestamp_unit=parse_unit,
    )
