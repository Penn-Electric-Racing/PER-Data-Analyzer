import polars as pl


def frame_with_sentinels() -> pl.DataFrame:
    """Build a frame mixing real values with denormal sentinels.

    Variable 1 is half sentinel (and non-integral), variable 2 is a clean
    boolean, and variable 3 is entirely
    sentinel -- the case that makes naive aggregates report a plausible value
    for a completely broken signal.

    Returns
    -------
    pl.DataFrame
        Columns ``var_id``, ``t_us`` and ``value``, sorted by (var_id, t_us).
    """
    sentinel = 2.802596928649634e-45
    return pl.DataFrame(
        {
            "var_id": [1, 1, 1, 1, 2, 2, 2, 3, 3],
            "t_us": [0, 1_000_000, 2_000_000, 3_000_000, 0, 1_000_000, 2_000_000, 0, 1_000_000],
            "value": [1.0, sentinel, 3.5, sentinel, 0.0, 1.0, 0.0, sentinel, sentinel],
        }
    ).sort(["var_id", "t_us"])
