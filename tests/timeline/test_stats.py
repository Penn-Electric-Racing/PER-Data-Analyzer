from perda.timeline.stats import STATS_COLUMNS, compute_session_stats


def test_stats_have_expected_columns(sentinel_frame):
    stats = compute_session_stats(sentinel_frame)
    assert stats.columns == [*STATS_COLUMNS, "dtype"]
    assert stats.height == 3


def test_stats_exclude_sentinels_from_distribution(sentinel_frame):
    stats = compute_session_stats(sentinel_frame).sort("var_id")
    first = stats.filter(stats["var_id"] == 1).to_dicts()[0]
    assert first["n"] == 4
    assert first["n_invalid"] == 2
    assert first["sentinel_bits"] == 2
    # Sentinels are ~1e-45; a naive min would report one of them instead of 1.0.
    assert first["v_min"] == 1.0
    assert first["v_max"] == 3.5
    assert first["v_mean"] == 2.25


def test_fully_invalid_variable_has_no_distribution(sentinel_frame):
    stats = compute_session_stats(sentinel_frame)
    third = stats.filter(stats["var_id"] == 3).to_dicts()[0]
    assert third["n"] == 2
    assert third["n_invalid"] == 2
    assert third["v_min"] is None
    assert third["v_max"] is None


def test_sample_rate_uses_span(sentinel_frame):
    stats = compute_session_stats(sentinel_frame)
    second = stats.filter(stats["var_id"] == 2).to_dicts()[0]
    assert second["hz"] == 1.0
    assert second["max_gap_us"] == 1_000_000


def test_dtype_inferred_alongside_stats(sentinel_frame):
    stats = compute_session_stats(sentinel_frame)
    dtypes = {r["var_id"]: r["dtype"] for r in stats.iter_rows(named=True)}
    assert dtypes[1] == "float"
    assert dtypes[2] == "bool"
    assert dtypes[3] == "unknown"
