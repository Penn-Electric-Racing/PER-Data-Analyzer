import textwrap
import pytest

from perda.analyzer.csv import parse_csv
from perda.analyzer.analyzer import Analyzer
from perda.analyzer.comparison import plot_comparison

def test_downsampling_strict_ceiling(tmp_path):
    # Write a log with 105 data points
    lines = ["PER Log: Thu Jun 11 10:00:00 2026 v2.0", "Value voltage (ams.pack.voltage): 1"]
    for i in range(105):
        lines.append(f"{i * 100},1,{i * 0.1}")
    
    p = tmp_path / "log_large.csv"
    p.write_text("\n".join(lines))

    # Initialize Analyzer directly with file path
    aly = Analyzer(str(p), verbose=0)
    di = aly.data["ams.pack.voltage"]
    assert len(di) == 105

    # Request max_points = 20
    # Expected ceiling division step: (105 + 20 - 1) // 20 = 124 // 20 = 6
    # Slicing length: ceil(105 / 6) = 18 points (which is <= 20)
    fig = plot_comparison([aly], "ams.pack.voltage", max_points=20)
    assert len(fig.data) == 1
    trace_points = len(fig.data[0].x)
    assert trace_points <= 20
    assert trace_points == 18

    # Verify that requesting max_points larger than length preserves all points
    fig_full = plot_comparison([aly], "ams.pack.voltage", max_points=200)
    assert len(fig_full.data[0].x) == 105
