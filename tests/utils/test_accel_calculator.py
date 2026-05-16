import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.units import Timescale
from perda.utils.accel_calculator import (
    AccelSegmentResult,
    compute_accel_results,
    detect_accel_event,
)


def test_detect_accel_event_returns_binary_signal(accel_torque, accel_speed):
    result = detect_accel_event(accel_torque, accel_speed)
    unique_vals = set(result.value_np.tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_detect_accel_event_no_torque_no_event():
    torque = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        label="torque",
        var_id=1,
    )
    speed = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([0.0, 5.0, 10.0], dtype=np.float64),
        label="speed",
        var_id=2,
    )
    result = detect_accel_event(torque, speed, torque_threshold=100)
    np.testing.assert_array_equal(result.value_np, [0.0, 0.0, 0.0])


def test_detect_accel_event_no_speed_no_event():
    torque = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([200.0, 200.0, 200.0], dtype=np.float64),
        label="torque",
        var_id=1,
    )
    speed = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        label="speed",
        var_id=2,
    )
    result = detect_accel_event(torque, speed, speed_threshold=0.5)
    np.testing.assert_array_equal(result.value_np, [0.0, 0.0, 0.0])


def test_detect_accel_event_output_aligned_to_speed_timestamps():
    torque = DataInstance(
        timestamp_np=np.array([0, 10, 20], dtype=np.int64),
        value_np=np.array([200.0, 200.0, 0.0], dtype=np.float64),
        label="torque",
        var_id=1,
    )
    speed = DataInstance(
        timestamp_np=np.array([0, 5, 10, 15, 20], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0, 0.0], dtype=np.float64),
        label="speed",
        var_id=2,
    )
    result = detect_accel_event(torque, speed)
    np.testing.assert_array_equal(result.timestamp_np, speed.timestamp_np)


def test_detect_accel_event_event_ends_when_speed_drops():
    torque = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64),
        label="torque",
        var_id=1,
    )
    speed = DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64),
        label="speed",
        var_id=2,
    )
    result = detect_accel_event(
        torque, speed, torque_threshold=100, speed_threshold=0.5
    )
    assert result.value_np[3] == 0.0


def test_detect_accel_event_label(accel_torque, accel_speed):
    result = detect_accel_event(accel_torque, accel_speed)
    assert result.label == "Accel Event"


def test_compute_accel_results_returns_list(accel_scenario):
    sig, dist = accel_scenario
    results = compute_accel_results(sig, dist, target_dist=75)
    assert isinstance(results, list)


def test_compute_accel_results_one_event_detected(accel_scenario):
    sig, dist = accel_scenario
    results = compute_accel_results(sig, dist, target_dist=75)
    assert len(results) == 1


def test_compute_accel_results_dist_reached_matches_target(accel_scenario):
    sig, dist = accel_scenario
    results = compute_accel_results(sig, dist, target_dist=75)
    assert results[0].dist_reached == 75


def test_compute_accel_results_time_to_dist_positive(accel_scenario):
    sig, dist = accel_scenario
    results = compute_accel_results(sig, dist, target_dist=75)
    assert results[0].time_to_dist > 0


def test_compute_accel_results_no_event_when_signal_all_zero():
    n = 50
    ts = np.arange(n, dtype=np.int64)
    sig = DataInstance(timestamp_np=ts, value_np=np.zeros(n), label="sig")
    dist = DataInstance(timestamp_np=ts, value_np=np.linspace(0, 100, n), label="dist")
    results = compute_accel_results(sig, dist)
    assert results == []


def test_compute_accel_results_skips_segment_shorter_than_target():
    n = 20
    ts = np.arange(n, dtype=np.int64) * 10
    signal_vals = np.zeros(n)
    signal_vals[2:8] = 1.0
    sig = DataInstance(timestamp_np=ts, value_np=signal_vals, label="sig")
    dist = DataInstance(timestamp_np=ts, value_np=np.linspace(0, 10, n), label="dist")
    results = compute_accel_results(sig, dist, target_dist=75)
    assert results == []


def test_accel_segment_result_str():
    r = AccelSegmentResult(
        start_time=1.5,
        time_to_dist=3.2,
        dist_reached=75,
        timescale=Timescale.S,
    )
    s = str(r)
    assert "1.50" in s
    assert "3.200" in s
    assert "75" in s
