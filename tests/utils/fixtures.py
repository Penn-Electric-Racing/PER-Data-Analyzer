import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale


@pytest.fixture
def sine_di_factory():
    """Factory that builds a DataInstance containing a pure sine wave.

    Defaults: 1000 samples, 200 Hz sample rate, timestamps in milliseconds.
    """

    def _factory(
        freq_hz: float,
        *,
        n: int = 1000,
        fs_hz: float = 200.0,
        label: str = "sine",
        var_id: int = 1,
    ) -> DataInstance:
        t = np.arange(n) / fs_hz
        values = np.sin(2 * np.pi * freq_hz * t).astype(np.float64)
        ts_ms = (t * 1000).astype(np.int64)
        return DataInstance(
            timestamp_np=ts_ms,
            value_np=values,
            label=label,
            var_id=var_id,
            cpp_name=f"test.{label}",
        )

    return _factory


@pytest.fixture
def sine_di_ms():
    """5 Hz sine wave with timestamps in milliseconds (200 Hz, 1000 samples)."""
    n, fs_hz, freq_hz = 1000, 200.0, 5.0
    t = np.arange(n) / fs_hz
    values = np.sin(2 * np.pi * freq_hz * t).astype(np.float64)
    return DataInstance(
        timestamp_np=(t * 1_000).astype(np.int64),
        value_np=values,
        label="sine_ms",
        var_id=10,
        cpp_name="test.sine_ms",
    )


@pytest.fixture
def sine_di_us():
    """Same 5 Hz sine wave but with timestamps in microseconds."""
    n, fs_hz, freq_hz = 1000, 200.0, 5.0
    t = np.arange(n) / fs_hz
    values = np.sin(2 * np.pi * freq_hz * t).astype(np.float64)
    return DataInstance(
        timestamp_np=(t * 1_000_000).astype(np.int64),
        value_np=values,
        label="sine_us",
        var_id=11,
        cpp_name="test.sine_us",
    )


@pytest.fixture
def uniform_di():
    """500-sample constant-1.0 signal at 1 sample/ms (used for z-score / FFT tests)."""
    n = 500
    ts_ms = np.arange(n, dtype=np.int64)
    return DataInstance(
        timestamp_np=ts_ms,
        value_np=np.ones(n, dtype=np.float64),
        label="uniform",
        var_id=20,
        cpp_name="test.uniform",
    )


@pytest.fixture
def empty_srd():
    """SingleRunData with no variables — used for missing-variable tests."""
    return SingleRunData(
        id_to_instance={},
        cpp_name_to_id={},
        id_to_cpp_name={},
        id_to_descript={},
        total_data_points=0,
        data_start_time=0,
        data_end_time=0,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def ned_srd():
    """SingleRunData with the four variables required by NED_VELOCITY."""
    ts = np.arange(4, dtype=np.int64)
    return SingleRunData(
        id_to_instance={
            1: DataInstance(
                timestamp_np=ts,
                value_np=np.array([1.0, 0.0, -1.0, 0.0]),
                label="pcm.vnav.velocityBody.x",
                var_id=1,
                cpp_name="pcm.vnav.velocityBody.x",
            ),
            2: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.0, 1.0, 0.0, -1.0]),
                label="pcm.vnav.velocityBody.y",
                var_id=2,
                cpp_name="pcm.vnav.velocityBody.y",
            ),
            3: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.1, 0.1, 0.1, 0.1]),
                label="pcm.vnav.velocityBody.z",
                var_id=3,
                cpp_name="pcm.vnav.velocityBody.z",
            ),
            4: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.0, 90.0, 180.0, 270.0]),
                label="pcm.vnav.yawPitchRoll.yaw",
                var_id=4,
                cpp_name="pcm.vnav.yawPitchRoll.yaw",
            ),
        },
        cpp_name_to_id={
            "pcm.vnav.velocityBody.x": 1,
            "pcm.vnav.velocityBody.y": 2,
            "pcm.vnav.velocityBody.z": 3,
            "pcm.vnav.yawPitchRoll.yaw": 4,
        },
        id_to_cpp_name={
            1: "pcm.vnav.velocityBody.x",
            2: "pcm.vnav.velocityBody.y",
            3: "pcm.vnav.velocityBody.z",
            4: "pcm.vnav.yawPitchRoll.yaw",
        },
        id_to_descript={1: "", 2: "", 3: "", 4: ""},
        total_data_points=16,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def ws_srd():
    """SingleRunData with the four wheel speed variables (in mph)."""
    ts = np.arange(4, dtype=np.int64)
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    names = [
        "pcm.wheelSpeeds.frontRight",
        "pcm.wheelSpeeds.frontLeft",
        "pcm.wheelSpeeds.rearRight",
        "pcm.wheelSpeeds.rearLeft",
    ]
    instances = {
        i
        + 1: DataInstance(
            timestamp_np=ts,
            value_np=vals.copy(),
            label=name,
            var_id=i + 1,
            cpp_name=name,
        )
        for i, name in enumerate(names)
    }
    return SingleRunData(
        id_to_instance=instances,
        cpp_name_to_id={name: i + 1 for i, name in enumerate(names)},
        id_to_cpp_name={i + 1: name for i, name in enumerate(names)},
        id_to_descript={i + 1: "" for i in range(len(names))},
        total_data_points=16,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def motor_srd():
    """SingleRunData with the motor RPM variable (negative = forward)."""
    ts = np.arange(4, dtype=np.int64)
    return SingleRunData(
        id_to_instance={
            1: DataInstance(
                timestamp_np=ts,
                value_np=np.array([-1000.0, -2000.0, -3000.0, 0.0]),
                label="pcm.moc.motor.angularSpeed",
                var_id=1,
                cpp_name="pcm.moc.motor.angularSpeed",
            )
        },
        cpp_name_to_id={"pcm.moc.motor.angularSpeed": 1},
        id_to_cpp_name={1: "pcm.moc.motor.angularSpeed"},
        id_to_descript={1: ""},
        total_data_points=4,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def steering_srd():
    """SingleRunData with both raw voltage and pre-existing angle channels.
    Raw voltage intentionally hits the three default calibration points (1.86, 2.93, 3.96 V)
    plus a midpoint.
    """
    ts = np.arange(4, dtype=np.int64)
    raw_volts = np.array([1.86, 2.93, 3.96, 2.93], dtype=np.float64)
    stale_angles = np.array(
        [-90.0, 5.0, 85.0, 5.0], dtype=np.float64
    )  # deliberately wrong
    return SingleRunData(
        id_to_instance={
            1: DataInstance(
                timestamp_np=ts,
                value_np=raw_volts,
                label="ludwig.steeringWheel.raw",
                var_id=1,
                cpp_name="ludwig.steeringWheel.raw",
            ),
            2: DataInstance(
                timestamp_np=ts,
                value_np=stale_angles,
                label="ludwig.steeringWheel.angle",
                var_id=2,
                cpp_name="ludwig.steeringWheel.angle",
            ),
        },
        cpp_name_to_id={
            "ludwig.steeringWheel.raw": 1,
            "ludwig.steeringWheel.angle": 2,
        },
        id_to_cpp_name={1: "ludwig.steeringWheel.raw", 2: "ludwig.steeringWheel.angle"},
        id_to_descript={1: "", 2: ""},
        total_data_points=8,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def steering_srd_no_angle():
    """SingleRunData with only the raw voltage channel (no pre-existing angle)."""
    ts = np.arange(3, dtype=np.int64)
    raw_volts = np.array([1.86, 2.93, 3.96], dtype=np.float64)
    return SingleRunData(
        id_to_instance={
            1: DataInstance(
                timestamp_np=ts,
                value_np=raw_volts,
                label="ludwig.steeringWheel.raw",
                var_id=1,
                cpp_name="ludwig.steeringWheel.raw",
            ),
        },
        cpp_name_to_id={"ludwig.steeringWheel.raw": 1},
        id_to_cpp_name={1: "ludwig.steeringWheel.raw"},
        id_to_descript={1: ""},
        total_data_points=3,
        data_start_time=0,
        data_end_time=2,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def accel_torque():
    """Torque DataInstance for detect_accel_event tests (high except at endpoints)."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([0.0, 200.0, 200.0, 200.0, 0.0], dtype=np.float64),
        label="torque",
        var_id=1,
    )


@pytest.fixture
def accel_speed():
    """Speed DataInstance for detect_accel_event tests."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0, 0.3], dtype=np.float64),
        label="speed",
        var_id=2,
    )


@pytest.fixture
def accel_scenario():
    """One clear acceleration event that covers 75+ m.

    200-point grid; signal active from index 10 onwards (no drop).
    Distance goes 0->200 m over the full range, so starting at index 10
    (~10 m) the remaining distance (190 m) exceeds the 75 m target.
    """
    n = 200
    ts = np.arange(n, dtype=np.int64)
    signal_vals = np.zeros(n)
    signal_vals[10:] = 1.0
    signal_di = DataInstance(timestamp_np=ts, value_np=signal_vals, label="sig")
    dist_di = DataInstance(
        timestamp_np=ts, value_np=np.linspace(0, 200, n), label="dist"
    )
    return signal_di, dist_di
