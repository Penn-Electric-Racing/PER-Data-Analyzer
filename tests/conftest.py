import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale

# ---------------------------------------------------------------------------
# DataInstance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def di_simple():
    """Short, clean DataInstance with timestamps [0,1,2,3] and linear values."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0]),
        label="simple",
        var_id=1,
        cpp_name="test.simple",
    )


@pytest.fixture
def di_sparse():
    """Irregular, wide-gap DataInstance — primary 'right' side for join tests."""
    return DataInstance(
        timestamp_np=np.array([0, 10, 100], dtype=np.int64),
        value_np=np.array([5.0, 15.0, 25.0]),
        label="sparse",
        var_id=2,
        cpp_name="test.sparse",
    )


@pytest.fixture
def di_with_metadata():
    """All five fields explicitly set; use when testing metadata round-trips."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3], dtype=np.int64),
        value_np=np.array([10.0, 20.0, 30.0, 40.0]),
        label="meta",
        var_id=99,
        cpp_name="test.meta",
    )


# ---------------------------------------------------------------------------
# SingleRunData fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def srd_basic():
    """Minimal but complete SingleRunData with two variables."""
    di_a = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 3.0]),
        label="var_a",
        var_id=1,
        cpp_name="cpp.var_a",
    )
    di_b = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([4.0, 5.0, 6.0]),
        label="var_b",
        var_id=2,
        cpp_name="cpp.var_b",
    )
    return SingleRunData(
        id_to_instance={1: di_a, 2: di_b},
        cpp_name_to_id={"cpp.var_a": 1, "cpp.var_b": 2},
        id_to_cpp_name={1: "cpp.var_a", 2: "cpp.var_b"},
        id_to_descript={1: "Variable A description", 2: "Variable B description"},
        total_data_points=6,
        data_start_time=0,
        data_end_time=2,
        timestamp_unit=Timescale.MS,
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
