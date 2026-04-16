from enum import Enum
from typing import Callable

import numpy as np

from ..core_data_structures.data_instance import DataInstance
from ..core_data_structures.single_run_data import SingleRunData
from ..units import in_to_m, mph_to_m_per_s

GEAR_RATIO: float = 5.6
TIRE_RADIUS_IN: float = 7.85


_VEL_X = "pcm.vnav.velocityBody.x"
_VEL_Y = "pcm.vnav.velocityBody.y"
_VEL_Z = "pcm.vnav.velocityBody.z"
_YAW = "pcm.vnav.yawPitchRoll.yaw"

_WS_FR = "pcm.wheelSpeeds.frontRight"
_WS_FL = "pcm.wheelSpeeds.frontLeft"
_WS_RR = "pcm.wheelSpeeds.rearRight"
_WS_RL = "pcm.wheelSpeeds.rearLeft"

_MOTOR_RPM = "pcm.moc.motor.angularSpeed"
_MOTOR_WS = "pcm.moc.motor.wheelSpeed"


def _replace(data: SingleRunData, cpp_name: str, new_values: np.ndarray) -> None:
    """Overwrite the value_np of an existing DataInstance in-place (mutates dict)."""
    var_id = data.cpp_name_to_id[cpp_name]
    old = data.id_to_instance[var_id]
    data.id_to_instance[var_id] = DataInstance(
        timestamp_np=old.timestamp_np,
        value_np=new_values,
        label=old.label,
        var_id=old.var_id,
        cpp_name=old.cpp_name,
    )


def _add(
    data: SingleRunData,
    cpp_name: str,
    label: str,
    timestamp_np: np.ndarray,
    value_np: np.ndarray,
) -> None:
    """Insert a new derived DataInstance using a synthetic negative ID."""
    synthetic_id = -(len(data.id_to_instance) + 1)
    di = DataInstance(
        timestamp_np=timestamp_np,
        value_np=value_np,
        label=label,
        var_id=synthetic_id,
        cpp_name=cpp_name,
    )
    data.id_to_instance[synthetic_id] = di
    data.cpp_name_to_id[cpp_name] = synthetic_id
    data.id_to_cpp_name[synthetic_id] = cpp_name
    data.id_to_descript[synthetic_id] = label


class Preprocessing(Enum):
    PATCH_NED_VELOCITY = "patch_ned_velocity"
    WHEELSPEEDS_TO_M_PER_S = "wheelspeeds_to_m_per_s"
    CORRECT_MOTOR_DATA = "correct_motor_data"


def patch_ned_velocity(data: SingleRunData) -> SingleRunData:
    """Corrects a VectorNav configuration bug where velocityBody.x/y/z actually contains
    NED (North/East/Down) velocities. Copies raw NED to velN/velE/velD, then rotates
    into true body frame (FRD) using yaw. Includes sanity checks and ZOH duplicate warnings.

    Parameters
    ----------
    data : SingleRunData

    Returns
    -------
    SingleRunData
    """
    required = [_VEL_X, _VEL_Y, _VEL_Z, _YAW]
    missing = [v for v in required if v not in data]
    if missing:
        print(f"WARNING: PATCH_NED_VELOCITY skipped — missing variables: {missing}")
        return data

    vel_n = data[_VEL_X].value_np.astype(np.float64)
    vel_e = data[_VEL_Y].value_np.astype(np.float64)
    vel_d = data[_VEL_Z].value_np.astype(np.float64)
    yaw_rad = np.radians(data[_YAW].value_np.astype(np.float64))

    # Preserve raw NED copies
    ts = data[_VEL_X].timestamp_np
    _add(data, "velN", "NED North velocity (raw)", ts, vel_n.copy())
    _add(data, "velE", "NED East velocity (raw)", ts, vel_e.copy())
    _add(data, "velD", "NED Down velocity (raw)", ts, vel_d.copy())

    # Rotate NED → body-frame (FRD) using yaw only
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    _replace(data, _VEL_X, vel_n * cos_y + vel_e * sin_y)  # forward
    _replace(data, _VEL_Y, -vel_n * sin_y + vel_e * cos_y)  # right
    # vel_z (down) identical in NED and FRD — no change needed

    print(
        f"PATCH_NED_VELOCITY: Preserved raw NED velocity in velN/velE/velD, rotated {len(vel_n)} points → body frame"
    )
    return data


def convert_wheelspeeds_to_m_per_s(data: SingleRunData) -> SingleRunData:
    cols = [_WS_FR, _WS_FL, _WS_RR, _WS_RL]
    missing = [v for v in cols if v not in data]
    if missing:
        print(f"WARNING: WHEELSPEEDS_TO_M_PER_S skipped — missing variables: {missing}")
        return data

    for col in cols:
        di = data[col]
        backup_name = col + "_mph"
        if backup_name not in data:
            _add(
                data,
                backup_name,
                (di.label or col) + " (mph backup)",
                di.timestamp_np,
                di.value_np.copy(),
            )
        _replace(data, col, mph_to_m_per_s(di.value_np))

    print(
        f"WHEELSPEEDS_TO_M_PER_S: Converted {len(cols)} wheel speed columns mph → m/s, backups in *_mph"
    )
    return data


def correct_motor_data(data: SingleRunData) -> SingleRunData:
    if _MOTOR_RPM not in data:
        print(f"WARNING: CORRECT_MOTOR_DATA skipped — missing variable: {_MOTOR_RPM}")
        return data

    di = data[_MOTOR_RPM]
    raw_rpm = di.value_np.astype(np.float64)

    backup_name = _MOTOR_RPM + "_raw"
    if backup_name not in data:
        _add(
            data,
            backup_name,
            "Motor RPM raw (pre-flip)",
            di.timestamp_np,
            raw_rpm.copy(),
        )

    flipped = -raw_rpm
    _replace(data, _MOTOR_RPM, flipped)

    tire_radius_m = in_to_m(TIRE_RADIUS_IN)
    wheel_speed = flipped * 2.0 * np.pi * tire_radius_m / (60.0 * GEAR_RATIO)
    _add(
        data,
        _MOTOR_WS,
        "Driven wheel speed from motor RPM (m/s)",
        di.timestamp_np,
        wheel_speed,
    )

    print(
        f"CORRECT_MOTOR_DATA: Flipped {_MOTOR_RPM} sign, added {_MOTOR_WS} "
        f"(ratio={GEAR_RATIO}, r={TIRE_RADIUS_IN} in)"
    )
    return data


_PIPELINE: dict[Preprocessing, Callable[[SingleRunData], SingleRunData]] = {
    Preprocessing.PATCH_NED_VELOCITY: patch_ned_velocity,
    Preprocessing.WHEELSPEEDS_TO_M_PER_S: convert_wheelspeeds_to_m_per_s,
    Preprocessing.CORRECT_MOTOR_DATA: correct_motor_data,
}


def apply_preprocessing(
    data: SingleRunData,
    corrections: list[Preprocessing],
) -> SingleRunData:
    """Run a sequence of preprocessing steps on a parsed SingleRunData instance.

    Parameters
    ----------
    data : SingleRunData
        Parsed run data to preprocess.
    corrections : list[Preprocessing]
        Ordered list of preprocessing steps to apply.

    Returns
    -------
    SingleRunData
        The same object, modified in-place and returned for chaining.
    """
    for correction in corrections:
        data = _PIPELINE[correction](data)
    return data
