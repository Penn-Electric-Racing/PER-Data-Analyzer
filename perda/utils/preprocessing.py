from typing import Callable, Union

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..core_data_structures.single_run_data import SingleRunData
from ..units import in_to_m, mph_to_m_per_s

GEAR_RATIO: float = 5.6
TIRE_RADIUS_IN: float = 7.85

VECTORNAV_BODY_VEL_X = "pcm.vnav.velocityBody.x"
VECTORNAV_BODY_VEL_Y = "pcm.vnav.velocityBody.y"
VECTORNAV_BODY_VEL_Z = "pcm.vnav.velocityBody.z"
VECTORNAV_YAW = "pcm.vnav.yawPitchRoll.yaw"

WHEELSPEED_FR = "pcm.wheelSpeeds.frontRight"
WHEELSPEED_FL = "pcm.wheelSpeeds.frontLeft"
WHEELSPEED_RR = "pcm.wheelSpeeds.rearRight"
WHEELSPEED_RL = "pcm.wheelSpeeds.rearLeft"

MOTOR_RPM = "pcm.moc.motor.angularSpeed"
MOTOR_WHEELSPEED = "pcm.moc.motor.wheelSpeed"

STEERING_RAW = "ludwig.steeringWheel.raw"
STEERING_ANGLE = "ludwig.steeringWheel.angle"

# Default 3-point voltage->angle calibration for the steering pot.
SteeringCalibration = tuple[tuple[float, float], ...]
DEFAULT_STEERING_CALIBRATION: SteeringCalibration = (
    (1.86, -97.0),  # max left
    (2.93, 0.0),  # zero
    (3.96, 97.0),  # max right
)

# Type alias for a preprocessing step function that takes and returns a SingleRunData instance.
PreprocessingStep = Callable[[SingleRunData], SingleRunData]


def patch_ned_velocity(data: SingleRunData) -> SingleRunData:
    """Correct a VectorNav bug where velocityBody.x/y/z contains NED instead of body-frame velocities.

    Copies the raw NED values to velN/velE/velD, then rotates into true body-frame (FRD) using yaw.

    Parameters
    ----------
    data : SingleRunData

    Returns
    -------
    SingleRunData
    """
    required = [
        VECTORNAV_BODY_VEL_X,
        VECTORNAV_BODY_VEL_Y,
        VECTORNAV_BODY_VEL_Z,
        VECTORNAV_YAW,
    ]
    missing = [v for v in required if v not in data]
    if missing:
        print(f"WARNING: patch_ned_velocity skipped — missing variables: {missing}")
        return data

    vel_n1, vel_e1, vel_d1, yaw_deg = left_join_data_instances(
        data[VECTORNAV_BODY_VEL_X],
        [data[VECTORNAV_BODY_VEL_Y], data[VECTORNAV_BODY_VEL_Z], data[VECTORNAV_YAW]],
    )

    yaw_rad = np.radians(yaw_deg.value_np)

    data["velN"] = DataInstance(
        timestamp_np=vel_n1.timestamp_np,
        value_np=vel_n1.value_np.copy(),
        label="NED North velocity (raw)",
        cpp_name="velN",
    )
    data["velE"] = DataInstance(
        timestamp_np=vel_e1.timestamp_np,
        value_np=vel_e1.value_np.copy(),
        label="NED East velocity (raw)",
        cpp_name="velE",
    )
    data["velD"] = DataInstance(
        timestamp_np=vel_d1.timestamp_np,
        value_np=vel_d1.value_np.copy(),
        label="NED Down velocity (raw)",
        cpp_name="velD",
    )

    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    data[VECTORNAV_BODY_VEL_X] = DataInstance(
        timestamp_np=vel_n1.timestamp_np,
        value_np=vel_n1.value_np * cos_y + vel_e1.value_np * sin_y,
        label=data[VECTORNAV_BODY_VEL_X].label,
        cpp_name=VECTORNAV_BODY_VEL_X,
    )  # forward
    data[VECTORNAV_BODY_VEL_Y] = DataInstance(
        timestamp_np=vel_e1.timestamp_np,
        value_np=-vel_n1.value_np * sin_y + vel_e1.value_np * cos_y,
        label=data[VECTORNAV_BODY_VEL_Y].label,
        cpp_name=VECTORNAV_BODY_VEL_Y,
    )  # right
    # vel_z (down) is identical in NED and FRD — no change needed

    print(
        f"patch_ned_velocity: preserved raw NED in velN/velE/velD, rotated {len(vel_n1)} points to body frame"
    )
    return data


def convert_wheelspeeds_to_m_per_s(data: SingleRunData) -> SingleRunData:
    """Convert wheel speed channels from mph to m/s, preserving originals as *_mph.

    Parameters
    ----------
    data : SingleRunData

    Returns
    -------
    SingleRunData
    """
    cols = [WHEELSPEED_FR, WHEELSPEED_FL, WHEELSPEED_RR, WHEELSPEED_RL]
    missing = [v for v in cols if v not in data]
    if missing:
        print(
            f"WARNING: convert_wheelspeeds_to_m_per_s skipped — missing variables: {missing}"
        )
        return data

    for col in cols:
        di = data[col]
        backup_name = col + "_mph"
        if backup_name not in data:
            data[backup_name] = DataInstance(
                timestamp_np=di.timestamp_np,
                value_np=di.value_np.copy(),
                label=(di.label or col) + " (mph backup)",
                cpp_name=backup_name,
            )
        data[col] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=mph_to_m_per_s(di.value_np),
            label=di.label,
            cpp_name=col,
        )

    print(
        f"convert_wheelspeeds_to_m_per_s: converted {len(cols)} channels mph -> m/s, backups in *_mph"
    )
    return data


def correct_motor_data(data: SingleRunData) -> SingleRunData:
    """Flip the motor RPM sign and derive driven wheel speed.

    Parameters
    ----------
    data : SingleRunData

    Returns
    -------
    SingleRunData
    """
    if MOTOR_RPM not in data:
        print(f"WARNING: correct_motor_data skipped — missing variable: {MOTOR_RPM}")
        return data

    di = data[MOTOR_RPM]
    raw_rpm: NDArray[float64] = di.value_np.astype(np.float64)

    backup_name = MOTOR_RPM + "_raw"
    if backup_name not in data:
        data[backup_name] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=raw_rpm.copy(),
            label="Motor RPM raw (pre-flip)",
            cpp_name=backup_name,
        )

    flipped = -raw_rpm
    data[MOTOR_RPM] = DataInstance(
        timestamp_np=di.timestamp_np,
        value_np=flipped,
        label=di.label,
        cpp_name=MOTOR_RPM,
    )

    tire_radius_m = in_to_m(TIRE_RADIUS_IN)
    wheel_speed: NDArray[float64] = (
        flipped * 2.0 * np.pi * tire_radius_m / (60.0 * GEAR_RATIO)
    )
    data[MOTOR_WHEELSPEED] = DataInstance(
        timestamp_np=di.timestamp_np,
        value_np=wheel_speed,
        label="Driven wheel speed from motor RPM (m/s)",
        cpp_name=MOTOR_WHEELSPEED,
    )

    print(
        f"correct_motor_data: flipped {MOTOR_RPM} sign, added {MOTOR_WHEELSPEED} "
        f"(ratio={GEAR_RATIO}, r={TIRE_RADIUS_IN} in)"
    )
    return data


class CorrectSteeringAngleLambda:
    """Callable preprocessing step for recomputing the steering angle to account for drift.

    Fits a polynomial through the calibration points and
    rewrites the angle channel in-place, preserving the original as
    ``ludwig.steeringWheel.angle_original``.
    """

    def __init__(
        self,
        calibration: SteeringCalibration = DEFAULT_STEERING_CALIBRATION,
    ) -> None:
        pts = sorted(calibration, key=lambda p: p[0])
        if len(pts) < 2:
            raise ValueError("calibration needs at least 2 (voltage, angle) points.")
        self.pts = pts

        volts: NDArray[float64] = np.array([p[0] for p in pts], dtype=np.float64)
        angles: NDArray[float64] = np.array([p[1] for p in pts], dtype=np.float64)
        self.deg = 2 if len(pts) >= 3 else 1
        self.coeffs: NDArray[float64] = np.polyfit(volts, angles, self.deg)

    def __call__(
        self,
        data_or_calibration: SingleRunData | SteeringCalibration,
    ) -> Union[SingleRunData, "CorrectSteeringAngleLambda"]:
        """Apply to data, or return a reconfigured step when called with calibration.

        Parameters
        ----------
        data_or_calibration : SingleRunData or tuple of (voltage, angle) pairs
            Pass ``SingleRunData`` to apply the step (normal pipeline use).
            Pass a calibration tuple to return a new configured ``CorrectSteeringAngleLambda``.

        Returns
        -------
        SingleRunData
            When called with data (pipeline use).
        CorrectSteeringAngleLambda
            When called with calibration (partial-application use).

        Examples
        --------
        >>> Analyzer(preprocessing=[correct_steering_angle])
        >>> Analyzer(preprocessing=[correct_steering_angle(calibration=((1.5, -90.0), (3.0, 0.0), (4.5, 90.0)))])
        """
        if isinstance(data_or_calibration, tuple):
            return CorrectSteeringAngleLambda(calibration=data_or_calibration)

        data: SingleRunData = data_or_calibration  # type: ignore[assignment]
        if STEERING_RAW not in data:
            print(
                f"WARNING: correct_steering_angle skipped — missing variable: {STEERING_RAW}"
            )
            return data

        raw_di = data[STEERING_RAW]
        raw_volts: NDArray[float64] = raw_di.value_np.astype(np.float64)
        recomputed: NDArray[float64] = np.polyval(self.coeffs, raw_volts)

        backup_name = STEERING_ANGLE + "_original"
        if STEERING_ANGLE in data and backup_name not in data:
            orig = data[STEERING_ANGLE]
            data[backup_name] = DataInstance(
                timestamp_np=orig.timestamp_np,
                value_np=orig.value_np.copy(),
                label=(orig.label or STEERING_ANGLE) + " (original)",
                cpp_name=backup_name,
            )

        data[STEERING_ANGLE] = DataInstance(
            timestamp_np=raw_di.timestamp_np,
            value_np=recomputed,
            label="Steering angle recomputed from raw voltage (deg)",
            cpp_name=STEERING_ANGLE,
        )

        cal_str = ", ".join(f"({v:.2f}V, {a:+.1f}°)" for v, a in self.pts)
        print(
            f"correct_steering_angle: recomputed {STEERING_ANGLE} from {STEERING_RAW} "
            f"using deg-{self.deg} fit ({cal_str})\n"
        )
        return data


# This is the global callable that should actually be used in the preprocessing pipeline.
# Either pass it directly, or with a custom calibration correct_steering_angle(calibration=...)
# This should be treated like a function that supports partial application.
correct_steering_angle = CorrectSteeringAngleLambda()


def apply_preprocessing(
    data: SingleRunData,
    steps: list[PreprocessingStep],
) -> SingleRunData:
    """Run a sequence of preprocessing steps on a SingleRunData instance.

    Parameters
    ----------
    data : SingleRunData
        Parsed run data to preprocess.
    steps : list of PreprocessingStep
        Ordered list of ``SingleRunData -> SingleRunData`` callables to apply.

    Returns
    -------
    SingleRunData
        The same object, modified in-place and returned for chaining.

    Examples
    --------
    >>> apply_preprocessing(data, [correct_motor_data, convert_wheelspeeds_to_m_per_s])
    >>> apply_preprocessing(data, [
    ...     correct_motor_data,
    ...     correct_steering_angle(calibration=((1.5, -90.0), (3.0, 0.0), (4.5, 90.0))),
    ... ])
    """
    for step in steps:
        data = step(data)
    return data
