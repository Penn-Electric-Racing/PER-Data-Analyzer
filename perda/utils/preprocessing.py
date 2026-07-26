from typing import Callable, Union

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..core_data_structures.data_instance import DataInstance, left_join_data_instances
from ..core_data_structures.single_run_data import SingleRunData
from ..units import in_to_m, mph_to_m_per_s

DEFAULT_GEAR_RATIO: float = 5.6
DEFAULT_TIRE_RADIUS_IN: float = 7.85

DEFAULT_VECTORNAV_BODY_VEL_X = "pcm.vnav.velocityBody.x"
DEFAULT_VECTORNAV_BODY_VEL_Y = "pcm.vnav.velocityBody.y"
DEFAULT_VECTORNAV_BODY_VEL_Z = "pcm.vnav.velocityBody.z"
DEFAULT_VECTORNAV_YAW = "pcm.vnav.yawPitchRoll.yaw"
DEFAULT_VECTORNAV_NED_VEL_N = "velN"
DEFAULT_VECTORNAV_NED_VEL_E = "velE"
DEFAULT_VECTORNAV_NED_VEL_D = "velD"

DEFAULT_WHEELSPEED_FR = "pcm.wheelSpeeds.frontRight"
DEFAULT_WHEELSPEED_FL = "pcm.wheelSpeeds.frontLeft"
DEFAULT_WHEELSPEED_RR = "pcm.wheelSpeeds.backRight"
DEFAULT_WHEELSPEED_RL = "pcm.wheelSpeeds.backLeft"

DEFAULT_MOTOR_RPM = "pcm.moc.motor.angularSpeed"
DEFAULT_MOTOR_WHEELSPEED = "pcm.moc.motor.wheelSpeed"

DEFAULT_STEERING_RAW = "ludwig.steeringWheel.raw"
DEFAULT_STEERING_ANGLE = "ludwig.steeringWheel.angle"

# Default 3-point voltage->angle calibration for the steering pot.
SteeringCalibration = tuple[tuple[float, float], ...]
DEFAULT_STEERING_CALIBRATION: SteeringCalibration = (
    (1.86, -97.0),  # max left
    (2.93, 0.0),  # zero
    (3.96, 97.0),  # max right
)

# Type alias for a preprocessing step function that takes and returns a SingleRunData instance.
PreprocessingStep = Callable[[SingleRunData], SingleRunData]


class PatchNedVelocityLambda:
    """Backing class for the ``patch_ned_velocity`` preprocessing step."""

    def __init__(
        self,
        body_vel_x: str = DEFAULT_VECTORNAV_BODY_VEL_X,
        body_vel_y: str = DEFAULT_VECTORNAV_BODY_VEL_Y,
        body_vel_z: str = DEFAULT_VECTORNAV_BODY_VEL_Z,
        yaw: str = DEFAULT_VECTORNAV_YAW,
        ned_vel_n: str = DEFAULT_VECTORNAV_NED_VEL_N,
        ned_vel_e: str = DEFAULT_VECTORNAV_NED_VEL_E,
        ned_vel_d: str = DEFAULT_VECTORNAV_NED_VEL_D,
    ) -> None:
        self.body_vel_x = body_vel_x
        self.body_vel_y = body_vel_y
        self.body_vel_z = body_vel_z
        self.yaw = yaw
        self.ned_vel_n = ned_vel_n
        self.ned_vel_e = ned_vel_e
        self.ned_vel_d = ned_vel_d

    def __call__(
        self,
        data: SingleRunData | None = None,
        *,
        body_vel_x: str | None = None,
        body_vel_y: str | None = None,
        body_vel_z: str | None = None,
        yaw: str | None = None,
        ned_vel_n: str | None = None,
        ned_vel_e: str | None = None,
        ned_vel_d: str | None = None,
    ) -> Union[SingleRunData, "PatchNedVelocityLambda"]:
        """Apply the step to ``data``, or return a reconfigured copy when called with keyword args.

        Parameters
        ----------
        data : SingleRunData, optional
            Run data to transform in-place. Omit when reconfiguring.
        body_vel_x : str, optional
            Channel incorrectly holding NED north; overwritten with body-frame forward velocity.
        body_vel_y : str, optional
            Channel incorrectly holding NED east; overwritten with body-frame right velocity.
        body_vel_z : str, optional
            Channel incorrectly holding NED down; left unchanged (NED down == FRD down).
        yaw : str, optional
            Yaw channel in degrees used to rotate NED -> FRD.
        ned_vel_n : str, optional
            Output channel name for the preserved raw NED north velocity.
        ned_vel_e : str, optional
            Output channel name for the preserved raw NED east velocity.
        ned_vel_d : str, optional
            Output channel name for the preserved raw NED down velocity.

        Returns
        -------
        SingleRunData or PatchNedVelocityLambda
            Transformed data when ``data`` is passed; reconfigured step otherwise.
        """
        if any(
            v is not None
            for v in (
                body_vel_x,
                body_vel_y,
                body_vel_z,
                yaw,
                ned_vel_n,
                ned_vel_e,
                ned_vel_d,
            )
        ):
            return PatchNedVelocityLambda(
                body_vel_x=body_vel_x or self.body_vel_x,
                body_vel_y=body_vel_y or self.body_vel_y,
                body_vel_z=body_vel_z or self.body_vel_z,
                yaw=yaw or self.yaw,
                ned_vel_n=ned_vel_n or self.ned_vel_n,
                ned_vel_e=ned_vel_e or self.ned_vel_e,
                ned_vel_d=ned_vel_d or self.ned_vel_d,
            )

        assert data is not None
        required = [self.body_vel_x, self.body_vel_y, self.body_vel_z, self.yaw]
        missing = [v for v in required if v not in data]
        if missing:
            print(f"WARNING: patch_ned_velocity skipped — missing variables: {missing}")
            return data

        vel_n1, vel_e1, vel_d1, yaw_deg = left_join_data_instances(
            data[self.body_vel_x],
            [data[self.body_vel_y], data[self.body_vel_z], data[self.yaw]],
        )

        yaw_rad = np.radians(yaw_deg.value_np)

        data[self.ned_vel_n] = DataInstance(
            timestamp_np=vel_n1.timestamp_np,
            value_np=vel_n1.value_np.copy(),
            label="NED North velocity (raw)",
            cpp_name=self.ned_vel_n,
        )
        data[self.ned_vel_e] = DataInstance(
            timestamp_np=vel_e1.timestamp_np,
            value_np=vel_e1.value_np.copy(),
            label="NED East velocity (raw)",
            cpp_name=self.ned_vel_e,
        )
        data[self.ned_vel_d] = DataInstance(
            timestamp_np=vel_d1.timestamp_np,
            value_np=vel_d1.value_np.copy(),
            label="NED Down velocity (raw)",
            cpp_name=self.ned_vel_d,
        )

        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        data[self.body_vel_x] = DataInstance(
            timestamp_np=vel_n1.timestamp_np,
            value_np=vel_n1.value_np * cos_y + vel_e1.value_np * sin_y,
            label=data[self.body_vel_x].label,
            cpp_name=self.body_vel_x,
        )  # forward
        data[self.body_vel_y] = DataInstance(
            timestamp_np=vel_e1.timestamp_np,
            value_np=-vel_n1.value_np * sin_y + vel_e1.value_np * cos_y,
            label=data[self.body_vel_y].label,
            cpp_name=self.body_vel_y,
        )  # right
        # vel_z (down) is identical in NED and FRD — no change needed

        print(
            f"patch_ned_velocity: preserved raw NED in {self.ned_vel_n}/{self.ned_vel_e}/{self.ned_vel_d}, rotated {len(vel_n1)} points to body frame"
        )
        return data


class ConvertWheelspeedsToMPerSLambda:
    """Backing class for the ``convert_wheelspeeds_to_m_per_s`` preprocessing step."""

    def __init__(
        self,
        wheelspeed_fr: str = DEFAULT_WHEELSPEED_FR,
        wheelspeed_fl: str = DEFAULT_WHEELSPEED_FL,
        wheelspeed_rr: str = DEFAULT_WHEELSPEED_RR,
        wheelspeed_rl: str = DEFAULT_WHEELSPEED_RL,
    ) -> None:
        self.wheelspeed_fr = wheelspeed_fr
        self.wheelspeed_fl = wheelspeed_fl
        self.wheelspeed_rr = wheelspeed_rr
        self.wheelspeed_rl = wheelspeed_rl

    def __call__(
        self,
        data: SingleRunData | None = None,
        *,
        wheelspeed_fr: str | None = None,
        wheelspeed_fl: str | None = None,
        wheelspeed_rr: str | None = None,
        wheelspeed_rl: str | None = None,
    ) -> Union[SingleRunData, "ConvertWheelspeedsToMPerSLambda"]:
        """Apply the step to ``data``, or return a reconfigured copy when called with keyword args.

        Parameters
        ----------
        data : SingleRunData, optional
            Run data to transform in-place. Omit when reconfiguring.
        wheelspeed_fr : str, optional
            Variable name for the front-right wheel speed channel (expected in mph).
        wheelspeed_fl : str, optional
            Variable name for the front-left wheel speed channel (expected in mph).
        wheelspeed_rr : str, optional
            Variable name for the rear-right wheel speed channel (expected in mph).
        wheelspeed_rl : str, optional
            Variable name for the rear-left wheel speed channel (expected in mph).

        Returns
        -------
        SingleRunData or ConvertWheelspeedsToMPerSLambda
            Transformed data when ``data`` is passed; reconfigured step otherwise.
        """
        if any(
            v is not None
            for v in (wheelspeed_fr, wheelspeed_fl, wheelspeed_rr, wheelspeed_rl)
        ):
            return ConvertWheelspeedsToMPerSLambda(
                wheelspeed_fr=wheelspeed_fr or self.wheelspeed_fr,
                wheelspeed_fl=wheelspeed_fl or self.wheelspeed_fl,
                wheelspeed_rr=wheelspeed_rr or self.wheelspeed_rr,
                wheelspeed_rl=wheelspeed_rl or self.wheelspeed_rl,
            )

        assert data is not None
        cols = [
            self.wheelspeed_fr,
            self.wheelspeed_fl,
            self.wheelspeed_rr,
            self.wheelspeed_rl,
        ]
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


class CorrectMotorDataLambda:
    """Backing class for the ``correct_motor_data`` preprocessing step."""

    def __init__(
        self,
        gear_ratio: float = DEFAULT_GEAR_RATIO,
        tire_radius_in: float = DEFAULT_TIRE_RADIUS_IN,
        motor_rpm: str = DEFAULT_MOTOR_RPM,
        motor_wheelspeed: str = DEFAULT_MOTOR_WHEELSPEED,
    ) -> None:
        self.gear_ratio = gear_ratio
        self.tire_radius_in = tire_radius_in
        self.motor_rpm = motor_rpm
        self.motor_wheelspeed = motor_wheelspeed

    def __call__(
        self,
        data: SingleRunData | None = None,
        *,
        gear_ratio: float | None = None,
        tire_radius_in: float | None = None,
        motor_rpm: str | None = None,
        motor_wheelspeed: str | None = None,
    ) -> Union[SingleRunData, "CorrectMotorDataLambda"]:
        """Apply the step to ``data``, or return a reconfigured copy when called with keyword args.

        Parameters
        ----------
        data : SingleRunData, optional
            Run data to transform in-place. Omit when reconfiguring.
        gear_ratio : float, optional
            Final-drive gear ratio.
        tire_radius_in : float, optional
            Loaded tire radius in inches.
        motor_rpm : str, optional
            Variable name for the motor angular speed channel.
        motor_wheelspeed : str, optional
            Variable name to write the derived driven-wheel linear speed (m/s).

        Returns
        -------
        SingleRunData or CorrectMotorDataLambda
            Transformed data when ``data`` is passed; reconfigured step otherwise.
        """
        if any(
            v is not None
            for v in (gear_ratio, tire_radius_in, motor_rpm, motor_wheelspeed)
        ):
            return CorrectMotorDataLambda(
                gear_ratio=gear_ratio if gear_ratio is not None else self.gear_ratio,
                tire_radius_in=(
                    tire_radius_in
                    if tire_radius_in is not None
                    else self.tire_radius_in
                ),
                motor_rpm=motor_rpm or self.motor_rpm,
                motor_wheelspeed=motor_wheelspeed or self.motor_wheelspeed,
            )

        assert data is not None
        if self.motor_rpm not in data:
            print(
                f"WARNING: correct_motor_data skipped — missing variable: {self.motor_rpm}"
            )
            return data

        di = data[self.motor_rpm]
        raw_rpm: NDArray[float64] = di.value_np.astype(np.float64)

        backup_name = self.motor_rpm + "_raw"
        if backup_name not in data:
            data[backup_name] = DataInstance(
                timestamp_np=di.timestamp_np,
                value_np=raw_rpm.copy(),
                label="Motor RPM raw (pre-flip)",
                cpp_name=backup_name,
            )

        flipped = -raw_rpm
        data[self.motor_rpm] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=flipped,
            label=di.label,
            cpp_name=self.motor_rpm,
        )

        tire_radius_m = in_to_m(self.tire_radius_in)
        wheel_speed: NDArray[float64] = (
            flipped * 2.0 * np.pi * tire_radius_m / (60.0 * self.gear_ratio)
        )
        data[self.motor_wheelspeed] = DataInstance(
            timestamp_np=di.timestamp_np,
            value_np=wheel_speed,
            label="Driven wheel speed from motor RPM (m/s)",
            cpp_name=self.motor_wheelspeed,
        )

        print(
            f"correct_motor_data: flipped {self.motor_rpm} sign, added {self.motor_wheelspeed} "
            f"(ratio={self.gear_ratio}, r={self.tire_radius_in} in)"
        )
        return data


class CorrectSteeringAngleLambda:
    """
    Backing class for the ``correct_steering_angle`` preprocessing step.

    Fits a polynomial through ``calibration`` points (voltage, angle) at construction time,
    then rewrites the angle channel from the raw voltage channel on each call.
    """

    def __init__(
        self,
        calibration: SteeringCalibration = DEFAULT_STEERING_CALIBRATION,
        steering_raw: str = DEFAULT_STEERING_RAW,
        steering_angle: str = DEFAULT_STEERING_ANGLE,
    ) -> None:
        pts = sorted(calibration, key=lambda p: p[0])
        if len(pts) < 2:
            raise ValueError("calibration needs at least 2 (voltage, angle) points.")
        self.pts = pts
        self.steering_raw = steering_raw
        self.steering_angle = steering_angle

        volts: NDArray[float64] = np.array([p[0] for p in pts], dtype=np.float64)
        angles: NDArray[float64] = np.array([p[1] for p in pts], dtype=np.float64)
        self.deg = 2 if len(pts) >= 3 else 1
        self.coeffs: NDArray[float64] = np.polyfit(volts, angles, self.deg)

    def __call__(
        self,
        data: SingleRunData | None = None,
        *,
        calibration: SteeringCalibration | None = None,
        steering_raw: str | None = None,
        steering_angle: str | None = None,
    ) -> Union[SingleRunData, "CorrectSteeringAngleLambda"]:
        """Apply the step to ``data``, or return a reconfigured copy when called with keyword args.

        Parameters
        ----------
        data : SingleRunData, optional
            Run data to transform in-place. Omit when reconfiguring.
        calibration : SteeringCalibration, optional
            Sequence of ``(voltage, angle_deg)`` pairs. A degree-2 polynomial is fit when
            >= 3 points are provided, linear otherwise.
        steering_raw : str, optional
            Variable name for the raw potentiometer voltage channel.
        steering_angle : str, optional
            Variable name to write the recomputed angle (degrees). If this channel already
            exists, its old values are backed up under ``<steering_angle>_original``.

        Returns
        -------
        SingleRunData or CorrectSteeringAngleLambda
            Transformed data when ``data`` is passed; reconfigured step otherwise.
        """
        if any(v is not None for v in (calibration, steering_raw, steering_angle)):
            return CorrectSteeringAngleLambda(
                calibration=calibration or tuple(self.pts),
                steering_raw=steering_raw or self.steering_raw,
                steering_angle=steering_angle or self.steering_angle,
            )

        assert data is not None
        if self.steering_raw not in data:
            print(
                f"WARNING: correct_steering_angle skipped — missing variable: {self.steering_raw}"
            )
            return data

        raw_di = data[self.steering_raw]
        raw_volts: NDArray[float64] = raw_di.value_np.astype(np.float64)
        recomputed: NDArray[float64] = np.polyval(self.coeffs, raw_volts).astype(
            np.float64
        )

        backup_name = self.steering_angle + "_original"
        if self.steering_angle in data and backup_name not in data:
            orig = data[self.steering_angle]
            data[backup_name] = DataInstance(
                timestamp_np=orig.timestamp_np,
                value_np=orig.value_np.copy(),
                label=(orig.label or self.steering_angle) + " (original)",
                cpp_name=backup_name,
            )

        data[self.steering_angle] = DataInstance(
            timestamp_np=raw_di.timestamp_np,
            value_np=recomputed,
            label="Steering angle recomputed from raw voltage (deg)",
            cpp_name=self.steering_angle,
        )

        cal_str = ", ".join(f"({v:.2f}V, {a:+.1f}°)" for v, a in self.pts)
        print(
            f"correct_steering_angle: recomputed {self.steering_angle} from {self.steering_raw} "
            f"using deg-{self.deg} fit ({cal_str})\n"
        )
        return data


patch_ned_velocity = PatchNedVelocityLambda()
"""Preprocessing step: fix VectorNav NED-in-body-frame bug.

Corrects a firmware bug where ``velocityBody.x/y/z`` contains NED-frame velocities instead of
body-frame (FRD) velocities. The raw NED values are preserved as new channels (``velN``,
``velE``, ``velD``), and the body velocity channels are overwritten with the correctly rotated
values using the yaw angle.

Use directly in a preprocessing pipeline, or call with keyword arguments to get a reconfigured
copy that operates on non-default variable names.

Examples
--------
>>> Analyzer(preprocessing=[patch_ned_velocity])
>>> Analyzer(preprocessing=[patch_ned_velocity(yaw="my.custom.yaw")])
>>> Analyzer(preprocessing=[patch_ned_velocity(body_vel_x="my.vel.x", ned_vel_n="my.ned.n")])
"""

convert_wheelspeeds_to_m_per_s = ConvertWheelspeedsToMPerSLambda()
"""Preprocessing step: convert all four wheel speed channels from mph to m/s.

Each channel is converted in-place; the original mph values are preserved as
``<channel_name>_mph`` backup channels (only created once — idempotent on repeated calls).

Use directly in a preprocessing pipeline, or call with keyword arguments to get a reconfigured
copy that operates on non-default variable names.

Examples
--------
>>> Analyzer(preprocessing=[convert_wheelspeeds_to_m_per_s])
>>> Analyzer(preprocessing=[convert_wheelspeeds_to_m_per_s(wheelspeed_fr="my.ws.fr")])
"""

correct_motor_data = CorrectMotorDataLambda()
"""Preprocessing step: flip motor RPM sign and derive driven wheel speed.

The motor reports RPM with the sign inverted (negative when driving forward). This step:

1. Preserves the original RPM as ``<motor_rpm>_raw``.
2. Flips the sign of the RPM channel in-place.
3. Derives a driven wheel linear speed (m/s) from RPM, gear ratio, and tire radius,
   and writes it to ``motor_wheelspeed``.

Use directly in a preprocessing pipeline, or call with keyword arguments to get a reconfigured
copy with different physical constants or variable names.

Examples
--------
>>> Analyzer(preprocessing=[correct_motor_data])
>>> Analyzer(preprocessing=[correct_motor_data(gear_ratio=6.2, tire_radius_in=8.0)])
"""

correct_steering_angle = CorrectSteeringAngleLambda()
"""Preprocessing step: recompute steering angle from raw potentiometer voltage.

Fits a polynomial through a set of ``(voltage, angle_deg)`` calibration points, then evaluates
it on the raw voltage channel to produce a corrected angle. This corrects for sensor drift
relative to the previously stored angle channel.

If the angle channel already exists, the old values are preserved as ``<steering_angle>_original``
before being overwritten. The backup is only created once — idempotent on repeated calls.

Use directly in a preprocessing pipeline, or call with keyword arguments to get a reconfigured
copy.

Examples
--------
>>> Analyzer(preprocessing=[correct_steering_angle])
>>> Analyzer(preprocessing=[correct_steering_angle(calibration=((1.5, -90.0), (3.0, 0.0), (4.5, 90.0)))])
>>> Analyzer(preprocessing=[correct_steering_angle(steering_raw="my.raw")])
>>> Analyzer(preprocessing=[correct_steering_angle(calibration=my_cal, steering_angle="my.angle")])
"""


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
    ...     correct_motor_data(gear_ratio=6.2, tire_radius_in=8.0),
    ...     correct_steering_angle(((1.5, -90.0), (3.0, 0.0), (4.5, 90.0))),
    ... ])
    """
    for step in steps:
        data = step(data)
    return data
