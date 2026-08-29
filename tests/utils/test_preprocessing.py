import numpy as np
import pytest

from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale, in_to_m, mph_to_m_per_s
from perda.utils.preprocessing import (
    DEFAULT_GEAR_RATIO,
    DEFAULT_MOTOR_WHEELSPEED,
    DEFAULT_STEERING_CALIBRATION,
    DEFAULT_TIRE_RADIUS_IN,
    DEFAULT_WHEELSPEED_FL,
    DEFAULT_WHEELSPEED_RL,
    DEFAULT_WHEELSPEED_RR,
    apply_preprocessing,
    convert_wheelspeeds_to_m_per_s,
    correct_motor_data,
    correct_steering_angle,
    patch_ned_velocity,
)


def test_ned_velocity_rotation_at_zero_yaw(ned_srd):
    # First point: velN=1, velE=0, yaw=0 -> vx=1, vy=0
    result = apply_preprocessing(ned_srd, [patch_ned_velocity])
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.x"].value_np[0], 1.0)
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.y"].value_np[0], 0.0)


def test_ned_velocity_rotation_at_90_yaw(ned_srd):
    # Second point: velN=0, velE=1, yaw=90 -> vx=1, vy=0
    result = apply_preprocessing(ned_srd, [patch_ned_velocity])
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.x"].value_np[1], 1.0)
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.y"].value_np[1], 0.0)


def test_ned_velocity_skips_and_warns_when_missing(empty_srd, capsys):
    result = apply_preprocessing(empty_srd, [patch_ned_velocity])
    assert "velN" not in result
    captured = capsys.readouterr()
    assert "patch_ned_velocity skipped" in captured.out


def test_wheelspeeds_converted_correctly(ws_srd):
    result = apply_preprocessing(ws_srd, [convert_wheelspeeds_to_m_per_s])
    expected = mph_to_m_per_s(np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_almost_equal(
        result["pcm.wheelSpeeds.frontRight"].value_np, expected
    )


def test_wheelspeeds_backups_created(ws_srd):
    result = apply_preprocessing(ws_srd, [convert_wheelspeeds_to_m_per_s])
    assert "pcm.wheelSpeeds.frontRight_mph" in result
    assert "pcm.wheelSpeeds.frontLeft_mph" in result
    assert "pcm.wheelSpeeds.backRight_mph" in result
    assert "pcm.wheelSpeeds.backLeft_mph" in result


def test_wheelspeeds_backup_values_are_original_mph(ws_srd):
    original = ws_srd["pcm.wheelSpeeds.frontRight"].value_np.copy()
    result = apply_preprocessing(ws_srd, [convert_wheelspeeds_to_m_per_s])
    np.testing.assert_array_almost_equal(
        result["pcm.wheelSpeeds.frontRight_mph"].value_np, original
    )


def test_wheelspeeds_skips_and_warns_when_missing(empty_srd, capsys):
    result = apply_preprocessing(empty_srd, [convert_wheelspeeds_to_m_per_s])
    assert "pcm.wheelSpeeds.frontRight_mph" not in result
    captured = capsys.readouterr()
    assert "convert_wheelspeeds_to_m_per_s skipped" in captured.out


def test_motor_data_flips_rpm_sign(motor_srd):
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed"].value_np,
        [1000.0, 2000.0, 3000.0, 0.0],
    )


def test_motor_data_keeps_sign_when_rpm_already_positive(motor_srd):
    motor_srd["pcm.moc.motor.angularSpeed"] = -motor_srd["pcm.moc.motor.angularSpeed"]
    already_positive = motor_srd["pcm.moc.motor.angularSpeed"].value_np.copy()
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed"].value_np, already_positive
    )


def test_motor_data_auto_inversion_ignores_negative_outlier(motor_srd):
    rpm = motor_srd["pcm.moc.motor.angularSpeed"]
    motor_srd["pcm.moc.motor.angularSpeed"] = DataInstance(
        timestamp_np=rpm.timestamp_np,
        value_np=np.array([1000.0, 2000.0, 3000.0, -99000.0]),
        label=rpm.label,
        cpp_name="pcm.moc.motor.angularSpeed",
    )
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed"].value_np,
        [1000.0, 2000.0, 3000.0, -99000.0],
    )


def test_motor_data_inversion_can_be_forced_off(motor_srd):
    original = motor_srd["pcm.moc.motor.angularSpeed"].value_np.copy()
    result = correct_motor_data(invert_rpm=False)(motor_srd)
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed"].value_np, original
    )


def test_motor_data_backup_is_original(motor_srd):
    original = motor_srd["pcm.moc.motor.angularSpeed"].value_np.copy()
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed_raw"].value_np, original
    )


def test_motor_data_wheel_speed_computed_correctly(motor_srd):
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    assert "pcm.moc.motor.wheelSpeed" in result
    tire_r_m = in_to_m(DEFAULT_TIRE_RADIUS_IN)
    expected = (
        np.array([1000.0, 2000.0, 3000.0, 0.0])
        * 2.0
        * np.pi
        * tire_r_m
        / (60.0 * DEFAULT_GEAR_RATIO)
    )
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.wheelSpeed"].value_np, expected
    )


def test_motor_data_skips_and_warns_when_missing(empty_srd, capsys):
    result = apply_preprocessing(empty_srd, [correct_motor_data])
    assert "pcm.moc.motor.wheelSpeed" not in result
    captured = capsys.readouterr()
    assert "correct_motor_data skipped" in captured.out


@pytest.mark.parametrize(
    "calibration, input_volts, expected_angles",
    [
        (
            DEFAULT_STEERING_CALIBRATION,  # ((1.86, -97), (2.93, 0), (3.96, 97))
            [1.86, 2.93, 3.96],
            [-97.0, 0.0, 97.0],
        ),
        (
            ((1.0, -180.0), (3.0, 0.0), (5.0, 180.0)),
            [1.0, 3.0, 5.0],
            [-180.0, 0.0, 180.0],
        ),
    ],
)
def test_steering_angle_correctness_at_calibration_points(
    calibration, input_volts, expected_angles, steering_srd_no_angle
):
    step = correct_steering_angle(calibration=calibration)
    # Overwrite the raw channel with the test voltages
    steering_srd_no_angle["ludwig.steeringWheel.raw"].value_np[:] = input_volts

    result = step(steering_srd_no_angle)

    np.testing.assert_array_almost_equal(
        result["ludwig.steeringWheel.angle"].value_np[:3],
        expected_angles,
        decimal=6,
    )


def test_steering_angle_overwrites_existing_angle(steering_srd):
    result = correct_steering_angle(steering_srd)

    np.testing.assert_almost_equal(
        result["ludwig.steeringWheel.angle"].value_np[0], -97.0, decimal=6
    )
    np.testing.assert_almost_equal(
        result["ludwig.steeringWheel.angle"].value_np[1], 0.0, decimal=6
    )
    np.testing.assert_almost_equal(
        result["ludwig.steeringWheel.angle"].value_np[2], 97.0, decimal=6
    )


def test_steering_angle_preserves_original_as_backup(steering_srd):
    stale_before = steering_srd["ludwig.steeringWheel.angle"].value_np.copy()
    result = correct_steering_angle(steering_srd)

    assert "ludwig.steeringWheel.angle_original" in result
    np.testing.assert_array_equal(
        result["ludwig.steeringWheel.angle_original"].value_np, stale_before
    )


def test_steering_angle_creates_angle_when_absent(steering_srd_no_angle):
    assert "ludwig.steeringWheel.angle" not in steering_srd_no_angle
    result = correct_steering_angle(steering_srd_no_angle)
    assert "ludwig.steeringWheel.angle" in result


def test_steering_angle_skips_and_warns_when_missing(empty_srd, capsys):
    result = correct_steering_angle(empty_srd)
    assert "ludwig.steeringWheel.angle" not in result
    captured = capsys.readouterr()
    assert "correct_steering_angle skipped" in captured.out


def test_steering_angle_partial_application_returns_new_step():
    custom = correct_steering_angle(
        calibration=((1.0, -180.0), (3.0, 0.0), (5.0, 180.0))
    )
    assert callable(custom)
    assert custom is not correct_steering_angle


@pytest.mark.parametrize(
    "step_factory, kwargs",
    [
        (lambda: patch_ned_velocity, {"yaw": "custom.yaw"}),
        (lambda: convert_wheelspeeds_to_m_per_s, {"wheelspeed_fr": "custom.fr"}),
        (lambda: correct_motor_data, {"gear_ratio": 6.2}),
        (lambda: correct_steering_angle, {"steering_raw": "custom.raw"}),
    ],
)
def test_partial_application_returns_new_callable(step_factory, kwargs):
    step = step_factory()
    reconfigured = step(**kwargs)
    assert callable(reconfigured)
    assert reconfigured is not step


def test_ned_velocity_partial_application_uses_custom_variable_names(empty_srd):
    custom_yaw = "my.yaw"
    reconfigured = patch_ned_velocity(
        body_vel_x="my.vel.x",
        body_vel_y="my.vel.y",
        body_vel_z="my.vel.z",
        yaw=custom_yaw,
        ned_vel_n="my.ned.n",
        ned_vel_e="my.ned.e",
        ned_vel_d="my.ned.d",
    )
    # missing-variable warning should name the custom variable, not the default
    reconfigured(empty_srd)
    assert reconfigured.yaw == custom_yaw
    assert reconfigured.ned_vel_n == "my.ned.n"


def test_ned_velocity_partial_application_applies_correctly():
    """Reconfigured step writes NED outputs to the overridden variable names."""
    ts = np.arange(4, dtype=np.int64)
    custom_srd = SingleRunData(
        id_to_instance={
            1: DataInstance(
                timestamp_np=ts,
                value_np=np.array([1.0, 0.0, -1.0, 0.0]),
                label="x",
                var_id=1,
                cpp_name="my.vel.x",
            ),
            2: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.0, 1.0, 0.0, -1.0]),
                label="y",
                var_id=2,
                cpp_name="my.vel.y",
            ),
            3: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.1, 0.1, 0.1, 0.1]),
                label="z",
                var_id=3,
                cpp_name="my.vel.z",
            ),
            4: DataInstance(
                timestamp_np=ts,
                value_np=np.array([0.0, 90.0, 180.0, 270.0]),
                label="yaw",
                var_id=4,
                cpp_name="my.yaw",
            ),
        },
        cpp_name_to_id={"my.vel.x": 1, "my.vel.y": 2, "my.vel.z": 3, "my.yaw": 4},
        id_to_cpp_name={1: "my.vel.x", 2: "my.vel.y", 3: "my.vel.z", 4: "my.yaw"},
        id_to_descript={1: "", 2: "", 3: "", 4: ""},
        total_data_points=16,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )
    reconfigured = patch_ned_velocity(
        body_vel_x="my.vel.x",
        body_vel_y="my.vel.y",
        body_vel_z="my.vel.z",
        yaw="my.yaw",
        ned_vel_n="my.ned.n",
        ned_vel_e="my.ned.e",
        ned_vel_d="my.ned.d",
    )
    result = reconfigured(custom_srd)
    assert "my.ned.n" in result
    assert "my.ned.e" in result
    np.testing.assert_almost_equal(result["my.vel.x"].value_np[0], 1.0)


def test_wheelspeeds_partial_application_uses_custom_variable_names():
    """Reconfigured step with one channel renamed converts and backs up that channel."""
    ts = np.arange(4, dtype=np.int64)
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    names = [
        "my.ws.fr",
        DEFAULT_WHEELSPEED_FL,
        DEFAULT_WHEELSPEED_RR,
        DEFAULT_WHEELSPEED_RL,
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
    custom_srd = SingleRunData(
        id_to_instance=instances,
        cpp_name_to_id={name: i + 1 for i, name in enumerate(names)},
        id_to_cpp_name={i + 1: name for i, name in enumerate(names)},
        id_to_descript={i + 1: "" for i in range(len(names))},
        total_data_points=16,
        data_start_time=0,
        data_end_time=3,
        timestamp_unit=Timescale.MS,
    )

    reconfigured = convert_wheelspeeds_to_m_per_s(wheelspeed_fr="my.ws.fr")
    result = reconfigured(custom_srd)

    assert "my.ws.fr_mph" in result
    np.testing.assert_array_almost_equal(
        result["my.ws.fr"].value_np, mph_to_m_per_s(vals)
    )


def test_motor_data_partial_application_uses_custom_gear_ratio(motor_srd):
    custom_ratio = 4.0
    reconfigured = correct_motor_data(gear_ratio=custom_ratio)
    result = reconfigured(motor_srd)

    tire_r_m = in_to_m(DEFAULT_TIRE_RADIUS_IN)
    expected = (
        np.array([1000.0, 2000.0, 3000.0, 0.0])
        * 2.0
        * np.pi
        * tire_r_m
        / (60.0 * custom_ratio)
    )
    np.testing.assert_array_almost_equal(
        result[DEFAULT_MOTOR_WHEELSPEED].value_np, expected
    )


def test_motor_data_partial_application_uses_custom_tire_radius(motor_srd):
    custom_radius = 9.0
    reconfigured = correct_motor_data(tire_radius_in=custom_radius)
    result = reconfigured(motor_srd)

    tire_r_m = in_to_m(custom_radius)
    expected = (
        np.array([1000.0, 2000.0, 3000.0, 0.0])
        * 2.0
        * np.pi
        * tire_r_m
        / (60.0 * DEFAULT_GEAR_RATIO)
    )
    np.testing.assert_array_almost_equal(
        result[DEFAULT_MOTOR_WHEELSPEED].value_np, expected
    )


def test_steering_angle_kwargs_partial_application(steering_srd):
    """Reconfigured step via kwargs writes to the overridden angle channel name."""
    custom_angle = "my.steering.angle"
    reconfigured = correct_steering_angle(steering_angle=custom_angle)
    result = reconfigured(steering_srd)

    assert custom_angle in result
    np.testing.assert_almost_equal(result[custom_angle].value_np[0], -97.0, decimal=6)
