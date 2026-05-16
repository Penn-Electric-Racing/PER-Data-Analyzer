import numpy as np
import pytest

from perda.units import in_to_m, mph_to_m_per_s
from perda.utils.preprocessing import *


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
    assert "pcm.wheelSpeeds.rearRight_mph" in result
    assert "pcm.wheelSpeeds.rearLeft_mph" in result


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


def test_motor_data_backup_is_original(motor_srd):
    original = motor_srd["pcm.moc.motor.angularSpeed"].value_np.copy()
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed_raw"].value_np, original
    )


def test_motor_data_wheel_speed_computed_correctly(motor_srd):
    result = apply_preprocessing(motor_srd, [correct_motor_data])
    assert "pcm.moc.motor.wheelSpeed" in result
    tire_r_m = in_to_m(TIRE_RADIUS_IN)
    expected = (
        np.array([1000.0, 2000.0, 3000.0, 0.0])
        * 2.0
        * np.pi
        * tire_r_m
        / (60.0 * GEAR_RATIO)
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
    step = correct_steering_angle(calibration)
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
    custom = correct_steering_angle(((1.0, -180.0), (3.0, 0.0), (5.0, 180.0)))
    assert callable(custom)
    assert custom is not correct_steering_angle
