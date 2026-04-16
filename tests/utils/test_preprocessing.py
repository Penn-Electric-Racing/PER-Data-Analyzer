import numpy as np

from perda.units import in_to_m, mph_to_m_per_s
from perda.utils.preprocessing import *

# ---------------------------------------------------------------------------
# NED_VELOCITY
# ---------------------------------------------------------------------------


def test_ned_velocity_rotation_at_zero_yaw(ned_srd):
    # First point: velN=1, velE=0, yaw=0 → vx=1, vy=0
    result = apply_preprocessing(ned_srd, [Preprocessing.PATCH_NED_VELOCITY])
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.x"].value_np[0], 1.0)
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.y"].value_np[0], 0.0)


def test_ned_velocity_rotation_at_90_yaw(ned_srd):
    # Second point: velN=0, velE=1, yaw=90 → vx=1, vy=0
    result = apply_preprocessing(ned_srd, [Preprocessing.PATCH_NED_VELOCITY])
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.x"].value_np[1], 1.0)
    np.testing.assert_almost_equal(result["pcm.vnav.velocityBody.y"].value_np[1], 0.0)


def test_ned_velocity_skips_and_warns_when_missing(empty_srd, capsys):
    result = apply_preprocessing(empty_srd, [Preprocessing.PATCH_NED_VELOCITY])
    assert "velN" not in result
    captured = capsys.readouterr()
    assert "PATCH_NED_VELOCITY skipped" in captured.out


# ---------------------------------------------------------------------------
# WHEEL_SPEEDS_TO_M_PER_S
# ---------------------------------------------------------------------------


def test_wheelspeeds_converted_correctly(ws_srd):
    result = apply_preprocessing(ws_srd, [Preprocessing.WHEELSPEEDS_TO_M_PER_S])
    expected = mph_to_m_per_s(np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_almost_equal(
        result["pcm.wheelSpeeds.frontRight"].value_np, expected
    )


def test_wheelspeeds_backups_created(ws_srd):
    result = apply_preprocessing(ws_srd, [Preprocessing.WHEELSPEEDS_TO_M_PER_S])
    assert "pcm.wheelSpeeds.frontRight_mph" in result
    assert "pcm.wheelSpeeds.frontLeft_mph" in result
    assert "pcm.wheelSpeeds.rearRight_mph" in result
    assert "pcm.wheelSpeeds.rearLeft_mph" in result


def test_wheelspeeds_backup_values_are_original_mph(ws_srd):
    original = ws_srd["pcm.wheelSpeeds.frontRight"].value_np.copy()
    result = apply_preprocessing(ws_srd, [Preprocessing.WHEELSPEEDS_TO_M_PER_S])
    np.testing.assert_array_almost_equal(
        result["pcm.wheelSpeeds.frontRight_mph"].value_np, original
    )


def test_wheelspeeds_skips_and_warns_when_missing(empty_srd, capsys):
    result = apply_preprocessing(empty_srd, [Preprocessing.WHEELSPEEDS_TO_M_PER_S])
    assert "pcm.wheelSpeeds.frontRight_mph" not in result
    captured = capsys.readouterr()
    assert "WHEELSPEEDS_TO_M_PER_S skipped" in captured.out


# ---------------------------------------------------------------------------
# CORRECT_MOTOR_DATA
# ---------------------------------------------------------------------------


def test_motor_data_flips_rpm_sign(motor_srd):
    result = apply_preprocessing(motor_srd, [Preprocessing.CORRECT_MOTOR_DATA])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed"].value_np,
        [1000.0, 2000.0, 3000.0, 0.0],
    )


def test_motor_data_backup_is_original(motor_srd):
    original = motor_srd["pcm.moc.motor.angularSpeed"].value_np.copy()
    result = apply_preprocessing(motor_srd, [Preprocessing.CORRECT_MOTOR_DATA])
    np.testing.assert_array_almost_equal(
        result["pcm.moc.motor.angularSpeed_raw"].value_np, original
    )


def test_motor_data_wheel_speed_computed_correctly(motor_srd):
    result = apply_preprocessing(motor_srd, [Preprocessing.CORRECT_MOTOR_DATA])
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
    result = apply_preprocessing(empty_srd, [Preprocessing.CORRECT_MOTOR_DATA])
    assert "pcm.moc.motor.wheelSpeed" not in result
    captured = capsys.readouterr()
    assert "CORRECT_MOTOR_DATA skipped" in captured.out
