import numpy as np
import pytest

from perda.core_data_structures.resampling import ResampleMethod, _interpolate

_ALL_METHODS = [
    ResampleMethod.LINEAR,
    ResampleMethod.ZOH,
    ResampleMethod.NEAREST,
    ResampleMethod.CUBIC,
]


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_interpolate_clamps_before_first(method):
    src_t = np.array([5.0, 10.0])
    src_v = np.array([50.0, 100.0])
    result = _interpolate(np.array([0.0]), src_t, src_v, method)
    np.testing.assert_allclose(result, [50.0])


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_interpolate_clamps_after_last(method):
    src_t = np.array([0.0, 5.0])
    src_v = np.array([10.0, 20.0])
    result = _interpolate(np.array([100.0]), src_t, src_v, method)
    np.testing.assert_allclose(result, [20.0])


@pytest.mark.parametrize(
    "method",
    [ResampleMethod.LINEAR, ResampleMethod.NEAREST, ResampleMethod.CUBIC],
)
def test_interpolate_exact_at_knots(method):
    src_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    src_v = src_t**2
    result = _interpolate(src_t, src_t, src_v, method)
    np.testing.assert_allclose(result, src_v, atol=1e-6)


def test_interpolate_linear_interior():
    src_t = np.array([0.0, 10.0])
    src_v = np.array([0.0, 10.0])
    result = _interpolate(np.array([5.0]), src_t, src_v, ResampleMethod.LINEAR)
    np.testing.assert_allclose(result, [5.0])


def test_interpolate_linear_multiple_points():
    src_t = np.array([0.0, 10.0])
    src_v = np.array([0.0, 100.0])
    targets = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    result = _interpolate(targets, src_t, src_v, ResampleMethod.LINEAR)
    np.testing.assert_allclose(result, [0.0, 25.0, 50.0, 75.0, 100.0])


def test_interpolate_zoh_holds_previous_value():
    src_t = np.array([0.0, 4.0, 8.0])
    src_v = np.array([10.0, 20.0, 30.0])
    result = _interpolate(np.array([1.0, 3.0]), src_t, src_v, ResampleMethod.ZOH)
    np.testing.assert_allclose(result, [10.0, 10.0])


def test_interpolate_zoh_at_knot_returns_knot_value():
    src_t = np.array([0.0, 5.0, 10.0])
    src_v = np.array([1.0, 2.0, 3.0])
    result = _interpolate(np.array([5.0]), src_t, src_v, ResampleMethod.ZOH)
    np.testing.assert_allclose(result, [2.0])


def test_interpolate_nearest_close_to_knot():
    src_t = np.array([0.0, 10.0])
    src_v = np.array([0.0, 100.0])
    # 1.0 is closer to 0.0 than to 10.0
    result = _interpolate(np.array([1.0]), src_t, src_v, ResampleMethod.NEAREST)
    np.testing.assert_allclose(result, [0.0])


def test_interpolate_cubic_fallback_fewer_than_4_points():
    src_t = np.array([0.0, 5.0, 10.0])
    src_v = np.array([0.0, 5.0, 10.0])
    result = _interpolate(np.array([2.5, 7.5]), src_t, src_v, ResampleMethod.CUBIC)
    np.testing.assert_allclose(result, [2.5, 7.5])


def test_interpolate_cubic_interior_smooth():
    src_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    src_v = src_t * 2.0
    result = _interpolate(np.array([0.5, 1.5, 2.5]), src_t, src_v, ResampleMethod.CUBIC)
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0], atol=1e-6)
