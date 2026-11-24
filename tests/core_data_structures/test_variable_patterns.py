import pytest

from perda.core_data_structures.variable_patterns import is_pattern, match_pattern

NAMES = [
    "pcm.vnav.rpy.yaw",
    "pcm.vnav.rpy.roll",
    "pcm.vnav.bodyVelocity.x",
    "ludwig.steeringWheel.angle",
    "pcm_vnav_rpy_pitch",
]


@pytest.mark.parametrize(
    "name, expected",
    [
        pytest.param("pcm.vnav.*", True, id="wildcard"),
        pytest.param("pcm.vnav.rpy.yaw", False, id="plain"),
    ],
)
def test_is_pattern(name, expected):
    assert is_pattern(name) is expected


@pytest.mark.parametrize(
    "pattern, expected",
    [
        pytest.param(
            "pcm.vnav.rpy.*",
            ["pcm.vnav.rpy.roll", "pcm.vnav.rpy.yaw"],
            id="sorted_suffix",
        ),
        pytest.param("*.angle", ["ludwig.steeringWheel.angle"], id="prefix"),
        pytest.param("pcm.*.x", ["pcm.vnav.bodyVelocity.x"], id="middle"),
        pytest.param(
            "pcm.vnav.rpy*",
            ["pcm.vnav.rpy.roll", "pcm.vnav.rpy.yaw"],
            id="dot_is_literal",
        ),
        pytest.param("pcm.vnav", [], id="no_wildcard_needs_exact_match"),
        pytest.param("*", sorted(NAMES), id="match_all"),
    ],
)
def test_match_pattern(pattern, expected):
    assert match_pattern(pattern, NAMES) == expected
