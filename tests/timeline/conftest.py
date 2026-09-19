import pytest

from .fixtures import frame_with_sentinels


@pytest.fixture
def sentinel_frame():
    """Sample frame mixing real values with denormal sentinels."""
    return frame_with_sentinels()
