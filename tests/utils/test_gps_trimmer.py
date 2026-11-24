import numpy as np
import pytest

pytest.importorskip("ipywidgets")

from perda.core_data_structures.data_instance import DataInstance
from perda.units import Timescale
from perda.utils.gps_trimmer import plot_gps_trimmer


def test_aligns_longitude_onto_latitude_timestamps():
    lat = DataInstance(
        timestamp_np=np.array([0, 10, 20], dtype=np.int64),
        value_np=np.array([40.0, 40.1, 40.2]),
        label="lat",
    )
    lon = DataInstance(
        timestamp_np=np.array([0, 20], dtype=np.int64),
        value_np=np.array([-75.0, -75.2]),
        label="lon",
    )
    widget = plot_gps_trimmer(lat, lon, Timescale.MS)
    trace = widget.children[0].data[0]
    # lon is linearly interpolated onto lat's timestamps: midway at t=10 gives -75.1
    np.testing.assert_allclose(trace.x, [-75.0, -75.1, -75.2])
    np.testing.assert_allclose(trace.y, [40.0, 40.1, 40.2])
