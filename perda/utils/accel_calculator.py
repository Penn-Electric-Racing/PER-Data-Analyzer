import numpy as np
from pydantic import BaseModel, Field

from ..analyzer.data_instance import DataInstance
from .units import *


class AccelSegmentResult(BaseModel):
    """Result of a single segment marked as an acceleration test"""

    start_time: float = Field(description="Timestamp when the segment began.")
    time_to_dist: float = Field(
        description="Time to travel `dist_reached` meters from segment start."
    )
    dist_reached: float = Field(description="Distance of the segment in meters.")
    timescale: Timescale = Field(description="Time unit for the output times.")

    def __str__(self) -> str:
        return (
            f"Accel at {self.start_time:.2f}s: "
            f"reached {self.dist_reached}m in {self.time_to_dist:.3f}s"
        )


def detect_accel_event(
    torque_obj: DataInstance,
    speed_obj: DataInstance,
    torque_threshold: float = 100,
    speed_threshold: float = 0.5,
) -> DataInstance:
    """Detect acceleration events based on torque and speed thresholds.

    An event is active when torque exceeds `torque_threshold` and speed exceeds
    `speed_threshold`, and ends when speed drops back to or below `speed_threshold`.

    Parameters
    ----------
    torque_obj : DataInstance
        Time-series of motor torque values.
    speed_obj : DataInstance
        Time-series of wheel speed values. The output is aligned to these timestamps.
    torque_threshold : float, optional
        Minimum torque (in Nm) required to trigger an acceleration event. Default is 100.
    speed_threshold : float, optional
        Speed value used as the trigger floor and reset condition. Default is 0.5.

    Returns
    -------
    DataInstance
        Binary signal (0.0 or 1.0) on `speed_obj` timestamps, labeled "Accel Event",
        where 1.0 indicates an active acceleration event.
    """
    torque_interp = np.interp(
        speed_obj.timestamp_np, torque_obj.timestamp_np, torque_obj.value_np
    )

    speed = speed_obj.value_np
    signal_values = np.zeros_like(speed)
    active = False

    for i in range(len(speed)):
        if not active:
            if torque_interp[i] > torque_threshold and speed[i] > speed_threshold:
                active = True
        else:
            if speed[i] <= speed_threshold:
                active = False
        if active:
            signal_values[i] = 1.0

    return DataInstance(
        timestamp_np=speed_obj.timestamp_np, value_np=signal_values, label="Accel Event"
    )


def compute_accel_results(
    signal_obj: DataInstance,
    distance_obj: DataInstance,
    target_dist: float = 75,
    source_time_unit: Timescale = Timescale.MS,
    target_time_unit: Timescale = Timescale.S,
) -> list[AccelSegmentResult]:
    """Compute time-to-distance results for each acceleration event.

    Parameters
    ----------
    signal_obj : DataInstance
        Binary accel event signal (0.0/1.0), typically from `detect_accel_event`.
    distance_obj : DataInstance
        Cumulative distance signal in meters.
    target_dist : float, optional
        Target distance in meters. Default is 75.
    source_time_unit : Timescale, optional
        Time unit of input timestamps. Default is Timescale.MS.
    target_time_unit : Timescale, optional
        Time unit for output times. Default is Timescale.S.

    Returns
    -------
    list[AccelSegmentResult]
        One result per qualifying segment.
    """
    sig = signal_obj.value_np
    time = signal_obj.timestamp_np

    diff_arr = np.diff(sig, prepend=0)
    start_indices = np.where(diff_arr == 1)[0]
    end_indices = np.where(diff_arr == -1)[0]

    results = []
    for start_idx in start_indices:
        t_start = time[start_idx]

        future_ends = end_indices[end_indices > start_idx]
        t_end_signal = time[future_ends[0]] if len(future_ends) > 0 else time[-1]

        dist_at_start = np.interp(
            t_start, distance_obj.timestamp_np, distance_obj.value_np
        )
        dist_at_signal_end = np.interp(
            t_end_signal, distance_obj.timestamp_np, distance_obj.value_np
        )

        if dist_at_signal_end - dist_at_start < target_dist:
            continue

        target_absolute_dist = dist_at_start + target_dist
        mask = distance_obj.timestamp_np >= t_start
        future_t = distance_obj.timestamp_np[mask]
        future_d = distance_obj.value_np[mask]

        if future_d[-1] >= target_absolute_dist:
            t_target_hit = np.interp(target_absolute_dist, future_d, future_t)
            results.append(
                AccelSegmentResult(
                    start_time=convert_time(
                        t_start, source_time_unit, target_time_unit
                    ),
                    time_to_dist=convert_time(
                        t_target_hit - t_start, source_time_unit, target_time_unit
                    ),
                    dist_reached=target_dist,
                    timescale=target_time_unit,
                )
            )

    return results
