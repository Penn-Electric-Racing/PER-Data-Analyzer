import numpy as np
from ..analyzer.analyzer import Analyzer
from ..analyzer.data_instance import DataInstance
from ..utils.integrate import *

def detect_accel_event(torque_obj, speed_obj, torque_threshold=100, speed_threshold=0.5):
    """Detect acceleration events based on torque and speed thresholds.

    An event is active when torque exceeds `torque_threshold` and speed exceeds
    `speed_threshold`, and ends when speed drops back to or below `speed_threshold`.

    Parameters
    ----------
    torque_obj : DataInstance
        Time-series of motor torque values.
    speed_obj : DataInstance
        Time-series of wheel speed values. The output is aligned to these timestamps.
    torque_threshold : float
        Minimum torque (in Nm) required to trigger an acceleration event.
    speed_threshold : float
        Speed value (in the units of `speed_obj`) used as the trigger floor and reset condition.

    Returns
    -------
    DataInstance
        Binary signal (0.0 or 1.0) on `speed_obj` timestamps, labeled "Accel Event",
        where 1.0 indicates an active acceleration event.
    """
    # Align Torque to Speed timestamps
    torque_interp = np.interp(
        speed_obj.timestamp_np,
        torque_obj.timestamp_np,
        torque_obj.value_np
    )

    speed = speed_obj.value_np
    signal_values = np.zeros_like(speed)
    active = False

    for i in range(len(speed)):
        # Trigger: Torque is high AND speed starts to climb
        if not active:
            if torque_interp[i] > torque_threshold and speed[i] > speed_threshold:
                active = True

        # Reset: Speed returns to (near) zero
        else:
            if speed[i] <= speed_threshold:
                active = False

        if active:
            signal_values[i] = 1.0

    var = DataInstance(timestamp_np=speed_obj.timestamp_np, value_np = signal_values, label="Accel Event")

    return var

def get_accel_triggers(aly : Analyzer, target_dist=75, timescale=1000000, torque_threshold=100, speed_threshold=0.5, filter_window_size=10, n_sigmas=3, smoothing_window_len=11, smoothing_poly_order=2):
    """Find all acceleration events in a run and compute time-to-distance for each.

    Detects acceleration events using `detect_accel_event`, integrates wheel speed to
    compute cumulative distance, and records the elapsed time from each event start until
    `target_dist` meters are covered.

    Parameters
    ----------
    aly : Analyzer
        Analyzer instance containing the run data.
    target_dist : float
        Target distance in meters to measure time to.
    timescale : float
        Factor to convert raw timestamps to seconds (e.g. 1000000 for microseconds).
    torque_threshold : float
        Minimum torque (in Nm) to trigger an acceleration event.
    speed_threshold : float
        Speed value used as the trigger floor and reset condition for event detection.
    filter_window_size : int
        Window size for outlier filtering in cumulative integration.
    n_sigmas : float
        Number of standard deviations for outlier detection in cumulative integration.
    smoothing_window_len : int
        Window length for Savitzky-Golay smoothing in cumulative integration.
    smoothing_poly_order : int
        Polynomial order for Savitzky-Golay smoothing in cumulative integration.

    Returns
    -------
    list[dict]
        List of dicts, one per qualifying event, each with keys:
        ``start_time`` (raw timestamp), ``time_to_dist`` (seconds to reach `target_dist`),
        and ``dist_reached`` (target distance in meters).
    """

    speed_obj = (aly.data["pcm.wheelSpeeds.frontRight"] + aly.data["pcm.wheelSpeeds.frontLeft"]) / 2.0
    signal_obj = detect_accel_event(torque_obj=aly.data["pcm.moc.motor.requestedTorque"], speed_obj=aly.data["pcm.wheelSpeeds.frontRight"], torque_threshold=torque_threshold, speed_threshold=speed_threshold)
    
    time_arr, _, distance = get_cumulative_integration(speed_obj, timescale=timescale, filter_window_size=filter_window_size, n_sigmas=n_sigmas, smoothing_window_len=smoothing_window_len, smoothing_poly_order=smoothing_poly_order)
    distance_obj = DataInstance(timestamp_np=time_arr, value_np = distance / 3600 * 1609.34, label="Distance")

    sig = signal_obj.value_np
    time = signal_obj.timestamp_np

    # Find starts (1) and ends (-1)
    diff = np.diff(sig, prepend=0)
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]

    results = []

    for start_idx in start_indices:
        t_start = time[start_idx]
        
        # Get the first end_index that occurs after this start_index
        future_ends = end_indices[end_indices > start_idx]
        t_end_signal = time[future_ends[0]] if len(future_ends) > 0 else time[-1]

        dist_at_start = np.interp(t_start, distance_obj.timestamp_np, distance_obj.value_np)
        dist_at_signal_end = np.interp(t_end_signal, distance_obj.timestamp_np, distance_obj.value_np)
        dist_covered_during_signal = dist_at_signal_end - dist_at_start

        if dist_covered_during_signal < target_dist:
            continue 

        target_absolute_dist = dist_at_start + target_dist
        mask = distance_obj.timestamp_np >= t_start
        future_t = distance_obj.timestamp_np[mask]
        future_d = distance_obj.value_np[mask]

        if future_d[-1] >= target_absolute_dist:
            t_target_hit = np.interp(target_absolute_dist, future_d, future_t)
            elapsed = t_target_hit - t_start

            results.append({
                "start_time": t_start,
                "time_to_dist": elapsed / timescale,
                "dist_reached": target_dist
            })

    for e in results:
        print(f"Accel at {e['start_time'] / timescale:.2f}s: reached {target_dist}m in {e['time_to_dist']:.3f}s")
    
    return results