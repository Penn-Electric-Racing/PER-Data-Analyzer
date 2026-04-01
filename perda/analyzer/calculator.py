from .analyzer import *
from scipy.integrate import cumulative_trapezoid
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

def get_cumulative_integration(data, timescale=1000000, filter_window_size=10, n_sigmas=3, smoothing_window_len=11, smoothing_poly_order=2):

    v = pd.Series(data.value_np)
    t = np.array(data.timestamp_np)

    # Calculate rolling median and the MAD (Median Absolute Deviation)
    rolling_median = v.rolling(window=filter_window_size, center=True).median()
    rolling_std = v.rolling(window=filter_window_size, center=True).apply(lambda x: np.abs(x - x.median()).median()) * 1.4826

    # Identify spikes (points more than X "standard deviations" from the local median)
    is_outlier = np.abs(v - rolling_median) > (n_sigmas * rolling_std)

    # interpolate the outliers
    v_cleaned = v.copy()
    v_cleaned[is_outlier] = np.nan
    v_cleaned = v_cleaned.interpolate(method='linear').bfill().ffill()

    # Smoothing
    if len(v_cleaned) > smoothing_window_len:
        v_smooth = savgol_filter(v_cleaned, smoothing_window_len, smoothing_poly_order)
    else:
        v_smooth = v_cleaned

    v_integrated = cumulative_trapezoid(v_smooth, x=t / timescale, initial=0)

    return t, v_smooth, v_integrated

def detect_accel_event(torque_obj, speed_obj, torque_threshold=100, speed_threshold=0.5):
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

def get_accel_triggers(aly : Analyzer, target_dist=75, timescale=1000000, torque_threshold=100, speed_threshold=0.5, window_size=10, n_sigmas=3, window_len=11, poly_order=2):

    speed_obj = (aly.data["pcm.wheelSpeeds.frontRight"] + aly.data["pcm.wheelSpeeds.frontLeft"]) / 2.0
    signal_obj = detect_accel_event(torque_obj=aly.data["pcm.moc.motor.requestedTorque"], speed_obj=aly.data["pcm.wheelSpeeds.frontRight"], torque_threshold=torque_threshold, speed_threshold=speed_threshold)
    
    time_arr, _, distance = get_cumulative_integration("pcm.wheelSpeeds.frontRight")
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