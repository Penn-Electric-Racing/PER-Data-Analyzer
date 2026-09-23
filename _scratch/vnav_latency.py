"""Measure VN-300 INS velocity latency from a car log.

Method
------
Cross-correlate two channels that see the same physical event through different
amounts of filtering, and read off the lag. Two independent pairs are used:

  TEST 1  d/dt(vnav.velocityBody.x)  vs  vnav.linearAccelBody.x
          Both are VN outputs, so this isolates the INS filter's velocity path
          but is blind to delay common to the whole device.

  TEST 2  vnav.velocityBody.x  vs  a healthy wheel-speed channel
          Fully independent sensor path, so it catches common-mode delay too.

Correlating the raw speed traces does NOT work: they are dominated by the
low-frequency ramp, the correlation peak is flat, and the answer comes back
near zero no matter what the true lag is. Differentiating first fixes this.
Every reported number is calibrated by re-running with known delays injected;
the recovered-vs-injected slope must be ~1 or the result is meaningless.
"""

import sys

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from perda.analyzer.analyzer import Analyzer
from perda.utils.resampling import resample_to_freq

VEL_X = "pcm.vnav.velocityBody.x"
ACC_X = "pcm.vnav.linearAccelBody.x"
GPS_FIX = "pcm.vnav.gpsFix"
WHEELS = ("pcm.wheelSpeeds.frontRight", "pcm.wheelSpeeds.frontLeft",
          "pcm.wheelSpeeds.backRight", "pcm.wheelSpeeds.backLeft")

FS_HZ = 200.0
TS_PER_S = 1e6
MPH_TO_MPS = 0.44704
INJECTIONS_S = (0.0, 0.010, 0.025, 0.050, 0.100)


def _grid(aly: Analyzer, name: str, t0: float, t1: float) -> NDArray[np.float64]:
    di = resample_to_freq(aly.data[name], FS_HZ, TS_PER_S)
    t = di.timestamp_np / TS_PER_S
    return np.interp(np.arange(t0, t1, 1 / FS_HZ), t, di.value_np)


def _lowpass(x: NDArray[np.float64], cutoff_hz: float) -> NDArray[np.float64]:
    b, a = butter(4, cutoff_hz / (FS_HZ / 2), "low")
    return filtfilt(b, a, x)


def lag_seconds(x: NDArray[np.float64], y: NDArray[np.float64],
                max_lag_s: float = 0.4) -> tuple[float, float]:
    """Lag of x behind y, with sub-sample parabolic refinement, and corr peak."""
    n = len(x)
    xn = (x - x.mean()) / (x.std() or 1.0)
    yn = (y - y.mean()) / (y.std() or 1.0)
    ml = int(max_lag_s * FS_HZ)
    xc = np.correlate(xn, yn, "full")[n - 1 - ml: n + ml]
    k = float(np.argmax(xc))
    i = int(k)
    if 0 < i < len(xc) - 1:
        y0, y1, y2 = xc[i - 1], xc[i], xc[i + 1]
        den = y0 - 2 * y1 + y2
        if den:
            k = i + 0.5 * (y0 - y2) / den
    return (k - ml) / FS_HZ, float(xc.max() / n)


def _sweep(build_a, ref_b, starts, window, min_corr=0.7) -> list[tuple[float, float, int]]:
    """For each injected delay, the median recovered lag across windows."""
    out = []
    for inj in INJECTIONS_S:
        sig = build_a(inj)
        got = [lag_seconds(sig[s:s + window], ref_b[s:s + window]) for s in starts]
        got = [(l, p) for l, p in got if p > min_corr]
        out.append((inj, float(np.median([l for l, _ in got])), len(got)))
    return out


def _report(title: str, sweep: list[tuple[float, float, int]]) -> float:
    print(f"\n{title}")
    for inj, rec, n in sweep:
        print(f"    injected {inj * 1000:5.0f} ms  ->  recovered {rec * 1000:+7.1f} ms   (n={n})")
    inj = np.array([s[0] for s in sweep])
    rec = np.array([s[1] for s in sweep])
    slope, intercept = np.polyfit(inj, rec, 1)
    verdict = "OK" if 0.85 < slope < 1.15 else "INSENSITIVE - result not trustworthy"
    print(f"    slope={slope:.2f} ({verdict})   measured lag = {intercept * 1000:+.1f} ms")
    return float(intercept)


def main(path: str) -> None:
    aly = Analyzer(path, verbose=0)
    for ch in (VEL_X, ACC_X):
        if ch not in aly.data:
            sys.exit(f"log has no {ch}")
    if GPS_FIX in aly.data:
        fix = aly.data[GPS_FIX].value_np
        print(f"GPS fix: 3D for {100 * (fix >= 3).mean():.1f}% of samples")
        if (fix >= 3).mean() < 0.5:
            print("  WARNING: poor fix, INS velocity is largely dead reckoning")

    vt = aly.data[VEL_X].timestamp_np / TS_PER_S
    at = aly.data[ACC_X].timestamp_np / TS_PER_S
    t0, t1 = max(vt[0], at[0]), min(vt[-1], at[-1])
    grid = np.arange(t0, t1, 1 / FS_HZ)
    speed = _grid(aly, VEL_X, t0, t1)
    imu = _lowpass(_grid(aly, ACC_X, t0, t1), 12.0)

    def ins_accel(inj: float) -> NDArray[np.float64]:
        shifted = np.interp(grid, vt + inj, aly.data[VEL_X].value_np)
        return _lowpass(np.gradient(shifted, 1 / FS_HZ), 12.0)

    w1 = int(4 * FS_HZ)
    s1 = [s for s in range(0, len(grid) - w1, w1 // 2) if np.abs(imu[s:s + w1]).max() > 3.0]
    print(f"\nwindows with |a_x| > 3 m/s^2: {len(s1)}")
    lag1 = _report("TEST 1  d/dt(INS velocity)  vs  linearAccelBody",
                   _sweep(ins_accel, imu, s1, w1))

    # Pick the wheel-speed channel that actually tracks the INS speed.
    best, best_err = None, np.inf
    print("\nwheel-speed sensor health (ratio to INS speed while moving):")
    for name in WHEELS:
        if name not in aly.data:
            continue
        w = _grid(aly, name, t0, t1) * MPH_TO_MPS
        ok = (speed > 8) & (w > 0.5)
        if ok.sum() < 1000:
            print(f"    {name:34s} DEAD / no usable samples")
            continue
        ratio = float(np.median(w[ok] / speed[ok]))
        err = abs(ratio - 1.0)
        print(f"    {name:34s} ratio={ratio:.3f}" + ("  <-- healthy" if err < 0.1 else "  <-- BAD"))
        if err < best_err:
            best, best_err = name, err

    if best is None or best_err > 0.1:
        print("\nno healthy wheel-speed channel; skipping TEST 2")
        return
    wheel = _lowpass(_grid(aly, best, t0, t1) * MPH_TO_MPS, 8.0)
    dwheel = np.gradient(wheel, 1 / FS_HZ)

    def ins_speed_d(inj: float) -> NDArray[np.float64]:
        shifted = np.interp(grid, vt + inj, aly.data[VEL_X].value_np)
        return np.gradient(_lowpass(shifted, 8.0), 1 / FS_HZ)

    w2 = int(6 * FS_HZ)
    s2 = [s for s in range(0, len(grid) - w2, w2 // 2) if speed[s:s + w2].mean() > 8]
    lag2 = _report(f"TEST 2  INS velocity  vs  {best}", _sweep(ins_speed_d, dwheel, s2, w2))

    print(f"\nSUMMARY: INS velocity lag = {lag1 * 1000:+.0f} ms vs its own IMU accel, "
          f"{lag2 * 1000:+.0f} ms vs an independent wheel-speed sensor.")


def _self_check() -> None:
    fs, delay = FS_HZ, 0.075
    t = np.arange(0, 20, 1 / fs)
    rng = np.random.default_rng(0)
    truth = np.convolve(rng.standard_normal(len(t)), np.ones(10) / 10, mode="same")
    shifted = np.interp(t, t + delay, truth)
    lag, peak = lag_seconds(shifted, truth)
    assert abs(lag - delay) < 1.5 / fs, f"got {lag}, want {delay}"
    assert peak > 0.8, peak
    print(f"self-check ok: recovered {lag * 1000:.0f} ms of an injected {delay * 1000:.0f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _self_check()
    else:
        main(sys.argv[1])
