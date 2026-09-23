"""How much grip is the car actually using? Lateral/longitudinal accel envelope."""
import numpy as np
from perda.analyzer.analyzer import Analyzer
from perda.utils.resampling import resample_to_freq

FS, TS, G = 100.0, 1e6, 9.80665
aly = Analyzer("_scratch/logcache/GodLog_05-01.csv", verbose=0)

def grid(name, t0, t1):
    di = resample_to_freq(aly.data[name], FS, TS)
    return np.interp(np.arange(t0, t1, 1/FS), di.timestamp_np/TS, di.value_np)

ax_ch, ay_ch = "pcm.vnav.linearAccelBody.x", "pcm.vnav.linearAccelBody.y"
v_ch, yaw_ch = "pcm.vnav.velocityBody.x", "pcm.vnav.compensatedAngularRate.z"
for c in (ax_ch, ay_ch, v_ch, yaw_ch):
    if c not in aly.data: print(f"MISSING {c}")
t = aly.data[ax_ch].timestamp_np/TS
t0, t1 = t[0], t[-1]
ax, ay, v = grid(ax_ch,t0,t1)/G, grid(ay_ch,t0,t1)/G, grid(v_ch,t0,t1)
yaw = np.abs(grid(yaw_ch,t0,t1))

moving = v > 5
print(f"samples moving >5 m/s: {moving.sum()} ({100*moving.mean():.0f}%)  max speed {v.max():.1f} m/s")
for name, arr in (("lateral |ay|", np.abs(ay)), ("longitudinal |ax|", np.abs(ax))):
    a = arr[moving]
    print(f"{name:18s} p50={np.percentile(a,50):.2f}g  p99={np.percentile(a,99):.2f}g  "
          f"p99.9={np.percentile(a,99.9):.2f}g  max={a.max():.2f}g")
comb = np.hypot(ax, ay)[moving]
print(f"{'combined':18s} p99={np.percentile(comb,99):.2f}g  p99.9={np.percentile(comb,99.9):.2f}g  max={comb.max():.2f}g")

# sustained cornering: how long does it hold >1g lateral?
hi = np.abs(ay) > 1.0
runs, n = [], 0
for b in hi:
    if b: n += 1
    elif n: runs.append(n); n = 0
runs = np.array(runs)/FS
print(f"\nsustained >1.0g lateral: {len(runs)} episodes, longest {runs.max():.2f}s, "
      f"{(runs>1.0).sum()} lasting >1s" if len(runs) else "\nnever exceeded 1.0g lateral")
