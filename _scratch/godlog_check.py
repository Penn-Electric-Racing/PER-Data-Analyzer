"""Load GodLog with perda; report TC mode + wheel-speed sensor health."""
import numpy as np
from perda.analyzer.analyzer import Analyzer

LOG = "_scratch/logcache/GodLog_05-01.csv"
aly = Analyzer(LOG, verbose=0)

for name in ("pcm.tractionControl.mode", "pcm.tractionControl.targetSlipRatio",
             "pcm.tractionControl.tcsTorque", "pcm.tractionControl.slipRatio"):
    if name in aly.data:
        v = aly.data[name].value_np
        vals, cnt = np.unique(v, return_counts=True)
        head = ", ".join(f"{a:g}:{b}" for a, b in list(zip(vals, cnt))[:6])
        print(f"{name:40s} n={len(v):7d}  {head}")
    else:
        print(f"{name:40s} MISSING")
