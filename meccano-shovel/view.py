"""Interactive viewer: the shovel working its dig cycle, over and over.

    mjpython view.py            # macOS (the passive viewer needs mjpython)
    python view.py              # Linux / Windows
    mjpython view.py --still    # just the model in the photographed pose, no gravel

SPACE pauses. Double-click a part and Ctrl-drag to shove it.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from shovel.build import load
from shovel.cycle import DigCycle, settle

ap = argparse.ArgumentParser()
ap.add_argument("--still", action="store_true")
ap.add_argument("--gravel", type=int, default=260)
a = ap.parse_args()

m = load(gravel=0 if a.still else a.gravel)
d = mujoco.MjData(m)
paused = False


def key(k):
    global paused
    if k == 32:
        paused = not paused


with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
    v.opt.geomgroup[3] = 0
    if a.still:
        settle(m, d)
    cyc = None if a.still else DigCycle(m, d)
    while v.is_running():
        t0 = time.time()
        if not paused:
            for _ in range(int(1 / 60 / m.opt.timestep)):
                if cyc:
                    cyc.step()
                    if d.time >= cyc.duration:      # go round again from the return pose
                        d.time = 3.0
                else:
                    mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0, 1 / 60 - (time.time() - t0)))
