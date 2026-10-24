"""Watch the Martian live in MuJoCo's viewer.

    uv run mjpython view.py                 # the attack (macOS needs mjpython for a live viewer)
    uv run mjpython view.py --scene stand   # standing, burning what is in range
    uv run mjpython view.py --speed 3       # run faster than real time

Keys: SPACE holds or releases the Heat-Ray. The camera list (right panel)
includes 'witness' (eye height in the village) and 'track'. Double-click a
body to select it; Ctrl+right-drag pushes it.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from tripod.build import write
from tripod.martian import Martian

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=["attack", "stand"], default="attack")
ap.add_argument("--speed", type=float, default=1.0)
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)
martian = Martian(model, data, walk=(args.scene == "attack"))
if args.scene == "stand":
    from tripod.control import BodyController
    from tripod.kinematics import rotz
    import numpy as np
    hold = BodyController(martian.gait.p, model)
    goal = hold.com_state(data)[0]


def key(keycode: int):
    if keycode == 32:                                  # space
        martian.armed = not martian.armed


frame_dt = 1 / 60
with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
    while viewer.is_running():
        wall = time.time()
        sim_end = data.time + frame_dt * args.speed
        while data.time < sim_end:
            if args.scene == "stand":
                hold(data, [0, 1, 2], goal, np.zeros(3), rotz(0.0))
            martian(data)
            mujoco.mj_step(model, data)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            martian.decorate(viewer.user_scn, data)
        viewer.sync()
        time.sleep(max(0.0, frame_dt - (time.time() - wall)))
