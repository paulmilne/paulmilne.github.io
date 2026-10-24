"""Render the tripod to video (or a single frame).

    uv run render.py                          # the attack: walk into the village, Heat-Ray firing
    uv run render.py --camera track           # follow the machine instead of watching from the village
    uv run render.py --scene stand            # stand, turn the hood and sweep the ray
    uv run render.py --gait transfer --seconds 120 --speed 2   # the slow edge-to-edge gait
    uv run render.py --still --at 25          # one PNG, 25 s in

Cameras: witness (eye height in the village, always looking at the hood),
track (follows the machine), martian (from under the hood).
"""
import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from tripod.build import write
from tripod.params import DEFAULT
from tripod.control import BodyController
from tripod.kinematics import rotz
from tripod.martian import Martian

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=["attack", "stand"], default="attack")
ap.add_argument("--gait", choices=["predictive", "transfer"], default="predictive",
                help="predictive: model-predictive COM, about 6 mph; transfer: stop at each edge, about 1 mph")
ap.add_argument("--camera", default="witness")
ap.add_argument("--seconds", type=float, default=60.0, help="simulated time to render")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--still", action="store_true")
ap.add_argument("--at", type=float, default=None, help="time of the still (default: --seconds)")
args = ap.parse_args()

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, args.height, args.width)
cam = model.camera(args.camera).id

martian = Martian(model, data, gait=args.gait, walk=(args.scene == "attack"))
hold = BodyController(DEFAULT, model) if args.scene == "stand" else None
com0 = hold.com_state(data)[0] if hold else None


def step():
    if hold is not None:                         # stand: slew the hood on planted feet, crouch and rise
        t = data.time
        yaw = 0.35 * math.sin(2 * math.pi * t / 12.0)
        goal = com0 + np.array([0.0, 0.0, -1.5 * (1 - math.cos(2 * math.pi * t / 9.0)) / 2])
        hold(data, [0, 1, 2], goal, np.zeros(3), rotz(yaw))
        martian.walk = False
    martian(data)
    mujoco.mj_step(model, data)


def frame():
    renderer.update_scene(data, camera=cam)
    martian.decorate(renderer.scene, data)
    return renderer.render()


T = args.at if (args.still and args.at is not None) else args.seconds
if args.still:
    while data.time < T:
        step()
    path = out / f"tripod_{args.scene}_{args.camera}.png"
    imageio.imwrite(path, frame())
else:
    path = out / f"tripod_{args.scene}_{args.camera}.mp4"
    per_frame = max(1, round(args.speed / (args.fps * model.opt.timestep)))
    with imageio.get_writer(path, fps=args.fps, quality=8) as w:
        n = 0
        while data.time < T:
            for _ in range(per_frame):
                step()
            w.append_data(frame())
            n += 1
            if n % (args.fps * 5) == 0:
                print(f"  {data.time:5.1f} s simulated")
print(f"wrote {path}")
