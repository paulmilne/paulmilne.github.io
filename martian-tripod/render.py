"""Render a short clip of the tripod: it stands, turns on its legs, and sweeps the Heat-Ray.

    uv run render.py                     # out/tripod_demo.mp4 from the 'witness' camera
    uv run render.py --camera track      # follow camera
    uv run render.py --still             # a single PNG instead of a video

This is a scripted servo animation, not yet a gait: the hip-yaw joints swing
in phase so the hood pivots on planted feet, the way a crane slews.
"""
import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from tripod.build import write
from tripod.params import DEFAULT, leg_ik

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="witness")
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--still", action="store_true")
args = ap.parse_args()

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
renderer = mujoco.Renderer(model, args.height, args.width)
cam = model.camera(args.camera).id
q_hip, q_knee = leg_ik(DEFAULT)

ray_yaw = model.joint("ray_yaw").qposadr[0]
ray_pitch = model.joint("ray_pitch").qposadr[0]


def control(t: float):
    # slow slew of the hood on planted feet, with a crouch-and-rise
    yaw = 0.35 * math.sin(2 * math.pi * t / 8.0)
    crouch = 0.08 * (1 - math.cos(2 * math.pi * t / 6.0))
    for i in range(3):
        data.ctrl[3 * i + 0] = yaw
        data.ctrl[3 * i + 1] = q_hip - crouch
        data.ctrl[3 * i + 2] = q_knee + 2 * crouch
    # the Heat-Ray arm is unactuated in the model; drive it kinematically for now
    data.qpos[ray_yaw] = 0.8 * math.sin(2 * math.pi * t / 5.0)
    data.qpos[ray_pitch] = 0.35 + 0.15 * math.sin(2 * math.pi * t / 3.0)
    data.qvel[model.joint("ray_yaw").dofadr[0]] = 0
    data.qvel[model.joint("ray_pitch").dofadr[0]] = 0


if args.still:
    for _ in range(int(1.0 / model.opt.timestep)):
        control(data.time); mujoco.mj_step(model, data)
    renderer.update_scene(data, camera=cam)
    path = out / f"tripod_{args.camera}.png"
    imageio.imwrite(path, renderer.render())
    print(f"wrote {path}")
else:
    path = out / f"tripod_demo_{args.camera}.mp4"
    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    with imageio.get_writer(path, fps=args.fps, quality=8) as w:
        for _ in range(int(args.seconds * args.fps)):
            for _ in range(steps_per_frame):
                control(data.time); mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cam)
            w.append_data(renderer.render())
    print(f"wrote {path}")
