"""Render the tripod to video, with sound (or a single frame).

    uv run render.py                          # the attack, from the village, with sound
    uv run render.py --camera track           # follow the machine instead
    uv run render.py --scene stand            # stand, turn the hood and sweep the ray
    uv run render.py --gait transfer --seconds 120 --speed 2   # the slow edge-to-edge gait
    uv run render.py --no-sound               # video only
    uv run render.py --audio-only             # just the soundtrack (no OpenGL needed)
    uv run render.py --still --at 25          # one PNG, 25 s in

Cameras: witness (eye height in the village, always looking at the hood),
track (follows the machine), martian (from under the hood). The sound is
heard from the camera: see tripod/audio.py.
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
from tripod.audio import SoundRecorder, synthesise, write_wav, mux

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
ap.add_argument("--no-sound", action="store_true")
ap.add_argument("--audio-only", action="store_true", help="write the soundtrack as a WAV, render no frames")
args = ap.parse_args()

out = Path(__file__).resolve().parent / "out"
out.mkdir(exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)
cam = model.camera(args.camera).id
renderer = None if args.audio_only else mujoco.Renderer(model, args.height, args.width)

martian = Martian(model, data, gait=args.gait, walk=(args.scene == "attack"))
hold = BodyController(DEFAULT, model) if args.scene == "stand" else None
com0 = hold.com_state(data)[0] if hold else None
recorder = None if (args.still or args.no_sound) else SoundRecorder(model, martian, cam)


def step():
    if hold is not None:                         # stand: slew the hood on planted feet, crouch and rise
        t = data.time
        yaw = 0.35 * math.sin(2 * math.pi * t / 12.0)
        goal = com0 + np.array([0.0, 0.0, -1.5 * (1 - math.cos(2 * math.pi * t / 9.0)) / 2])
        hold(data, [0, 1, 2], goal, np.zeros(3), rotz(yaw))
        martian.walk = False
    martian(data)
    mujoco.mj_step(model, data)
    if recorder is not None:
        recorder(data)


def frame():
    renderer.update_scene(data, camera=cam)
    martian.decorate(renderer.scene, data)
    return renderer.render()


T = args.at if (args.still and args.at is not None) else args.seconds
stem = f"tripod_{args.scene}_{args.camera}"
if args.still:
    while data.time < T:
        step()
    path = out / f"{stem}.png"
    imageio.imwrite(path, frame())
    print(f"wrote {path}")
else:
    t0 = data.time
    per_frame = max(1, round(args.speed / (args.fps * model.opt.timestep)))
    n_frames = 0
    video = out / f"{stem}_silent.mp4" if recorder else out / f"{stem}.mp4"
    writer = None if args.audio_only else imageio.get_writer(video, fps=args.fps, quality=8)
    while data.time < T:
        for _ in range(per_frame):
            step()
        if writer:
            writer.append_data(frame())
        n_frames += 1
        if n_frames % (args.fps * 5) == 0:
            print(f"  {data.time:5.1f} s simulated")
    if writer:
        writer.close()
    if recorder is None:
        print(f"wrote {video}")
    else:
        # frame n shows simulation time t0 + (n + 1) * per_frame * dt; the audio uses the same clock
        dt_frame = per_frame * model.opt.timestep
        audio, marks = synthesise(recorder, t0 + dt_frame, n_frames / args.fps, speed=dt_frame * args.fps)
        wav = out / f"{stem}.wav"
        write_wav(wav, audio)
        print(f"wrote {wav}  ({len(marks)} footfalls)")
        if writer:
            final = mux(video, wav, out / f"{stem}.mp4")
            video.unlink()
            print(f"wrote {final}")
