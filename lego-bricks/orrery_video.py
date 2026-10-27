"""Film the Technic orrery.

    python orrery_video.py                                   # a year from above the table, 24 s
    python orrery_video.py --camera earthcam --days 60 --seconds 20 --sunlight-only --name orrery_moon.mp4

The crank is turned at a steady rate by its servo; each frame advances the
simulation by the right number of sidereal days, and the corner shows the day
(solar days, counted from the Earth's turns relative to the arm) and the Moon's
phase, both read from the mechanism itself.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bricks.orrery import angles, load, sunlight_only

OUT = Path(__file__).resolve().parent / "out"
PHASES = ["new Moon", "waxing crescent", "first quarter", "waxing gibbous", "full Moon", "waning gibbous",
          "last quarter", "waning crescent"]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def phase_name(frac):
    return PHASES[int((frac % 1) * 8 + 0.5) % 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--days", type=float, default=366.2564, help="sidereal days (crank turns) to film")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--sunlight-only", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="sidereal days to run before filming")
    ap.add_argument("--name")
    a = ap.parse_args()

    m, facts = load()
    if a.sunlight_only:
        sunlight_only(m)
    d = mujoco.MjData(m)
    w, h = map(int, a.size.split("x"))
    r = mujoco.Renderer(m, h, w)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    d.ctrl[0] = 2 * math.pi                       # one crank turn (sidereal day) per simulated second
    steps_day = int(round(1.0 / m.opt.timestep))
    for _ in range(int(a.start * steps_day)):
        mujoco.mj_step(m, d)
    n = int(a.seconds * a.fps)
    per_frame = a.days / n
    big, small = font(int(h * 0.040)), font(int(h * 0.028))
    OUT.mkdir(exist_ok=True)
    path = OUT / (a.name or f"orrery_{a.camera}.mp4")
    acc = 0.0
    with imageio.get_writer(path, fps=a.fps, codec="libx264", quality=None, macro_block_size=8,
                            ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]) as vid:
        for k in range(n):
            acc += per_frame * steps_day
            steps = int(acc)
            acc -= steps
            for _ in range(steps):
                mujoco.mj_step(m, d)
            r.update_scene(d, camera=a.camera, scene_option=opt)
            img = Image.fromarray(r.render())
            T = {key: v / (2 * math.pi) for key, v in angles(m, d, facts).items()}
            day = T["earth"] - T["arm"]
            lunation = T["moon"] - T["arm"]
            dr = ImageDraw.Draw(img)
            dr.text((int(w * 0.03), int(h * 0.04)), f"day {day:6.1f}", font=big, fill=(240, 236, 220))
            dr.text((int(w * 0.03), int(h * 0.04) + int(h * 0.055)),
                    f"year {T['arm']:.3f}   ·   lunation {lunation:.2f}   ·   {phase_name(lunation + 0.5)}",
                    font=small, fill=(200, 196, 185))
            vid.append_data(np.asarray(img))
            if k % (a.fps * 4) == 0:
                print(f"  frame {k}/{n}  day {day:.1f}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
