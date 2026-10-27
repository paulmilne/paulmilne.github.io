"""Film the solar-system orrery.

    python solar_video.py                                    # 12 years from above the table, 30 s
    python solar_video.py --camera top --years 165 --seconds 24 --name solar_neptune.mp4

One crank turn is one year; the corner shows the date, counted on from the start
date by the Earth's turns, and how many times Mercury and Neptune have gone round.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

from bricks.solar import load, turns

OUT = Path(__file__).resolve().parent / "out"
START = dt.date(2026, 10, 27)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--years", type=float, default=11.86)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--name")
    a = ap.parse_args()
    m, facts = load()
    d = mujoco.MjData(m)
    d.ctrl[0] = 2 * math.pi
    w, h = map(int, a.size.split("x"))
    r = mujoco.Renderer(m, h, w)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    steps_year = int(round(1.0 / m.opt.timestep))
    # let the crank's servo come up to speed, then wind the clock back to the start date
    for _ in range(200):
        mujoco.mj_step(m, d)
    q_start = {p: v for p, v in turns(m, d, facts).items()}
    n = int(a.seconds * a.fps)
    per_frame = a.years / n
    big, small = font(int(h * 0.040)), font(int(h * 0.028))
    OUT.mkdir(exist_ok=True)
    path = OUT / (a.name or f"solar_{a.camera}.mp4")
    acc = 0.0
    with imageio.get_writer(path, fps=a.fps, codec="libx264", quality=None, macro_block_size=8,
                            ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]) as vid:
        for k in range(n):
            acc += per_frame * steps_year
            steps = int(acc)
            acc -= steps
            for _ in range(steps):
                mujoco.mj_step(m, d)
            r.update_scene(d, camera=a.camera, scene_option=opt)
            img = Image.fromarray(r.render())
            T = turns(m, d, facts)
            years = T["Earth"] - q_start["Earth"]
            date = START + dt.timedelta(days=years * 365.25636)
            dr = ImageDraw.Draw(img)
            dr.text((int(w * 0.03), int(h * 0.04)), date.strftime("%d %B %Y"), font=big, fill=(240, 236, 220))
            dr.text((int(w * 0.03), int(h * 0.04) + int(h * 0.055)),
                    f"Mercury {T['Mercury'] - q_start['Mercury']:6.1f} orbits   ·   "
                    f"Neptune {T['Neptune'] - q_start['Neptune']:5.3f}", font=small, fill=(200, 196, 185))
            vid.append_data(np.asarray(img))
            if k % (a.fps * 4) == 0:
                print(f"  frame {k}/{n}  {date}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
