"""Render the Meccano steam shovel.

    python render.py --still                 # studio stills from the two photograph viewpoints
    python render.py --video                 # a full dig cycle, out/shovel_dig.mp4
    python render.py --video --camera work   # from another camera

Stills are rendered at twice the output size and averaged down, which, with
MSAA, keeps the hole edges and string clean.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np

from shovel.build import load
from shovel.cycle import settle
try:
    from shovel.cycle import DigCycle
except ImportError:
    DigCycle = None

OUT = Path(__file__).resolve().parent / "out"


def make_renderer(m, w, h):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    r = mujoco.Renderer(m, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0] * 6
    opt.tendongroup[:] = [1, 1, 1, 0, 0, 0]
    return r, opt


def shot(r, opt, d, camera, ss=2):
    r.update_scene(d, camera=camera, scene_option=opt)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    img = r.render().astype(np.float32)
    if ss > 1:
        h, w = img.shape[0] // ss, img.shape[1] // ss
        img = img[:h * ss, :w * ss].reshape(h, ss, w, ss, 3).mean((1, 3))
    return img.clip(0, 255).astype(np.uint8)


def stills(size=(1600, 1100)):
    m = load()
    d = mujoco.MjData(m)
    settle(m, d)
    w, h = size
    r, opt = make_renderer(m, 2 * w, 2 * h)
    OUT.mkdir(exist_ok=True)
    for cam in ("photo_side", "photo_front", "work"):
        img = shot(r, opt, d, cam)
        iio.imwrite(OUT / f"shovel_{cam}.png", img)
        print("wrote", OUT / f"shovel_{cam}.png")


def one_key_light(m):
    """For video: put the whole key on one shadow-casting light (four soft-shadow passes
    per frame are slow on a software GL)."""
    names = [m.light(i).name for i in range(m.nlight)]
    for i, n in enumerate(names):
        if n == "key0":
            m.light_diffuse[i] *= 4
            m.light_specular[i] *= 4
        elif n.startswith("key"):
            m.light_castshadow[i] = 0
            m.light_diffuse[i] = 0
            m.light_specular[i] = 0
    m.vis.quality.shadowsize = 2048


def video(camera="work", seconds=None, fps=30, size=(1280, 720), gravel=260, ss=1):
    m = load(gravel=gravel)
    one_key_light(m)
    d = mujoco.MjData(m)
    cyc = DigCycle(m, d)
    w, h = size
    r, opt = make_renderer(m, ss * w, ss * h)
    total = seconds or cyc.duration
    steps_per_frame = int(round(1 / fps / m.opt.timestep))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"shovel_dig_{camera}.mp4"
    import imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=8,
                            ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]) as vid:
        n = int(total * fps)
        for k in range(n):
            for _ in range(steps_per_frame):
                cyc.step()
            vid.append_data(shot(r, opt, d, camera, ss=ss))
            if k % (fps * 5) == 0:
                print(f"  {d.time:5.1f} s  {cyc.phase}", flush=True)
    print("wrote", path, f"({cyc.report()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--camera", default="work")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--gravel", type=int, default=260)
    a = ap.parse_args()
    if a.still or not a.video:
        stills()
    if a.video:
        video(a.camera, a.seconds, gravel=a.gravel)
