"""Rendering LEGO buildings with MuJoCo.

One model holds a set of 2x4 bricks as mocap bodies (a box and eight studs each);
a building is shown by moving bricks into place and parking the rest out of
sight, so one renderer serves every picture.
"""
from __future__ import annotations

import math
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .count import DIMS
from .geometry import BRICK, PITCH, PLAY, STUD_D, STUD_H

MM = 0.001
# classic LEGO colours (approximately: bright red, blue, yellow, green, white, black)
COLOURS = {"red": (0.79, 0.10, 0.04), "blue": (0.0, 0.33, 0.75), "yellow": (0.95, 0.80, 0.22),
           "green": (0.14, 0.47, 0.26), "white": (0.95, 0.95, 0.93), "black": (0.12, 0.13, 0.15)}
ORDER = ("red", "blue", "yellow", "green", "white", "black")


def model_xml(n_bricks: int = 6) -> str:
    L, W = 4 * PITCH / 2 - PLAY, 2 * PITCH / 2 - PLAY
    parts = []
    for k in range(n_bricks):
        studs = "".join(
            f'<geom type="cylinder" size="{STUD_D/2*MM} {STUD_H/2*MM}" '
            f'pos="{(i - 1.5)*PITCH*MM} {(j - 0.5)*PITCH*MM} {(BRICK/2 + STUD_H/2)*MM}" material="b{k}"/>'
            for i in range(4) for j in range(2))
        parts.append(f'<body name="brick{k}" mocap="true" pos="0 0 -1">'
                     f'<geom type="box" size="{L*MM} {W*MM} {BRICK/2*MM}" material="b{k}"/>{studs}</body>')
    mats = "".join(f'<material name="b{k}" rgba="1 0 0 1" specular="0.55" shininess="0.75" reflectance="0.02"/>'
                   for k in range(n_bricks))
    return f"""
<mujoco model="lego">
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <map znear="0.002" shadowclip="0.3" shadowscale="0.6"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.96 0.96 0.95" width="64" height="64"/>
    <material name="floor" rgba="0.97 0.97 0.96 1" specular="0.05"/>
    {mats}
  </asset>
  <worldbody>
    <light directional="true" dir="-0.4 -0.6 -1" diffuse="0.45 0.45 0.44" specular="0.3 0.3 0.3" castshadow="true"/>
    <light directional="true" dir="0.6 0.3 -0.8" diffuse="0.15 0.15 0.15" castshadow="false"/>
    <geom type="plane" size="1 1 0.01" material="floor"/>
    {''.join(parts)}
  </worldbody>
</mujoco>"""


class BrickRenderer:
    def __init__(self, n_bricks: int = 6, width: int = 480, height: int = 360):
        self.m = mujoco.MjModel.from_xml_string(model_xml(n_bricks))
        self.d = mujoco.MjData(self.m)
        self.n = n_bricks
        self.r = mujoco.Renderer(self.m, height, width)
        self.cam = mujoco.MjvCamera()

    def place(self, building, colours=None):
        building = sorted(building, key=lambda b: (b[2], b[0], b[1]))     # colour from the bottom up
        colours = colours or [ORDER[k % len(ORDER)] for k in range(len(building))]
        pts = []
        for k in range(self.n):
            if k < len(building):
                x, y, z, o = building[k]
                w, h = DIMS[o]
                c = np.array([(x + w / 2) * PITCH, (y + h / 2) * PITCH, z * BRICK + BRICK / 2]) * MM
                self.d.mocap_pos[k] = c
                q = np.zeros(4)
                mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), math.pi / 2 if o else 0.0)
                self.d.mocap_quat[k] = q
                mid = self.m.material(f"b{k}").id
                self.m.mat_rgba[mid, :3] = COLOURS[colours[k]] if isinstance(colours[k], str) else colours[k]
                pts += [c - [w * PITCH * MM / 2, h * PITCH * MM / 2, BRICK * MM / 2],
                        c + [w * PITCH * MM / 2, h * PITCH * MM / 2, BRICK * MM / 2]]
            else:
                self.d.mocap_pos[k] = [0, 0, -1]
        mujoco.mj_forward(self.m, self.d)
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        # stand the building on the floor
        shift = -lo[2]
        for k in range(min(len(building), self.n)):
            self.d.mocap_pos[k, 2] += shift
        mujoco.mj_forward(self.m, self.d)
        self.centre = (lo + hi) / 2 + [0, 0, shift]
        self.extent = float(np.linalg.norm(hi - lo))

    def image(self, building, colours=None, azimuth=135.0, elevation=-28.0, zoom=1.0):
        self.place(building, colours)
        self.cam.lookat[:] = self.centre
        self.cam.distance = max(self.extent, 0.04) * 1.35 / zoom
        self.cam.azimuth, self.cam.elevation = azimuth, elevation
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def random_building(n: int, seed: int = 0):
    """A building grown by adding bricks at random attachment points (not a uniform sample)."""
    from .count import attachments, canon
    rng = np.random.default_rng(seed)
    b = ((0, 0, 0, 0),)
    while len(b) < n:
        opts = list(attachments(b))
        b = b + (opts[rng.integers(len(opts))],)
    return canon(b)
