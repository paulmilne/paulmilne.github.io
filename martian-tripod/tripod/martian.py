"""The Martian: a walking tripod that stops to look round, and burns what it sees.

Ties the gait to the Heat-Ray, and adds one behaviour from the book: the
fighting-machines halt, rear up to their full height and stand scanning the
country before moving on.

  WALK   The predictive tumbling gait.
  LOOK   At the end of a step (all three feet down) the gait is stopped. The
         centre of mass rises 7 m (the hood about 10 m, to 40 m) on straightening legs, so the thighs, horizontal
         while walking, slope steeply down to the knees; it slews from side to
         side; then it settles back to walking height and a fresh gait takes
         over from wherever the feet now are. It has to stop: raised that high,
         the legs cannot reach a full stride ahead.

Throughout, each target is a building or a tree (a body whose geoms are named
"burn_..."). The Martian swings the projector onto the nearest one not yet
alight and holds the beam there until it catches; fire spreads through the
rest of that building on its own. The ray only fires while the hood is nearly
level, so a topple does not sweep the beam across the fields.
"""
from __future__ import annotations

import math

import numpy as np
import mujoco

from .params import Tripod, DEFAULT
from .gait import GaitConfig, TumbleGait, PredictiveConfig, PredictiveGait, smoothstep
from .heatray import HeatRay
from .control import BodyController
from .kinematics import rotz


class Martian:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, p: Tripod = DEFAULT,
                 gait: str = "predictive", reach: float = 80.0, walk: bool = True, max_tilt_deg: float = 3.0,
                 look_every: float = 22.0, look_time: float = 16.0, rise: float = 7.0, scan_deg: float = 25.0):
        self.m, self.p = model, p
        self.walk = walk
        self.armed = True                         # set False to hold fire
        self.gait_kind = gait
        self.ctl = BodyController(p, model)
        self.gait = self._new_gait(data)
        self.ray = HeatRay(model)
        self.reach = reach
        self.targets: dict[int, list[int]] = {}
        for g in self.ray.burnable:
            self.targets.setdefault(int(model.geom_bodyid[g]), []).append(g)
        self.target: int | None = None
        self.hood = model.body("hood").id
        self.max_tilt = math.radians(max_tilt_deg)
        # looking round
        self.look_every, self.look_time, self.rise = look_every, look_time, rise
        self.scan = math.radians(scan_deg)
        self.mode = "walk"
        self.mode_t = data.time
        self.look_from = None

    def _new_gait(self, data, com_height: float | None = None):
        if self.gait_kind == "transfer":
            return TumbleGait(self.p, self.m, data, GaitConfig(), controller=self.ctl)
        # carry on the leg rhythm where it left off: the rearmost foot steps first
        base = PredictiveConfig().order
        feet = [data.site_xpos[self.m.site(f"foot{i}").id][:2] for i in range(3)]
        c = sum(feet) / 3
        h = math.radians(PredictiveConfig().heading_deg)
        along = [float((f - c) @ np.array([math.cos(h), math.sin(h)])) for f in feet]
        first = int(np.argmin(along))
        k = base.index(first)
        order = base[k:] + base[:k]
        return PredictiveGait(self.p, self.m, data, PredictiveConfig(order=order), controller=self.ctl,
                              com_height=com_height)

    # ------------------------------------------------------------ targets
    def target_point(self, data: mujoco.MjData, body: int) -> np.ndarray:
        return np.mean([data.geom_xpos[g] for g in self.targets[body]], axis=0)

    def alight(self, body: int) -> bool:
        return any(self.ray.heat[g] >= self.ray.ignite for g in self.targets[body])

    def pick(self, data: mujoco.MjData) -> int | None:
        here = data.xpos[self.hood]
        best, best_d = None, self.reach
        for b in self.targets:
            if self.alight(b):
                continue
            d = np.linalg.norm((self.target_point(data, b) - here)[:2])
            if d < best_d:
                best, best_d = b, d
        return best

    # ------------------------------------------------------------ looking round
    def _look(self, data: mujoco.MjData) -> None:
        """Rise, scan, settle: a held stance with a moving COM-height and hood-yaw goal."""
        T, t = self.look_time, data.time - self.mode_t
        com0, yaw0, feet_c = self.look_from
        tr = 0.25 * T                                         # rise, scan, settle in 1 : 2 : 1
        a, da, dda = smoothstep(t / tr)
        b, db, ddb = smoothstep((t - (T - tr)) / tr)
        up = a * (1 - b)
        dup = (da * (1 - b) - a * db) / tr
        ddup = (dda * (1 - b) - 2 * da * db - a * ddb) / tr ** 2
        scan_t = (t - tr) / (T - 2 * tr)
        yaw = yaw0 + (self.scan * math.sin(2 * math.pi * scan_t) if 0 < scan_t < 1 else 0.0)
        goal = (1 - a) * com0 + a * feet_c                    # drift over the middle of the feet as it rises
        goal[2] = com0[2] + self.rise * up
        vel = (feet_c - com0) * da / tr
        vel[2] = self.rise * dup
        acc = np.array([0.0, 0.0, self.rise * ddup])
        self.ctl(data, [0, 1, 2], goal, vel, rotz(yaw), com_acc_ff=acc)

    def __call__(self, data: mujoco.MjData) -> dict:
        info = {"two_legged": False}
        if not self.walk:
            pass
        elif self.mode == "walk":
            info = self.gait(data)
            due = self.look_every > 0 and data.time - self.mode_t >= self.look_every
            if due and not info["two_legged"] and info.get("phase") != "swing":
                com, _ = self.ctl.com_state(data)
                R = data.xmat[self.hood].reshape(3, 3)
                feet = sum(self.gait.planted) / 3
                feet_c = np.array([feet[0], feet[1], com[2]])
                self.look_from = (com.copy(), math.atan2(R[1, 0], R[0, 0]), feet_c)
                self.mode, self.mode_t = "look", data.time
                self._look(data)
        else:
            com, vcom = self.ctl.com_state(data)
            settled = abs(com[2] - self.look_from[0][2]) < 0.3 and np.linalg.norm(vcom) < 0.3
            if data.time - self.mode_t >= self.look_time and (settled or data.time - self.mode_t > self.look_time + 4):
                self.gait = self._new_gait(data, self.look_from[0][2])   # walk on from wherever the feet are
                self.mode, self.mode_t = "walk", data.time
                info = self.gait(data)
            else:
                self._look(data)
        info["mode"] = self.mode
        if self.target is None or self.alight(self.target):
            self.target = self.pick(data)
        if self.target is not None:
            self.ray.aim(data, self.target_point(data, self.target))
            level = data.xmat[self.hood][8] > math.cos(self.max_tilt)
            self.ray.firing = self.armed and self.ray.on_target() and level
        else:
            self.ray.firing = False
        self.ray.step(data, self.m.opt.timestep)
        info["target"] = self.m.body(self.target).name if self.target is not None else None
        return info

    def decorate(self, scene: mujoco.MjvScene, data: mujoco.MjData) -> None:
        self.ray.decorate(scene, data)
