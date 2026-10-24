"""The Martian: a walking tripod that burns what it passes.

Ties the tumbling gait to the Heat-Ray. Each target is a building or a tree
(a body whose geoms are named "burn_..."). The Martian picks the nearest one
not yet alight, swings the projector onto it, and holds the beam there until
it catches; fire spreads through the rest of that building on its own. The
ray only fires when the projector is on target and the hood is nearly level,
so a topple does not sweep the beam across the fields.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .params import Tripod, DEFAULT
import math

from .gait import GaitConfig, TumbleGait, PredictiveConfig, PredictiveGait
from .heatray import HeatRay


class Martian:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, p: Tripod = DEFAULT,
                 gait: str = "predictive", reach: float = 80.0, walk: bool = True, max_tilt_deg: float = 3.0):
        self.m = model
        self.walk = walk
        self.armed = True                         # set False to hold fire
        if gait == "transfer":
            self.gait = TumbleGait(p, model, data, GaitConfig())
        else:
            self.gait = PredictiveGait(p, model, data, PredictiveConfig())
        self.max_tilt = math.radians(max_tilt_deg)
        self.ray = HeatRay(model)
        self.reach = reach
        self.targets: dict[int, list[int]] = {}
        for g in self.ray.burnable:
            self.targets.setdefault(int(model.geom_bodyid[g]), []).append(g)
        self.target: int | None = None
        self.hood = model.body("hood").id

    def target_point(self, data: mujoco.MjData, body: int) -> np.ndarray:
        geoms = self.targets[body]
        return np.mean([data.geom_xpos[g] for g in geoms], axis=0)

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

    def __call__(self, data: mujoco.MjData) -> dict:
        info = self.gait(data) if self.walk else {"two_legged": False}
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
