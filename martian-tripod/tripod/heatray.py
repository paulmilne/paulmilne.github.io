"""The Heat-Ray: a projector on a jointed arm under the front of the hood.

Wells describes it as "an invisible, inevitable sword of heat": you do not
see the beam, only what it touches - a flash, then flame. Here the beam is a
ray cast from the projector's muzzle along its axis (MuJoCo's mj_ray), so it
stops at the first thing it meets; it passes through the machine's own geoms
(group 1).
Anything it rests on heats up; a geom whose name starts "burn_" glows, catches
fire and is left charred.

Aiming solves the projector's yaw and pitch for a point in the world; the
arm's two position servos do the rest, so the beam lags and swings like a
real turret. The beam is only drawn faintly, and only as a heat shimmer.
"""
from __future__ import annotations

import math
import numpy as np
import mujoco

TRIPOD_GROUP = 1                     # geom group of the machine itself (the ray ignores it)


class HeatRay:
    def __init__(self, model: mujoco.MjModel, power: float = 1.0, ignite: float = 1.5, char: float = 6.0,
                 aim_tol: float = 0.015):
        self.m = model
        self.arm = model.body("heatray").id
        self.hood = model.body("hood").id
        self.muzzle = model.site("muzzle").id
        self.yaw_q = model.joint("ray_yaw").qposadr[0]
        self.pitch_q = model.joint("ray_pitch").qposadr[0]
        self.yaw_a = model.actuator("ray_yaw").id
        self.pitch_a = model.actuator("ray_pitch").id
        self.power, self.ignite, self.char, self.aim_tol = power, ignite, char, aim_tol
        self.burnable = {i for i in range(model.ngeom) if model.geom(i).name.startswith("burn_")}
        self.rgba0 = {g: model.geom_rgba[g].copy() for g in self.burnable}
        self.heat = {g: 0.0 for g in self.burnable}
        self.groups = np.array([1 if k != TRIPOD_GROUP else 0 for k in range(6)], dtype=np.uint8)
        self.firing = False
        self.hit_point: np.ndarray | None = None
        self.hit_geom = -1
        self.aim_error = math.inf
        self._geomid = np.array([-1], dtype=np.int32)
        self.t = 0.0

    # ------------------------------------------------------------ aiming
    def aim(self, data: mujoco.MjData, target: np.ndarray) -> float:
        """Command the turret toward a world point. Returns the current pointing error (rad)."""
        R = data.xmat[self.hood].reshape(3, 3)
        origin = data.xpos[self.arm]
        v = R.T @ (np.asarray(target) - origin)
        yaw = math.atan2(v[1], v[0])
        pitch = math.atan2(-v[2], math.hypot(v[0], v[1]))
        cur = data.qpos[self.yaw_q]
        yaw = cur + (yaw - cur + math.pi) % (2 * math.pi) - math.pi      # turn the short way
        data.ctrl[self.yaw_a] = yaw
        data.ctrl[self.pitch_a] = min(max(pitch, -0.4), 1.4)
        axis = data.site_xmat[self.muzzle].reshape(3, 3)[:, 0]
        want = np.asarray(target) - data.site_xpos[self.muzzle]
        want /= np.linalg.norm(want)
        self.aim_error = math.acos(max(-1.0, min(1.0, float(axis @ want))))
        return self.aim_error

    def on_target(self) -> bool:
        return self.aim_error < self.aim_tol

    # ------------------------------------------------------------ firing
    def step(self, data: mujoco.MjData, dt: float) -> None:
        """Advance the beam and the fires by dt seconds of simulated time."""
        self.t += dt
        self.hit_point, self.hit_geom = None, -1
        if self.firing:
            p = data.site_xpos[self.muzzle].copy()
            axis = data.site_xmat[self.muzzle].reshape(3, 3)[:, 0].copy()
            dist = mujoco.mj_ray(self.m, data, p, axis, self.groups, 1, self.arm, self._geomid)
            if dist >= 0:
                self.hit_point = p + dist * axis
                self.hit_geom = int(self._geomid[0])
                if self.hit_geom in self.heat:
                    self.heat[self.hit_geom] += self.power * dt
                    # heat spreads to the rest of the same building or tree
                    body = self.m.geom_bodyid[self.hit_geom]
                    for g in self.burnable:
                        if g != self.hit_geom and self.m.geom_bodyid[g] == body:
                            self.heat[g] += 0.35 * self.power * dt
        for g, h in self.heat.items():
            if h <= 0:
                continue
            if h < self.ignite:                                   # scorching
                k = h / self.ignite
                c = (1 - k) * self.rgba0[g][:3] + k * np.array([0.55, 0.32, 0.15])
            elif h < self.char:                                   # alight
                flick = 0.5 + 0.5 * math.sin(37 * self.t + g)
                c = np.array([0.95, 0.45 + 0.25 * flick, 0.10])
            else:                                                 # charred
                c = np.array([0.07, 0.06, 0.05])
            self.m.geom_rgba[g][:3] = c
            if h >= self.ignite:
                self.heat[g] = h + 0.15 * dt                      # fire keeps itself going

    def burning(self) -> list[int]:
        return [g for g, h in self.heat.items() if self.ignite <= h < self.char + 4.0]

    # ------------------------------------------------------------ drawing
    def decorate(self, scene: mujoco.MjvScene, data: mujoco.MjData) -> None:
        """Add the beam shimmer, the flash where it lands, and flames, to a scene."""
        def add(gtype, size, pos, rgba, emission=1.0):
            if scene.ngeom >= scene.maxgeom:
                return None
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(),
                                np.asarray(rgba, np.float32))
            try:
                g.emission = emission
            except AttributeError:
                pass
            scene.ngeom += 1
            return g

        if self.firing:
            start = data.site_xpos[self.muzzle]
            end = self.hit_point if self.hit_point is not None else \
                start + 400 * data.site_xmat[self.muzzle].reshape(3, 3)[:, 0]
            g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [0.4, 0, 0], start, [1.0, 0.95, 0.8, 0.10], 0.4)
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.4, np.asarray(start, float), np.asarray(end, float))
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.7, 0, 0], start, [1.0, 0.95, 0.7, 0.9], 1.0)
            if self.hit_point is not None:
                r = 1.4 + 0.5 * math.sin(53 * self.t)
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], self.hit_point, [1.0, 0.92, 0.6, 0.85], 1.0)
        for g in self.burning():
            pos = data.geom_xpos[g]
            size = self.m.geom_size[g]
            top = pos[2] + (size[2] if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE else size[0])
            for k in range(3):
                ph = self.t * (3 + k) + g * 1.7 + k
                h = 1.2 + 0.8 * (0.5 + 0.5 * math.sin(ph))
                off = np.array([math.sin(ph * 0.7) * size[0] * 0.4, math.cos(ph * 0.9) * size[1] * 0.4, top + h * (0.6 + 0.5 * k)])
                add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.7 + 0.2 * k, 0.7 + 0.2 * k, h], [pos[0] + off[0], pos[1] + off[1], off[2]],
                    [1.0, 0.55 - 0.15 * k, 0.1, 0.75 - 0.15 * k], 1.0)
