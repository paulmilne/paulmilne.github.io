"""A three-dimensional tumbling gait: lean out to an edge, topple, catch.

The rimless-wheel study showed that a tripod cannot roll passively: each
footfall stops it dead, so the legs have to carry the body over every step.
This gait does that in 3-D, one leg at a time:

  1. TRANSFER  On three feet, carry the centre of mass out until it sits just
               inside the edge between the two feet that will stay down.
  2. SWING     Lift the third leg. On two feet the machine is a hinge about
               the line between them and topples slowly toward the lifted
               side; the lifted leg swings forward and lands where the body
               will be next, catching the topple.
  3. SETTLE    Back on three feet, steady the hood.

The steps go round the legs in turn, so the hood circles from edge to edge
and the lean swings round the compass every cycle: the "milking stool tilted
and bowled along". An optional spin turns the hood a little on every step.

Phases end on events, not just the clock: the swing does not start until the
centre of mass has actually arrived at the edge and slowed down, and the step
does not end until the swing foot is on the ground. The legs are driven by the
whole-body controller in control.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import mujoco

from .params import Tripod
from .kinematics import rotz, slot
from .control import BodyController


def smoothstep(x: float) -> tuple[float, float, float]:
    """Value, first and second derivative of 3x^2 - 2x^3 on [0, 1]."""
    if x <= 0.0 or x >= 1.0:
        return (0.0 if x <= 0.0 else 1.0), 0.0, 0.0
    return x * x * (3 - 2 * x), 6 * x * (1 - x), 6 - 12 * x


@dataclass
class GaitConfig:
    transfer_time: float = 8.0  # s to carry the centre of mass out to the support edge
    swing_time: float = 1.6     # s the lifted leg is in the air
    settle_time: float = 0.8    # s back on three feet before the next step
    stride: float = 15.0        # m each foot moves forward per cycle
    heading_deg: float = 0.0    # direction of travel in the world
    spin_deg: float = 0.0       # hood yaw added per step (the "bowling" spin)
    margin: float = 3.0         # m the centre of mass stays inside the support edge
    lift: float = 5.0           # m swing-foot clearance
    dip: float = 0.8            # m the body sinks while on two feet
    arrive_tol: float = 0.6     # m: how close the COM must be to the edge point to lift
    arrive_speed: float = 0.4   # m/s: and how slow
    order: tuple[int, int, int] = (1, 2, 0)


class TumbleGait:
    def __init__(self, p: Tripod, model: mujoco.MjModel, data: mujoco.MjData, cfg: GaitConfig | None = None,
                 controller: BodyController | None = None):
        self.p, self.m, self.cfg = p, model, cfg or GaitConfig()
        self.ctl = controller or BodyController(p, model)
        self.site = [model.site(f"foot{i}").id for i in range(3)]
        self.touch = [model.sensor(f"foot{i}_touch").adr[0] for i in range(3)]
        self.hood = model.body("hood").id
        mujoco.mj_forward(model, data)
        self.t0 = data.time
        com, _ = self.ctl.com_state(data)
        self.start = com.copy()
        self.com_z = com[2]
        R = data.xmat[self.hood].reshape(3, 3)
        self.yaw = self.yaw0 = math.atan2(R[1, 0], R[0, 0])
        self.planted = []
        for s in self.site:
            f = data.site_xpos[s].copy()
            f[2] = p.leg.foot_radius
            self.planted.append(f)
        h = math.radians(self.cfg.heading_deg)
        self.dir = np.array([math.cos(h), math.sin(h), 0.0])
        self.start_xy = self.centroid()
        self.k = -1
        self.steps_done = 0
        self._begin_step(data)

    def centroid(self) -> np.ndarray:
        c = sum(self.planted) / 3
        c[2] = 0.0
        return c

    def _begin_step(self, data: mujoco.MjData):
        c, p = self.cfg, self.p
        self.k += 1
        self.swing = c.order[self.k % 3]
        self.stance = [j for j in range(3) if j != self.swing]
        self.phase, self.t_phase = "transfer", data.time
        com, _ = self.ctl.com_state(data)
        self.from_com = com
        self.from_yaw, self.to_yaw = self.yaw, self.yaw + math.radians(c.spin_deg)
        a, b = (self.planted[j] for j in self.stance)
        mid = 0.5 * (a + b)
        inward = self.planted[self.swing] - mid
        inward[2] = 0.0
        inward /= np.linalg.norm(inward)
        self.edge_axis = (b - a) / np.linalg.norm(b - a)
        self.edge_pt = mid + c.margin * inward
        self.edge_pt[2] = self.com_z - c.dip
        # feet land on a schedule: after step k the support centre should have
        # advanced k+1 thirds of a stride, and the lifted foot lands a third of a
        # stride ahead of that, in its slot around the centre
        centre = self.start_xy + self.dir * c.stride * (self.k + 2) / 3
        tgt = centre + slot(p, self.swing, self.to_yaw)
        tgt[2] = p.leg.foot_radius
        self.swing_from = self.planted[self.swing].copy()
        self.swing_target = tgt

    def plan(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Planned COM position, velocity, acceleration and hood yaw."""
        c = self.cfg
        if self.phase == "transfer":
            T = c.transfer_time
            u, du, ddu = smoothstep((t - self.t_phase) / T)
            d = self.edge_pt - self.from_com
            return self.from_com + u * d, d * du / T, d * ddu / T ** 2, self.from_yaw + u * (self.to_yaw - self.from_yaw)
        if self.phase == "swing":
            return self.edge_pt.copy(), np.zeros(3), np.zeros(3), self.to_yaw
        T = c.settle_time
        u, du, ddu = smoothstep((t - self.t_phase) / T)
        pos = self.edge_pt.copy()
        pos[2] = self.com_z - (1 - u) * c.dip
        return pos, np.array([0.0, 0.0, c.dip * du / T]), np.array([0.0, 0.0, c.dip * ddu / T ** 2]), self.to_yaw

    def __call__(self, data: mujoco.MjData) -> dict:
        c, p = self.cfg, self.p
        t = data.time
        com, vcom = self.ctl.com_state(data)
        if self.phase == "transfer" and t - self.t_phase >= c.transfer_time:
            near = np.linalg.norm((com - self.edge_pt)[:2]) < c.arrive_tol
            slow = np.linalg.norm(vcom[:2]) < c.arrive_speed
            if (near and slow) or t - self.t_phase > 2 * c.transfer_time:
                self.phase, self.t_phase = "swing", t
        elif self.phase == "swing" and t - self.t_phase >= c.swing_time:
            down = data.sensordata[self.touch[self.swing]] > 0.05 * 9.81 * self.ctl.mass
            if down or t - self.t_phase > c.swing_time + 1.5:
                landed = data.site_xpos[self.site[self.swing]].copy()
                landed[2] = p.leg.foot_radius
                self.planted[self.swing] = landed
                self.phase, self.t_phase = "settle", t
        elif self.phase == "settle" and t - self.t_phase >= c.settle_time:
            self.yaw = self.to_yaw
            self.steps_done += 1
            self._begin_step(data)

        pos, vel, acc, yaw = self.plan(t)
        if self.phase == "swing":
            v = min(1.0, (t - self.t_phase) / c.swing_time)
            u = smoothstep(v)[0]
            f = (1 - u) * self.swing_from + u * self.swing_target
            f[2] += c.lift * math.sin(math.pi * v)
            if v >= 1.0:                            # feel for the ground
                f[2] -= 0.6 * (t - self.t_phase - c.swing_time)
            self.ctl(data, self.stance, pos, vel, rotz(yaw), swing={self.swing: f}, hinge=self.edge_axis,
                     com_acc_ff=acc)
        else:
            self.ctl(data, [0, 1, 2], pos, vel, rotz(yaw), com_acc_ff=acc)
        return {"step": self.k, "phase": self.phase, "swing": self.swing, "two_legged": self.phase == "swing"}

    @property
    def speed(self) -> float:
        c = self.cfg
        return c.stride / (3 * (c.transfer_time + c.swing_time + c.settle_time))
