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


# ============================================================================
# The predictive gait: the same footfalls, but the centre of mass never stops.
# ============================================================================

@dataclass
class PredictiveConfig:
    ds_time: float = 0.8        # s on three feet at the start of each step
    ss_time: float = 1.2        # s with one leg in the air
    stride: float = 22.0        # m per cycle on the timeline (the body settles at about 2.5 m/s, 5.5 mph)
    heading_deg: float = 0.0
    lift: float = 4.0           # m swing-foot clearance
    margin: float = 1.5         # m the ZMP keeps inside the support polygon
    horizon: float = 14.0       # s the planner looks ahead
    replan_every: float = 0.4   # s between re-plans (also re-plans at every change of support)
    max_shift: float = 6.0      # m the timeline may be re-anchored per step
    anchor_gain: float = 0.5    # fraction of the body's sideways offset from the timeline taken up each step
    anchor_along: float = 0.5   # and of its offset along the line of travel
    w_pos: float = 0.05         # planner weight on keeping up with the footstep timeline
    order: tuple[int, int, int] = (1, 2, 0)

    # so walk_test can treat both gaits alike
    @property
    def spin_deg(self) -> float:
        return 0.0


class PredictiveGait:
    """Footsteps on a fixed rhythm; the centre of mass planned by model-predictive control.

    The planner (planner.py) chooses a zero-moment-point trajectory inside the
    moving support polygon so the centre of mass keeps travelling at the walking
    speed. On two feet the ZMP is confined to the line between them, so across
    that line the machine falls freely and the next footfall catches it: the
    tumble, now planned rather than waited out.
    """

    def __init__(self, p: Tripod, model: mujoco.MjModel, data: mujoco.MjData, cfg: PredictiveConfig | None = None,
                 controller: BodyController | None = None, start_delay: float = 0.6):
        from .planner import Schedule, COMPlanner
        self.p, self.m, self.cfg = p, model, cfg or PredictiveConfig()
        self.ctl = controller or BodyController(p, model)
        self.site = [model.site(f"foot{i}").id for i in range(3)]
        self.touch = [model.sensor(f"foot{i}_touch").adr[0] for i in range(3)]
        self.hood = model.body("hood").id
        mujoco.mj_forward(model, data)
        self.t0 = data.time
        com, _ = self.ctl.com_state(data)
        self.com_z = com[2]
        R = data.xmat[self.hood].reshape(3, 3)
        self.yaw0 = math.atan2(R[1, 0], R[0, 0])
        self.planted = []
        for s in self.site:
            f = data.site_xpos[s].copy()
            f[2] = p.leg.foot_radius
            self.planted.append(f)
        c = self.cfg
        h = math.radians(c.heading_deg)
        self.dir = np.array([math.cos(h), math.sin(h), 0.0])
        self.sched = Schedule(p, [f.copy() for f in self.planted], data.time + start_delay, c.ds_time, c.ss_time,
                              c.stride, h, c.order)
        self.planner = COMPlanner(self.sched, self.com_z, horizon=c.horizon, margin=c.margin, w_pos=c.w_pos)
        self.last_support = None
        self.last_plan = -1e9
        self.k_seen = -1
        self.swing_from = None
        self.steps_done = 0
        self.plan_failures = 0

    def centroid(self) -> np.ndarray:
        c = sum(self.planted) / 3
        c[2] = 0.0
        return c

    @property
    def speed(self) -> float:
        return self.sched.speed

    def __call__(self, data: mujoco.MjData) -> dict:
        c, p, s = self.cfg, self.p, self.sched
        t = data.time
        k, tau = s.step_of(t)
        if k != self.k_seen:                               # a new step: the last swing foot is down
            if self.k_seen >= 0:
                leg = s.order[self.k_seen % 3]
                landed = data.site_xpos[self.site[leg]].copy()
                landed[2] = p.leg.foot_radius
                self.planted[leg] = landed
                s.targets[self.k_seen] = landed[:2].copy()   # plan with where it really is
                self.steps_done += 1
            self.k_seen = k
        _, swing = s.feet_at(t)
        support = (k, swing)
        com, vcom = self.ctl.com_state(data)
        if support != self.last_support or t - self.last_plan >= c.replan_every:
            if swing is not None and support != self.last_support:
                self.swing_from = self.planted[swing].copy()
                # Step to where the body actually is. The footstep timeline is re-anchored
                # to the measured centre of mass (half the error each step, to smooth out
                # the side-to-side lean), and footholds not yet taken are re-planned from it,
                # so a body that falls behind or drifts sideways is not left reaching for
                # feet placed on the original timeline.
                off = com[:2] - s.nominal(t)
                n = np.linalg.norm(off)
                if n > c.max_shift:
                    off *= c.max_shift / n
                d2 = s.dir
                along = (off @ d2) * d2
                s.start = s.start + c.anchor_gain * (off - along) + c.anchor_along * along
                for kk in [kk for kk in s.targets if kk >= k]:
                    del s.targets[kk]
            ok = self.planner.solve(t, com[:2], vcom[:2], [f.copy() for f in self.planted])
            if not ok:
                self.plan_failures += 1
            self.last_support, self.last_plan = support, t
        if self.planner.X is None:
            cxy, vxy, axy = com[:2], np.zeros(2), np.zeros(2)
        else:
            cxy, vxy, axy, _ = self.planner.reference(t)
        goal = np.array([cxy[0], cxy[1], self.com_z])
        vel = np.array([vxy[0], vxy[1], 0.0])
        acc = np.array([axy[0], axy[1], 0.0])
        R_goal = rotz(self.yaw0)
        if swing is None:
            self.ctl(data, [0, 1, 2], goal, vel, R_goal, com_acc_ff=acc)
            two = False
        else:
            v = min(1.0, (tau - c.ds_time) / c.ss_time)
            u = smoothstep(v)[0]
            tgt = np.array([*s.target(k), p.leg.foot_radius])
            f = (1 - u) * self.swing_from + u * tgt
            f[2] += c.lift * math.sin(math.pi * v)
            if v > 0.9:
                f[2] -= 0.3                                  # reach for the ground
            stance = [j for j in range(3) if j != swing]
            a, b = (self.planted[j] for j in stance)
            axis = (b - a) / np.linalg.norm(b - a)
            self.ctl(data, stance, goal, vel, R_goal, swing={swing: f}, hinge=axis, com_acc_ff=acc)
            two = True
        return {"step": k, "phase": "swing" if two else "support", "swing": swing, "two_legged": two}
