"""Model-predictive centre-of-mass planning for the tripod.

The transfer gait stops the centre of mass at a support edge before every
step. This planner instead keeps it moving, using the linear inverted
pendulum (LIPM): with the centre of mass at a constant height h, its
horizontal motion obeys

    c'' = w^2 (c - z),    w = sqrt(g / h),

where z is the zero-moment point (ZMP), the point on the ground through which
the net ground force acts. The ZMP can be anywhere inside the support polygon:
the triangle of the three feet, or, with one leg in the air, the line segment
between the other two. That segment constraint is the hinge of the transfer
gait, seen from the planner's side: across the line the centre of mass simply
falls, and the only way to stop it is to put a foot down.

Given the footstep schedule, the planner chooses a ZMP trajectory inside the
moving support polygon so that the centre of mass travels at the walking speed
with smooth accelerations. The problem is a quadratic programme (solved with
OSQP) over a horizon of about two gait cycles. It is re-solved from the measured
state at every change of support, which makes it model-predictive control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import osqp
import scipy.sparse as sp

from .params import Tripod
from .kinematics import slot

G = 9.81


@dataclass
class Schedule:
    """Which feet are down when, and where they land."""
    p: Tripod
    feet0: list[np.ndarray]               # initial foot positions (z ignored)
    t0: float                             # time of step 0's start
    ds_time: float                        # three feet down at the start of each step
    ss_time: float                        # one leg swinging
    stride: float                         # each foot advances this far per cycle
    heading: float = 0.0                  # radians
    order: tuple[int, int, int] = (1, 2, 0)
    n_steps: int = 1000
    targets: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.dir = np.array([math.cos(self.heading), math.sin(self.heading)])
        c = sum(f[:2] for f in self.feet0) / 3
        self.start = c

    @property
    def step_time(self) -> float:
        return self.ds_time + self.ss_time

    @property
    def speed(self) -> float:
        return self.stride / (3 * self.step_time)

    def nominal(self, t: float) -> np.ndarray:
        """Where the centre of mass would be walking steadily on schedule (xy)."""
        lead = self.stride / 3                          # feet land about a third of a stride ahead
        return self.start + self.dir * (self.speed * max(0.0, t - self.t0) + lead * 0.5)

    def step_of(self, t: float) -> tuple[int, float]:
        k = int(math.floor((t - self.t0) / self.step_time))
        return k, t - self.t0 - k * self.step_time

    def target(self, k: int) -> np.ndarray:
        """Where step k's swing foot lands (xy)."""
        if k not in self.targets:
            leg = self.order[k % 3]
            centre = self.start + self.dir * self.stride * (k + 2) / 3
            yaw = math.atan2(self.dir[1], self.dir[0]) - 0.0
            s = slot(self.p, leg, 0.0)[:2]
            self.targets[k] = centre + s
        return self.targets[k]

    def feet_at(self, t: float) -> tuple[list[np.ndarray | None], int | None]:
        """Planted foot xy positions at time t (None for a swinging leg), and the swing leg."""
        k, tau = self.step_of(t)
        feet: list[np.ndarray] = [f[:2].copy() for f in self.feet0]
        for j in range(0, max(k, 0)):                       # completed steps
            feet[self.order[j % 3]] = self.target(j)
        swing = None
        if k >= 0 and tau >= self.ds_time:
            swing = self.order[k % 3]
            feet[swing] = None
        return feet, swing


def support_constraints(feet: list, margin: float, seg_tol: float = 0.05):
    """Linear constraints (rows a, lower, upper) on the ZMP for one instant."""
    pts = [f for f in feet if f is not None]
    rows = []
    if len(pts) == 3:
        c = sum(pts) / 3
        for i in range(3):
            a, b = pts[i], pts[(i + 1) % 3]
            e = b - a
            n = np.array([e[1], -e[0]])
            n /= np.linalg.norm(n)
            if n @ (c - a) > 0:
                n = -n                                     # outward normal
            rows.append((n, -np.inf, n @ a - margin))
    else:
        a, b = pts
        e = b - a
        L = np.linalg.norm(e)
        u = e / L
        n = np.array([-u[1], u[0]])
        rows.append((n, n @ a - seg_tol, n @ a + seg_tol))
        m = min(margin, 0.3 * L)
        rows.append((u, u @ a + m, u @ a + L - m))
    return rows


class COMPlanner:
    def __init__(self, sched: Schedule, com_height: float, dt: float = 0.1, horizon: float = 14.0,
                 margin: float = 1.5, w_vel: float = 1.0, w_zmp_rate: float = 30.0, w_zmp_centre: float = 0.002,
                 w_pos: float = 0.05):
        self.s, self.h, self.dt = sched, com_height, dt
        self.N = int(round(horizon / dt))
        self.w = math.sqrt(G / com_height)
        self.margin = margin
        self.wv, self.wdz, self.wzc, self.wp = w_vel, w_zmp_rate, w_zmp_centre, w_pos
        C, S, w = math.cosh(self.w * dt), math.sinh(self.w * dt), self.w
        self.Ad = np.array([[C, S / w], [w * S, C]])
        self.Bd = np.array([1 - C, -w * S])
        self.plan_t0 = None
        self.X = None
        self.Z = None

    def solve(self, t_now: float, c0: np.ndarray, v0: np.ndarray, planted_now: list[np.ndarray] | None = None) -> bool:
        """Plan from the measured COM state; if that is infeasible, relax the ZMP margin and try again."""
        for m in (self.margin, 0.3 * self.margin, 0.0):
            if self._solve(t_now, c0, v0, planted_now, m):
                return True
        return False

    def _solve(self, t_now, c0, v0, planted_now, margin) -> bool:
        """Variables per knot: x = [cx, cy, vx, vy], z = [zx, zy]."""
        N, dt = self.N, self.dt
        nx, nz = 4 * (N + 1), 2 * N
        nv = nx + nz
        xi = lambda j, k: 4 * j + k                 # k: 0 cx, 1 cy, 2 vx, 3 vy
        zi = lambda j, k: nx + 2 * j + k
        P = sp.lil_matrix((nv, nv))
        q = np.zeros(nv)
        vref = self.s.dir * self.s.speed
        for j in range(1, N + 1):
            cnom = self.s.nominal(t_now + j * dt)
            for k in range(2):
                P[xi(j, 2 + k), xi(j, 2 + k)] += self.wv
                q[xi(j, 2 + k)] -= self.wv * vref[k]
                P[xi(j, k), xi(j, k)] += self.wp           # keep up with the footstep schedule
                q[xi(j, k)] -= self.wp * cnom[k]
        rows, lo, hi = [], [], []
        def add(coeffs, l, u):
            rows.append(coeffs); lo.append(l); hi.append(u)
        # initial state
        for k in range(2):
            add({xi(0, k): 1.0}, c0[k], c0[k])
            add({xi(0, 2 + k): 1.0}, v0[k], v0[k])
        for j in range(N):
            t = t_now + (j + 0.5) * dt
            feet, _ = self.s.feet_at(t)
            if planted_now is not None and j * dt < 0.5:
                feet = [pf[:2] if f is not None else None for f, pf in zip(feet, planted_now)]
            pts = [f for f in feet if f is not None]
            cen = sum(pts) / len(pts)
            for k in range(2):
                # dynamics: c' = C c + S/w v + (1-C) z ; v' = wS c + C v - wS z
                add({xi(j + 1, k): 1.0, xi(j, k): -self.Ad[0, 0], xi(j, 2 + k): -self.Ad[0, 1], zi(j, k): -self.Bd[0]}, 0.0, 0.0)
                add({xi(j + 1, 2 + k): 1.0, xi(j, k): -self.Ad[1, 0], xi(j, 2 + k): -self.Ad[1, 1], zi(j, k): -self.Bd[1]}, 0.0, 0.0)
                P[zi(j, k), zi(j, k)] += self.wzc
                q[zi(j, k)] -= self.wzc * cen[k]
                if j > 0:
                    P[zi(j, k), zi(j, k)] += self.wdz
                    P[zi(j - 1, k), zi(j - 1, k)] += self.wdz
                    P[zi(j, k), zi(j - 1, k)] -= self.wdz
                    P[zi(j - 1, k), zi(j, k)] -= self.wdz
            for n, l, u in support_constraints(feet, margin):
                add({zi(j, 0): n[0], zi(j, 1): n[1]}, l, u)
        A = sp.lil_matrix((len(rows), nv))
        for r, coeffs in enumerate(rows):
            for c, v in coeffs.items():
                A[r, c] = v
        prob = osqp.OSQP()
        prob.setup(sp.csc_matrix(P * 2), q * 2, sp.csc_matrix(A), np.array(lo), np.array(hi),
                   verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polishing=True)
        res = prob.solve()
        if res.info.status_val not in (1, 2):             # solved / solved inaccurate
            return False
        x = res.x
        self.plan_t0 = t_now
        self.X = x[:nx].reshape(N + 1, 4)
        self.Z = x[nx:].reshape(N, 2)
        return True

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """COM position, velocity, acceleration (xy) and ZMP at time t, integrating the plan exactly."""
        j = int((t - self.plan_t0) // self.dt)
        j = max(0, min(j, self.N - 1))
        tau = t - self.plan_t0 - j * self.dt
        c, v, z = self.X[j, :2], self.X[j, 2:], self.Z[j]
        w = self.w
        C, S = math.cosh(w * tau), math.sinh(w * tau)
        cc = z + (c - z) * C + v * S / w
        vv = (c - z) * w * S + v * C
        aa = w * w * (cc - z)
        return cc, vv, aa, z
