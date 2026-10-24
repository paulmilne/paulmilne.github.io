"""Whole-body balance controller for the tripod.

Joint-angle servos are the wrong tool for a 248-tonne machine on point feet:
a rigid topple about the support leaves every joint angle unchanged, so the
servos never see it, and when their targets disagree about where the body is,
a foot lifts. So the legs are torque-controlled instead, in the way most
legged robots are:

  1. From the centre of mass and hood orientation, compute the force and
     moment the ground must apply to bring the body back on plan
     (a spring-damper on the COM, another on the hood orientation, plus weight).
  2. Share that wrench between the feet that are down, as a least-squares
     solve that keeps every foot pushing (no pulling on the ground) and inside
     its friction cone.
  3. Map each foot's force to its joint torques through the leg Jacobian,
     adding the torque that holds up the legs' own weight.

A swinging leg is driven by a joint PD to the inverse-kinematics solution of
its foot target, plus gravity compensation.

On two feet the moment about the line through them cannot be produced at all;
that component is dropped rather than fought, and the machine topples about
that line at whatever rate gravity dictates.
"""
from __future__ import annotations

import math
import numpy as np
import mujoco

from .params import Tripod
from .kinematics import foot_ik


def axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * ang / (2 * math.sin(ang))


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class BodyController:
    def __init__(self, p: Tripod, model: mujoco.MjModel, com_freq: float = 0.5, att_freq: float = 0.4,
                 damping: float = 1.0, mu: float = 0.7, posture_kp: float = 1.5e8, posture_kd: float = 3.0e7):
        self.p, self.m = p, model
        self.hood = model.body("hood").id
        self.mass = float(model.body_subtreemass[self.hood])
        self.dofs = [list(range(model.joint(f"leg{i}_yaw").dofadr[0], model.joint(f"leg{i}_yaw").dofadr[0] + 3)) for i in range(3)]
        self.qadr = [list(range(model.joint(f"leg{i}_yaw").qposadr[0], model.joint(f"leg{i}_yaw").qposadr[0] + 3)) for i in range(3)]
        self.act = [[model.actuator(f"leg{i}_{j}").id for j in ("yaw", "hip", "knee")] for i in range(3)]
        self.shin = [model.body(f"leg{i}_shin").id for i in range(3)]
        self.site = [model.site(f"foot{i}").id for i in range(3)]
        w = 2 * math.pi * com_freq
        self.kp_com, self.kd_com = self.mass * w * w, 2 * damping * self.mass * w
        # attitude gains in torque units, scaled by the machine's pitch inertia about its feet (~ m h^2)
        h = p.stance_height * 0.85
        I = self.mass * h * h
        wa = 2 * math.pi * att_freq
        self.kp_att, self.kd_att = I * wa * wa, 2 * damping * I * wa
        self.mu = mu
        self.posture_kp, self.posture_kd = posture_kp, posture_kd
        self.jacp = np.zeros((3, model.nv))
        self.last_forces = np.zeros((3, 3))

    def com_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_subtreeVel(self.m, data)
        return data.subtree_com[self.hood].copy(), data.subtree_linvel[self.hood].copy()

    def foot_point(self, data: mujoco.MjData, i: int) -> np.ndarray:
        f = data.site_xpos[self.site[i]].copy()
        f[2] -= self.p.leg.foot_radius
        return f

    def distribute(self, feet: list[np.ndarray], com: np.ndarray, F: np.ndarray, T: np.ndarray,
                   hinge: np.ndarray | None = None) -> list[np.ndarray]:
        """Share the wanted force F and moment T (about the COM) between the planted feet.

        Weighted least squares with an active set: a foot that would have to pull
        on the ground is fixed at a light push and the rest re-solved, so the feet
        never ask for a force the ground cannot give. Moments are weighted by
        1/height so a metre of lever and a newton of force count alike.
        """
        n = len(feet)
        fmin = 0.02 * self.mass * 9.81 / n
        wT = 1.0 / (0.85 * self.p.stance_height)
        Pm = np.eye(3)
        if hinge is not None:                       # the moment about the support line is unavailable
            Pm = Pm - np.outer(hinge, hinge)
        fixed: set[int] = set()
        forces = [np.array([0.0, 0.0, fmin]) for _ in range(n)]
        for _ in range(n):
            free = [k for k in range(n) if k not in fixed]
            if not free:
                break
            A = np.zeros((6, 3 * len(free)))
            for j, k in enumerate(free):
                A[:3, 3 * j:3 * j + 3] = np.eye(3)
                A[3:, 3 * j:3 * j + 3] = wT * Pm @ skew(feet[k] - com)
            b = np.concatenate([F, wT * Pm @ T])
            for k in fixed:
                fk = np.array([0.0, 0.0, fmin])
                b[:3] -= fk
                b[3:] -= wT * Pm @ skew(feet[k] - com) @ fk
            AtA = A.T @ A
            f = np.linalg.solve(AtA + 1e-6 * np.trace(AtA) / len(AtA) * np.eye(len(AtA)), A.T @ b)
            low = [k for j, k in enumerate(free) if f[3 * j + 2] < fmin]
            for j, k in enumerate(free):
                forces[k] = f[3 * j:3 * j + 3].copy()
            if not low:
                break
            fixed.update(low)
            for k in low:
                forces[k] = np.array([0.0, 0.0, fmin])
        for fk in forces:                           # stay inside the friction cone
            fk[2] = max(fk[2], fmin)
            t = math.hypot(fk[0], fk[1])
            if t > self.mu * fk[2]:
                fk[:2] *= self.mu * fk[2] / t
        return forces

    def __call__(self, data: mujoco.MjData, stance: list[int], com_goal: np.ndarray, com_vel_goal: np.ndarray,
                 R_goal: np.ndarray, swing: dict[int, np.ndarray] | None = None, hinge: np.ndarray | None = None,
                 com_acc_ff: np.ndarray | None = None) -> None:
        p = self.p
        com, vcom = self.com_state(data)
        R = data.xmat[self.hood].reshape(3, 3)
        omega = data.cvel[self.hood][:3].copy()          # world-frame angular velocity
        acc = self.kp_com * (com_goal - com) + self.kd_com * (com_vel_goal - vcom)
        if com_acc_ff is not None:
            acc = acc + self.mass * com_acc_ff
        F = acc + np.array([0.0, 0.0, self.mass * 9.81])
        T = self.kp_att * axis_angle(R_goal @ R.T) - self.kd_att * omega
        feet = [self.foot_point(data, i) for i in stance]
        forces = self.distribute(feet, com, F, T, hinge)
        tau = np.zeros(self.m.nu)
        bias = data.qfrc_bias
        hood_pos = data.xpos[self.hood]
        for k, i in enumerate(stance):
            mujoco.mj_jac(self.m, data, self.jacp, None, feet[k], self.shin[i])
            J = self.jacp[:, self.dofs[i]]
            t = bias[self.dofs[i]] - J.T @ forces[k]
            # posture: the contact forces only act on the machine as a whole; the hood's
            # attitude relative to its legs is set here, by steering each stance leg's
            # joints toward the shape that holds the hood at its goal attitude
            try:
                q_post = np.array(foot_ik(p, i, hood_pos, R_goal, data.site_xpos[self.site[i]]))
                t = t + self.posture_kp * (q_post - data.qpos[self.qadr[i]]) - self.posture_kd * data.qvel[self.dofs[i]]
            except ValueError:
                pass
            for a, v in zip(self.act[i], t):
                tau[a] = v
            self.last_forces[i] = forces[k]
        for i, foot_target in (swing or {}).items():
            q = data.qpos[self.qadr[i]]
            qd = data.qvel[self.dofs[i]]
            try:
                q_goal = np.array(foot_ik(p, i, data.xpos[self.hood], R, foot_target))
            except ValueError:
                q_goal = q.copy()
            t = p.kp * (q_goal - q) - p.kv * qd + bias[self.dofs[i]]
            for a, v in zip(self.act[i], t):
                tau[a] = v
            self.last_forces[i] = 0.0
        data.ctrl[:9] = np.clip(tau[:9], -p.max_torque, p.max_torque)
