"""Leg kinematics in the world frame.

Leg i is mounted under the hood at (hip_radius, azimuth_i, -hip_drop) and is
rotated by its azimuth, so its local +x points radially outward. Joint order
along the leg: yaw about local z, hip pitch about y, knee about y.
"""
from __future__ import annotations

import math
import numpy as np

from .params import Tripod, planar_ik


def mount(p: Tripod, i: int) -> tuple[np.ndarray, float]:
    """Mount position in the hood frame and its azimuth (rad)."""
    az = math.radians(p.leg_azimuths_deg[i])
    return np.array([p.hip_radius * math.cos(az), p.hip_radius * math.sin(az), -p.hip_drop]), az


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def foot_ik(p: Tripod, i: int, hood_pos: np.ndarray, hood_R: np.ndarray, foot_world: np.ndarray):
    """Joint angles (yaw, hip, knee) that put foot i's sphere centre at foot_world."""
    m, az = mount(p, i)
    local = rotz(az).T @ (hood_R.T @ (np.asarray(foot_world) - hood_pos) - m)
    yaw = math.atan2(local[1], local[0])
    q_hip, q_knee = planar_ik(p.leg.thigh, p.leg.shin, math.hypot(local[0], local[1]), local[2], p.knee_up)
    return yaw, q_hip, q_knee


def foot_fk(p: Tripod, i: int, hood_pos: np.ndarray, hood_R: np.ndarray, q) -> np.ndarray:
    yaw, q_hip, q_knee = q
    m, az = mount(p, i)
    phi1, phi2 = -q_hip, -(q_hip + q_knee)
    L1, L2 = p.leg.thigh, p.leg.shin
    planar = np.array([L1 * math.cos(phi1) + L2 * math.cos(phi2), 0.0, L1 * math.sin(phi1) + L2 * math.sin(phi2)])
    return hood_pos + hood_R @ (m + rotz(az) @ rotz(yaw) @ planar)


def slot(p: Tripod, i: int, yaw: float = 0.0) -> np.ndarray:
    """Nominal foot position for leg i relative to the hood centre, on the ground plane."""
    az = math.radians(p.leg_azimuths_deg[i]) + yaw
    return np.array([p.stance_radius * math.cos(az), p.stance_radius * math.sin(az), 0.0])
