"""Manoeuvre models: impulsive and finite burns given in the Apollo LVLH frame.

Apollo maneuver PADs give the velocity change as "Noun 81" components in the
Local Vertical / Local Horizontal frame of the body being orbited:
  X  along the horizontal component of velocity (forward),
  Y  opposite the orbit angular momentum (so +Y points to the right of track
     for a prograde orbit),
  Z  towards the body's centre (down).
"""
import numpy as np
import dynamics as D

FT = 0.3048e-3           # km per foot
VE_SPS = 3.10            # km/s, SPS effective exhaust velocity (Isp about 316 s)


def lvlh(r, v):
    rh = r / np.linalg.norm(r)
    vh = v - np.dot(v, rh) * rh
    xh = vh / np.linalg.norm(vh)
    zh = -rh
    yh = np.cross(zh, xh)
    return xh, yh, zh


def rel_state(t, y, body):
    if body == "moon":
        return y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)
    return y[:3], y[3:]


def rotate_dir(d, pitch, yaw):
    """Small rotation of an LVLH direction: pitch about Y (towards -Z), yaw about Z."""
    x, y, z = d
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    x, z = cp * x + sp * z, -sp * x + cp * z
    x, y = cy * x - sy * y, sy * x + cy * y
    return np.array([x, y, z])


def impulsive(t, y, dv_fps, body="earth"):
    r, v = rel_state(t, y, body)
    x, yy, z = lvlh(r, v)
    d = np.asarray(dv_fps, float) * FT
    return np.r_[y[:3], y[3:] + d[0] * x + d[1] * yy + d[2] * z]


def finite_thrust(t_ign, dir_lvlh, a0, ve=VE_SPS, body="moon", pitch_rate=0.0):
    """Thrust acceleration function for D.make_rhs: constant thrust, mass falling
    linearly, direction fixed in the rotating LVLH frame apart from an optional
    steady pitch rate (rad/s)."""
    u0 = np.asarray(dir_lvlh, float)
    u0 = u0 / np.linalg.norm(u0)

    def thrust(t, r, v):
        tau = t - t_ign
        u = rotate_dir(u0, pitch_rate * tau, 0.0) if pitch_rate else u0
        a = a0 / (1.0 - a0 * tau / ve)
        if body == "moon":
            rr, vv = r - D.moon_pos(t), v - D.moon_vel(t)
        else:
            rr, vv = r, v
        x, y, z = lvlh(rr, vv)
        return a * (u[0] * x + u[1] * y + u[2] * z)
    return thrust


def a0_for(dv_kms, duration, ve=VE_SPS):
    """Initial acceleration of a constant-thrust burn that gives dv in the given time."""
    return ve * (1.0 - np.exp(-dv_kms / ve)) / duration


def burn(t_ign, y, duration, dir_lvlh, dv_kms, body="moon", t_out=None, ve=VE_SPS, pitch_rate=0.0):
    a0 = a0_for(dv_kms, duration, ve)
    rhs = D.make_rhs(finite_thrust(t_ign, dir_lvlh, a0, ve, body, pitch_rate))
    ts = ([] if t_out is None else list(t_out)) + [t_ign + duration]
    return D.integrate(rhs, t_ign, y, ts, h_max=2.0)
