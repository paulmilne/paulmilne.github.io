"""Helpers for the lunar phase: Moon-relative elements, radio occultation,
the landing-site plane test, and a damped least-squares solver."""
import numpy as np
import dynamics as D
import moon_frame as M

NMI = 1.852
FT = 0.3048e-3
RM = M.R_MOON_REF


def moon_rel(t, y):
    return y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)


def elements(r, v, mu=D.MU_MOON, R=RM):
    """Osculating apoapsis and periapsis heights (km), period (s), inclination (rad), e."""
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1.0 / (2.0 / rn - vn * vn / mu)
    h = np.cross(r, v)
    ev = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(ev)
    period = 2 * np.pi * np.sqrt(a**3 / mu) if a > 0 else np.nan
    return dict(apo=a * (1 + e) - R, peri=a * (1 - e) - R, period=period, e=e, a=a, h=h)


def earth_visible(t, y):
    """Signed clearance (km) of the line of sight to Earth's centre past the lunar limb.
    Positive when Earth is in view; negative when the Moon blocks it."""
    s = y[:3]
    m = D.moon_pos(t)
    u = -s / np.linalg.norm(s)
    w = m - s
    along = w @ u
    if along < 0:
        return 1e4
    return np.linalg.norm(w - along * u) - RM


def crossings(ts, ys, fn=earth_visible):
    """Times where fn changes sign (linear interpolation): list of (t, 'LOS'|'AOS')."""
    f = np.array([fn(t, y) for t, y in zip(ts, ys)])
    out = []
    for i in range(1, len(f)):
        if (f[i - 1] > 0) != (f[i] > 0):
            tc = ts[i - 1] + (ts[i] - ts[i - 1]) * f[i - 1] / (f[i - 1] - f[i])
            out.append((tc, "LOS" if f[i - 1] > 0 else "AOS"))
    return out


def site_offset_km(t, y, lat, lon):
    """Distance (km, on the surface) of a selenographic site from the orbit plane."""
    r, v = moon_rel(t, y)
    h = np.cross(r, v)
    h /= np.linalg.norm(h)
    s = M.from_selenographic(t, lat, lon, 1.0)
    return RM * np.arcsin(np.clip(s @ h, -1, 1))


def solve(res_fn, p0, steps, iters=12, lam=1e-2, log=None):
    """Levenberg-Marquardt with forward-difference Jacobian."""
    p = np.array(p0, float)
    r = res_fn(p)
    for it in range(iters):
        J = np.empty((len(r), len(p)))
        for j in range(len(p)):
            dp = np.zeros(len(p))
            dp[j] = steps[j]
            J[:, j] = (res_fn(p + dp) - r) / steps[j]
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(6):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            rt = res_fn(p + step)
            if rt @ rt < r @ r:
                p, r, lam = p + step, rt, lam * 0.3
                improved = True
                break
            lam *= 10
        if log:
            log(it, p, r)
        if not improved or np.linalg.norm(step) < 1e-7:
            break
    return p, r
