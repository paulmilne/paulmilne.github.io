"""Equations of motion and an adaptive integrator (numpy only).

Force model: Earth point mass plus J2 oblateness, and the Moon and Sun as
third bodies (with the indirect term, because the origin is Earth's centre).
Moon and Sun states come from ephem_1969.csv, interpolated with cubic Hermite
polynomials, which reproduce DE421 to metres at a one-hour table step.
"""
from pathlib import Path
import numpy as np

MU_EARTH = 398600.4418     # km^3/s^2
MU_MOON = 4902.800066
MU_SUN = 1.32712440018e11
J2 = 1.08263e-3
R_J2 = 6378.137            # reference radius for J2
R_MOON = 1737.4

_EPH = np.loadtxt(Path(__file__).parent / "ephem_1969.csv", delimiter=",", skiprows=1)
_T = _EPH[:, 0]


def _hermite(cols_p, cols_v, t):
    i = int(np.clip(np.searchsorted(_T, t) - 1, 0, len(_T) - 2))
    t0, t1 = _T[i], _T[i + 1]
    h = t1 - t0
    s = (t - t0) / h
    p0, p1 = _EPH[i, cols_p], _EPH[i + 1, cols_p]
    v0, v1 = _EPH[i, cols_v] * h, _EPH[i + 1, cols_v] * h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def moon_pos(t):
    return _hermite([1, 2, 3], [4, 5, 6], t)


def moon_vel(t, dt=1.0):
    return (moon_pos(t + dt) - moon_pos(t - dt)) / (2 * dt)


def sun_pos(t):
    return _hermite([7, 8, 9], [10, 11, 12], t)


def _third_body(r, rb, mu):
    d = rb - r
    return mu * (d / np.linalg.norm(d) ** 3 - rb / np.linalg.norm(rb) ** 3)


# Low-degree lunar gravity field (unnormalised), reference radius 1738.0 km.
# Values from modern lunar gravity models (GRAIL era); the large mascon terms are
# not included, which is discussed in Part 2.
MOON_J2 = 2.0330e-4
MOON_C22 = 2.2380e-5
MOON_RREF = 1738.0
LUNAR_HARMONICS = True
_ORIENT_CACHE = {}


def _moon_orient(t):
    k = int(t // 30)
    m = _ORIENT_CACHE.get(k)
    if m is None:
        import moon_frame
        m = moon_frame.body_to_work(k * 30.0 + 15.0)
        if len(_ORIENT_CACHE) > 20000:
            _ORIENT_CACHE.clear()
        _ORIENT_CACHE[k] = m
    return m


def lunar_harmonics(t, d):
    """Acceleration from lunar J2 and C22 at offset d (working frame) from the Moon's centre."""
    M = _moon_orient(t)
    x, y, z = M.T @ d
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    k2 = 1.5 * MOON_J2 * MU_MOON * MOON_RREF**2 / r**5
    zz = 5 * z * z / r2
    a_j2 = -k2 * np.array([x * (1 - zz), y * (1 - zz), z * (3 - zz)])
    k22 = 3 * MU_MOON * MOON_RREF**2 * MOON_C22
    q = (x * x - y * y)
    a_c22 = k22 * np.array([2 * x / r**5 - 5 * x * q / r**7,
                            -2 * y / r**5 - 5 * y * q / r**7,
                            -5 * z * q / r**7])
    return M @ (a_j2 + a_c22)


def gravity(t, r, moon=True, sun=True):
    rn = np.linalg.norm(r)
    a = -MU_EARTH * r / rn**3
    z2 = (r[2] / rn) ** 2
    k = 1.5 * J2 * MU_EARTH * R_J2**2 / rn**5
    a = a + k * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])
    if moon:
        rm = moon_pos(t)
        a = a + _third_body(r, rm, MU_MOON)
        if LUNAR_HARMONICS:
            d = r - rm
            if d @ d < 20000.0**2:          # only matters close to the Moon
                a = a + lunar_harmonics(t, d)
    if sun:
        a = a + _third_body(r, sun_pos(t), MU_SUN)
    return a


def make_rhs(thrust=None, **kw):
    """thrust(t, r, v) -> acceleration vector (km/s^2), or None for a coast."""
    def rhs(t, y):
        a = gravity(t, y[:3], **kw)
        if thrust is not None:
            a = a + thrust(t, y[:3], y[3:])
        return np.concatenate([y[3:], a])
    return rhs


# Dormand-Prince 5(4) coefficients.
_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
_A = [[],
      [1/5],
      [3/40, 9/40],
      [44/45, -56/15, 32/9],
      [19372/6561, -25360/2187, 64448/6561, -212/729],
      [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
      [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
_B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
_B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])


def integrate(rhs, t0, y0, t_out, rtol=1e-11, atol=1e-9, h0=1.0, h_max=600.0):
    """Integrate y' = rhs(t, y) from t0, returning y at each time in t_out.
    t_out must be monotonic: ascending to integrate forwards, descending to integrate backwards."""
    t, y, h = t0, np.array(y0, float), h0
    t_out = list(t_out)
    sgn = -1.0 if t_out and t_out[0] < t0 else 1.0
    out = []
    for t_target in t_out:
        while sgn * (t_target - t) > 1e-9:
            h = min(h, sgn * (t_target - t), h_max)
            hs = sgn * h
            k = []
            for i in range(7):
                yi = y + hs * sum(_A[i][j] * k[j] for j in range(i)) if i else y
                k.append(rhs(t + _C[i] * hs, yi))
            k = np.array(k)
            y5 = y + hs * (_B5 @ k)
            err = hs * ((_B5 - _B4) @ k)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
            e = np.sqrt(np.mean((err / scale) ** 2))
            if e <= 1.0:
                t, y = t + hs, y5
            h *= min(5.0, max(0.2, 0.9 * (e + 1e-16) ** -0.2))
        out.append(y.copy())
    return np.array(out)
