"""Null geodesics in the Kerr metric, traced backward from a camera.

Units are geometrised with G = c = M = 1, so lengths are in units of the
black hole's mass M and the spin parameter a lies in [0, 1).

The ray equations are Hamilton's equations for

    H = [ Delta p_r^2 + p_theta^2 - F(r, theta) ] / (2 rho^2),
    F = P^2 / Delta - (b - a)^2 - cos^2(theta) (b^2 / sin^2(theta) - a^2),
    P = r^2 + a^2 - a b,

with the photon's energy normalised to E = -p_t = 1 and b = p_phi its axial
angular momentum. H = 0 on a null ray. This is the form used by James,
von Tunzelmann, Franklin and Thorne (2015), Class. Quantum Grav. 32, 065001,
derived here directly from the inverse metric.
"""
import math

import numpy as np
from numba import njit, prange

# outcome codes returned by trace()
ESCAPED, CAPTURED, DISK, UNFINISHED = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# metric quantities
# ----------------------------------------------------------------------------
def horizon(a):
    """Outer event horizon r_+ = 1 + sqrt(1 - a^2)."""
    return 1.0 + math.sqrt(1.0 - a * a)


def metric_functions(r, th, a):
    """Delta, rho^2, Sigma^2, lapse alpha, frame-dragging omega and the
    cylindrical radius varpi (Bardeen, Press and Teukolsky 1972 notation)."""
    D = r * r - 2.0 * r + a * a
    rho2 = r * r + a * a * np.cos(th) ** 2
    Sig2 = (r * r + a * a) ** 2 - a * a * D * np.sin(th) ** 2
    alpha = np.sqrt(rho2 * D / Sig2)
    omega = 2.0 * a * r / Sig2
    varpi = np.sqrt(Sig2 / rho2) * np.sin(th)
    return D, rho2, Sig2, alpha, omega, varpi


def metric(r, th, a):
    """Covariant Boyer-Lindquist metric g_{mu nu}, coordinates (t, r, th, phi)."""
    D, rho2, Sig2, _, _, _ = metric_functions(r, th, a)
    s2 = math.sin(th) ** 2
    g = np.zeros((4, 4))
    g[0, 0] = -(1.0 - 2.0 * r / rho2)
    g[0, 3] = g[3, 0] = -2.0 * a * r * s2 / rho2
    g[1, 1] = rho2 / D
    g[2, 2] = rho2
    g[3, 3] = Sig2 * s2 / rho2
    return g


def carter_q(r, th, pth, b, a):
    """Carter's constant Q = p_theta^2 + cos^2(theta) (b^2/sin^2(theta) - a^2)."""
    return pth ** 2 + np.cos(th) ** 2 * (b ** 2 / np.sin(th) ** 2 - a * a)


def hamiltonian(r, th, pr, pth, b, a):
    D = r * r - 2.0 * r + a * a
    rho2 = r * r + a * a * np.cos(th) ** 2
    P = r * r + a * a - a * b
    F = P * P / D - (b - a) ** 2 - np.cos(th) ** 2 * (b * b / np.sin(th) ** 2 - a * a)
    return (D * pr * pr + pth * pth - F) / (2.0 * rho2)


@njit(cache=True, fastmath=False)
def rhs(y, b, a, out):
    """Hamilton's equations d(r, th, phi, p_r, p_th)/dzeta for a photon with
    E = 1 and axial angular momentum b."""
    r, th, pr, pth = y[0], y[1], y[3], y[4]
    s, c = math.sin(th), math.cos(th)
    s2 = s * s
    D = r * r - 2.0 * r + a * a
    dD = 2.0 * r - 2.0
    rho2 = r * r + a * a * c * c
    P = r * r + a * a - a * b
    F = P * P / D - (b - a) ** 2 - c * c * (b * b / s2 - a * a)
    dF_r = 4.0 * r * P / D - P * P * dD / (D * D)
    dF_th = 2.0 * b * b * c / (s2 * s) - 2.0 * a * a * s * c
    N = D * pr * pr + pth * pth - F          # = 2 rho^2 H, zero on shell
    out[0] = D * pr / rho2
    out[1] = pth / rho2
    out[2] = (a * P / D + b / s2 - a) / rho2
    out[3] = -((dD * pr * pr - dF_r) / (2.0 * rho2) - N * 2.0 * r / (2.0 * rho2 * rho2))
    out[4] = -((-dF_th) / (2.0 * rho2) + N * 2.0 * a * a * s * c / (2.0 * rho2 * rho2))


# Dormand-Prince 5(4) tableau
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = np.array([
    [0, 0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0, 0],
    [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
])
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


@njit(cache=True)
def _dp_step(y, h, b, a, sgn, k, ytmp, ynew, err):
    """One Dormand-Prince step of size h along sgn * rhs (sgn = -1 traces
    backward in time). Fills ynew and the embedded error estimate err."""
    for s in range(7):
        for i in range(5):
            acc = y[i]
            for j in range(s):
                acc += h * _A[s, j] * k[j, i]
            ytmp[i] = acc
        rhs(ytmp, b, a, k[s])
        for i in range(5):
            k[s, i] *= sgn
    for i in range(5):
        y5 = y[i]
        y4 = y[i]
        for s in range(7):
            y5 += h * _B5[s] * k[s, i]
            y4 += h * _B4[s] * k[s, i]
        ynew[i] = y5
        err[i] = y5 - y4


@njit(cache=True)
def trace_one(y0, b, a, r_in, r_out, r_esc, tol, max_steps, sgn, path, record):
    """Trace one ray. Returns (outcome, steps) and leaves the final state in
    y0. If record > 0 the first `record` states are written into path.

    The ray stops on escaping past r_esc, on reaching r_+ (1 + 1e-5), or on
    crossing the equator between r_in and r_out (an opaque thin disk); a
    disk crossing is located to within 1e-10 in theta by secant iteration."""
    rplus = 1.0 + math.sqrt(1.0 - a * a)
    k = np.empty((7, 5))
    ytmp = np.empty(5)
    ynew = np.empty(5)
    err = np.empty(5)
    y = y0.copy()
    h = 0.01 * y[0]
    disk_on = r_out > r_in
    n = 0
    nrec = 0
    while n < max_steps:
        if record > 0 and nrec < record:
            for i in range(5):
                path[nrec, i] = y[i]
            nrec += 1
        _dp_step(y, h, b, a, sgn, k, ytmp, ynew, err)
        # error norm: relative in r, absolute in angles, scaled in momenta
        e = 0.0
        for i in range(5):
            if i == 0:
                sc = tol * y[0]
            else:
                sc = tol * (1.0 + abs(y[i]))
            v = abs(err[i]) / sc
            if v > e:
                e = v
        if e > 1.0 or ynew[0] <= 0.0 or ynew[1] <= 0.0 or ynew[1] >= math.pi:
            h *= max(0.2, 0.9 * e ** -0.2) if e > 1.0 else 0.5
            if abs(h) < 1e-14:
                break
            continue
        n += 1
        # thin-disk crossing?
        if disk_on and (y[1] - 0.5 * math.pi) * (ynew[1] - 0.5 * math.pi) <= 0.0 \
                and ynew[1] != y[1]:
            # secant on the step fraction so that theta = pi/2
            f0 = y[1] - 0.5 * math.pi
            f1 = ynew[1] - 0.5 * math.pi
            s0, s1 = 0.0, 1.0
            for _ in range(30):
                s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
                s2 = min(max(s2, 0.0), 1.0)
                _dp_step(y, s2 * h, b, a, sgn, k, ytmp, ynew, err)
                f2 = ynew[1] - 0.5 * math.pi
                s0, f0, s1, f1 = s1, f1, s2, f2
                if abs(f2) < 1e-10:
                    break
            if r_in <= ynew[0] <= r_out:
                for i in range(5):
                    y0[i] = ynew[i]
                return DISK, n
            # missed the disk: take the full step again and carry on
            _dp_step(y, h, b, a, sgn, k, ytmp, ynew, err)
        for i in range(5):
            y[i] = ynew[i]
        if y[0] > r_esc:
            for i in range(5):
                y0[i] = y[i]
            return ESCAPED, n
        if y[0] < rplus * (1.0 + 1e-5):
            for i in range(5):
                y0[i] = y[i]
            return CAPTURED, n
        h *= min(5.0, 0.9 * max(e, 1e-10) ** -0.2)
        # never step further than a fixed fraction of the radius
        hmax = 0.1 * y[0] * y[0] / (1.0 + abs(y[3]) * y[0] + abs(y[4]))
        if h > hmax:
            h = hmax
    for i in range(5):
        y0[i] = y[i]
    return UNFINISHED, n


@njit(parallel=True, cache=True)
def _trace_many(Y, B, a, r_in, r_out, r_esc, tol, max_steps, outcome, nsteps):
    dummy = np.empty((1, 5))
    for i in prange(Y.shape[0]):
        y = Y[i].copy()
        o, n = trace_one(y, B[i], a, r_in, r_out, r_esc, tol, max_steps, -1.0, dummy, 0)
        Y[i] = y
        outcome[i] = o
        nsteps[i] = n


def trace(Y0, B, a, r_in=0.0, r_out=0.0, r_esc=1000.0, tol=1e-9, max_steps=20000):
    """Trace many rays backward from the camera.

    Y0 is (n, 5) of (r, theta, phi, p_r, p_theta) at the camera, B is (n,)
    of b = p_phi. Returns the final states, outcome codes and step counts."""
    Y = np.array(Y0, dtype=np.float64, copy=True)
    B = np.ascontiguousarray(B, dtype=np.float64)
    outcome = np.empty(len(Y), dtype=np.int8)
    nsteps = np.empty(len(Y), dtype=np.int32)
    _trace_many(Y, B, float(a), float(r_in), float(r_out), float(r_esc),
                float(tol), int(max_steps), outcome, nsteps)
    return Y, outcome, nsteps


def trace_path(y0, b, a, r_in=0.0, r_out=0.0, r_esc=60.0, tol=1e-9, n=20000, sgn=-1.0):
    """Trace one ray and return its recorded path (m, 5) and outcome."""
    path = np.zeros((n, 5))
    y = np.array(y0, dtype=np.float64)
    o, m = trace_one(y, float(b), float(a), float(r_in), float(r_out),
                     float(r_esc), tol, n - 1, sgn, path, n - 1)
    m = min(m + 1, n - 1)
    path[m] = y
    return path[: m + 1], o


# ----------------------------------------------------------------------------
# the camera
# ----------------------------------------------------------------------------
class Camera:
    """A pinhole camera at (r, theta, phi = 0), looking at the hole.

    It moves at speed beta (in units of c) in the +phi direction relative
    to the local zero-angular-momentum observer (ZAMO, the "FIDO" of the
    DNGR paper). beta = 0 is a ZAMO camera; beta = 'orbit' puts it on a
    prograde circular equatorial geodesic orbit, as for the film's Endurance.

    Image axes: x to the right (the +phi direction), y up (towards the
    spin axis, -theta), looking along -r.
    """

    def __init__(self, r, theta, a, beta=0.0, fov=60.0, roll=0.0):
        self.r, self.theta, self.a = float(r), float(theta), float(a)
        D, rho2, Sig2, alpha, omega, varpi = metric_functions(self.r, self.theta, self.a)
        self.D, self.rho2, self.alpha, self.omega, self.varpi = D, rho2, alpha, omega, varpi
        if beta == "orbit":
            # circular-orbit angular velocity, then speed relative to the ZAMO
            Om = 1.0 / (self.r ** 1.5 + self.a)
            beta = varpi * (Om - omega) / alpha
        self.beta = float(beta)
        self.fov = float(fov)
        self.roll = math.radians(roll)

    def directions(self, nx, ny, fov=None, centre=(0.0, 0.0)):
        """Unit vectors N (camera to sky) in the camera frame, as (ny, nx, 3)
        arrays of (along -r, right = +phi, up = -theta) components.
        centre offsets the view (degrees right and up)."""
        fov = self.fov if fov is None else fov
        t = math.tan(math.radians(fov) / 2)
        xs = (np.arange(nx) + 0.5) / nx * 2 - 1
        ys = ((np.arange(ny) + 0.5) / ny * 2 - 1) * ny / nx
        X, Y = np.meshgrid(xs * t, -ys * t)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        X, Y = cr * X - sr * Y, sr * X + cr * Y
        # yaw/pitch offset of the optical axis
        yaw, pitch = map(math.radians, centre)
        f = np.ones_like(X)
        v = np.stack([f, X, Y], -1)
        # rotate about 'up' by yaw, then about 'right' by pitch
        cy, sy = math.cos(yaw), math.sin(yaw)
        v = np.stack([cy * v[..., 0] - sy * v[..., 1], sy * v[..., 0] + cy * v[..., 1], v[..., 2]], -1)
        cp, sp = math.cos(pitch), math.sin(pitch)
        v = np.stack([cp * v[..., 0] - sp * v[..., 2], v[..., 1], sp * v[..., 0] + cp * v[..., 2]], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def panorama(self, nx, ny):
        """Directions for an equirectangular view of the camera's whole sky:
        longitude -180..180 deg across (0 = straight at the hole), latitude
        +90 (along the spin axis) at the top to -90 at the bottom."""
        lam = ((np.arange(nx) + 0.5) / nx * 2 - 1) * math.pi
        lat = (0.5 - (np.arange(ny) + 0.5) / ny) * math.pi
        L, B = np.meshgrid(lam, lat)
        return np.stack([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)], -1)

    @staticmethod
    def right_handed(N):
        """Camera-frame directions (forward, right, up) re-expressed on the
        right-handed triad (e_r, e_theta, e_phi), so that orientations on
        the camera's sky and on the celestial sphere can be compared."""
        return np.stack([-N[..., 0], -N[..., 2], N[..., 1]], -1)

    def initial_conditions(self, N):
        """Map camera-frame directions N (..., 3) to ray initial data.

        Returns Y0 (n, 5), b (n,) and the photon energy measured by the
        camera, E_cam (n,), all for the forward-going photon with E = 1."""
        N = N.reshape(-1, 3)
        # the photon travels towards the camera: n = -N, in (r, theta, phi) components
        n_r = N[:, 0]            # -(-1) * along(-r): forward is -r, so n_r = +N_forward
        n_phi = -N[:, 1]
        n_th = N[:, 2]           # up is -theta, so n_theta = -(-N_up)
        # aberration from the camera frame to the ZAMO frame (boost along phi)
        bt = self.beta
        g = 1.0 / math.sqrt(1.0 - bt * bt)
        den = 1.0 + bt * n_phi
        nF_phi = (n_phi + bt) / den
        nF_r = n_r / (g * den)
        nF_th = n_th / (g * den)
        E_F = 1.0 / (self.alpha + self.omega * self.varpi * nF_phi)
        p_r = E_F * math.sqrt(self.rho2 / self.D) * nF_r
        p_th = E_F * math.sqrt(self.rho2) * nF_th
        b = E_F * self.varpi * nF_phi
        E_cam = g * E_F * (1.0 - bt * nF_phi)
        Y0 = np.empty((len(N), 5))
        Y0[:, 0] = self.r
        Y0[:, 1] = self.theta
        Y0[:, 2] = 0.0
        Y0[:, 3] = p_r
        Y0[:, 4] = p_th
        return Y0, b, E_cam


def sky_direction(Y, B, a):
    """Unit vector (x, y, z) of the asymptotic direction in which each escaped
    backward ray leaves, i.e. the point on the celestial sphere it sees.
    Uses the local orthonormal momentum at the escape radius."""
    r, th, ph, pr, pth = Y.T
    D = r * r - 2 * r + a * a
    rho = np.sqrt(r * r + a * a * np.cos(th) ** 2)
    # backward ray: its direction of travel is minus the photon's momentum
    v_r = -pr * np.sqrt(D) / rho
    v_th = -pth / rho
    v_ph = -B / (r * np.sin(th))
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    x = v_r * st * cp + v_th * ct * cp - v_ph * sp
    y = v_r * st * sp + v_th * ct * sp + v_ph * cp
    z = v_r * ct - v_th * st
    v = np.stack([x, y, z], -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------
# photon orbits and the shadow (Bardeen 1973)
# ----------------------------------------------------------------------------
def photon_orbit_radii(a):
    """Prograde and retrograde equatorial circular photon orbits."""
    rp = 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(-a)))
    rm = 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(a)))
    return rp, rm


def spherical_photon_constants(r, a):
    """b and Q of the spherical photon orbit at radius r (unit energy)."""
    b = -(r ** 3 - 3 * r ** 2 + a * a * r + a * a) / (a * (r - 1.0))
    q = -r ** 3 * (r ** 3 - 6 * r ** 2 + 9 * r - 4 * a * a) / (a * a * (r - 1.0) ** 2)
    return b, q


def shadow_edge(a, theta_obs, n=2000):
    """Bardeen's critical curve for a distant observer at inclination
    theta_obs: celestial coordinates (alpha, beta) in units of M, with
    alpha positive in the direction of the hole's rotation."""
    if a == 0:
        t = np.linspace(0, 2 * np.pi, n)
        R = 3 * math.sqrt(3)
        return R * np.cos(t), R * np.sin(t)
    from scipy.optimize import brentq
    rp, rm = photon_orbit_radii(a)
    ct, st = math.cos(theta_obs), math.sin(theta_obs)

    def beta2(r):
        b, q = spherical_photon_constants(r, a)
        return q + a * a * ct ** 2 - b * b * ct ** 2 / st ** 2

    # the orbits visible from this inclination span [r_lo, r_hi], where beta^2 = 0
    grid = np.linspace(rp, rm, 4001)[1:-1]
    v = beta2(grid)
    pos = np.nonzero(v >= 0)[0]
    r_lo, r_hi = (rp if pos[0] == 0 else grid[pos[0]]), (rm if pos[-1] == len(grid) - 1 else grid[pos[-1]])
    if pos[0] > 0:
        r_lo = brentq(beta2, grid[pos[0] - 1], grid[pos[0]], xtol=1e-14)
    if pos[-1] < len(grid) - 1:
        r_hi = brentq(beta2, grid[pos[-1]], grid[pos[-1] + 1], xtol=1e-14)
    # cosine spacing resolves the ends, where the curve meets the beta = 0 axis
    r = r_lo + (r_hi - r_lo) * 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))

    def xy(r):
        b, _ = spherical_photon_constants(r, a)
        return -b / st, np.sqrt(np.clip(beta2(r), 0, None))

    # refine where the curve turns sharply (the flat edge of a fast spinner)
    for _ in range(30):
        x, y = xy(r)
        long = np.hypot(np.diff(x), np.diff(y)) > 2e-3 * (1 + abs(x).max())
        if not long.any():
            break
        r = np.sort(np.concatenate([r, 0.5 * (r[:-1] + r[1:])[long]]))
    b, q = spherical_photon_constants(r, a)
    # a ray reaching the camera with b > 0 arrives from the -phi side of the image,
    # which is the side where the hole's rotation carries it towards the camera
    alpha = -b / st
    beta = np.sqrt(np.clip(beta2(r), 0, None))
    return np.concatenate([alpha, alpha[::-1]]), np.concatenate([beta, -beta[::-1]])
