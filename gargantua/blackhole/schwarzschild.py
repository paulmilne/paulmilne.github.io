"""Light bending in the Schwarzschild metric (G = c = M = 1).

A photon with impact parameter b obeys the orbit equation

    (du/dphi)^2 = 1/b^2 - u^2 + 2 u^3,     u = 1/r,

whose right-hand side is the cubic P(u) = 2 (u - u1)(u - u2)(u - u3). Every
angle a ray sweeps out is an integral of 1/sqrt(P) and so an elliptic
integral. This module evaluates them two ways, by quadrature and in closed
form, and builds the lookup table a non-spinning black hole needs for ray
tracing: the sky direction seen along each viewing angle.
"""
import math

import numpy as np
from scipy import integrate, special

B_CRIT = 3.0 * math.sqrt(3.0)   # critical impact parameter, 3 sqrt(3) M
R_PHOTON = 3.0                  # photon sphere


def turning_point(b):
    """Closest approach r0 for impact parameter b > 3 sqrt(3):
    the largest root of r^3 - b^2 r + 2 b^2 = 0."""
    b = np.asarray(b, dtype=float)
    # trigonometric root of the depressed cubic
    return 2.0 * b / math.sqrt(3.0) * np.cos(np.arccos(-3.0 * math.sqrt(3.0) / b) / 3.0)


def impact_parameter(r0):
    """b(r0) = r0 / sqrt(1 - 2/r0): inverse of turning_point."""
    r0 = np.asarray(r0, dtype=float)
    return r0 / np.sqrt(1.0 - 2.0 / r0)


def cubic_roots(b):
    """Roots u1 < 0 < u2 <= u3 of 2u^3 - u^2 + 1/b^2 for b > 3 sqrt(3)."""
    u2 = 1.0 / turning_point(b)
    # 2u^3 - u^2 + 1/b^2 = 2(u - u2)(u^2 + c1 u + c0)
    c1 = u2 - 0.5
    c0 = u2 * c1
    disc = np.sqrt(c1 * c1 - 4.0 * c0)
    return (-c1 - disc) / 2.0, u2, (-c1 + disc) / 2.0


# ----------------------------------------------------------------------------
# deflection of a ray from infinity, by quadrature and in closed form
# ----------------------------------------------------------------------------
def sweep_quad(b, u_from=0.0):
    """Angle swept from u_from (default infinity) in to the turning point,
    int_{u_from}^{u0} du / sqrt(P(u)), by adaptive quadrature. The
    substitution u = u0 - s^2 removes the square-root endpoint singularity."""
    u0 = 1.0 / float(turning_point(b))
    P = lambda u: 1.0 / b ** 2 - u * u + 2.0 * u ** 3
    s_max = math.sqrt(u0 - u_from)

    def f(s):
        u = u0 - s * s
        Pu = P(u)
        if s < 1e-12:
            # limit 2s / sqrt(P) -> 2 / sqrt(-P'(u0))
            return 2.0 / math.sqrt(2.0 * u0 - 6.0 * u0 * u0)
        return 2.0 * s / math.sqrt(Pu)

    val, _ = integrate.quad(f, 0.0, s_max, epsabs=1e-13, epsrel=1e-13, limit=500)
    return val


def sweep_elliptic(b, u_from=0.0):
    """The same integral in closed form,

        int_y^{u2} du / sqrt(2 (u3-u)(u2-u)(u-u1)) = sqrt(2/(u3-u1)) F(phi, k),
        k^2 = (u2-u1)/(u3-u1),  sin^2 phi = (u3-u1)(u2-y) / ((u2-u1)(u3-y)),

    (Byrd and Friedman 233.00). Vectorised over b."""
    u1, u2, u3 = cubic_roots(b)
    y = np.asarray(u_from, dtype=float)
    m = (u2 - u1) / (u3 - u1)
    sin2 = (u3 - u1) * (u2 - y) / ((u2 - u1) * (u3 - y))
    phi = np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))
    return np.sqrt(2.0 / (u3 - u1)) * special.ellipkinc(phi, m)


def deflection(b):
    """Total deflection alpha(b) = 2 * sweep - pi of a ray from infinity."""
    return 2.0 * sweep_elliptic(b) - math.pi


def deflection_weak(b, order=2):
    """Weak-field series alpha = 4/b + 15 pi / (4 b^2) + 128 / (3 b^3) + ..."""
    b = np.asarray(b, dtype=float)
    out = 4.0 / b
    if order >= 2:
        out = out + 15.0 * math.pi / (4.0 * b ** 2)
    if order >= 3:
        out = out + 128.0 / (3.0 * b ** 3)
    return out


def deflection_strong(b):
    """Strong-field limit (Bozza 2002):
    alpha = -log(b / b_c - 1) + log(216 (7 - 4 sqrt 3)) - pi."""
    b = np.asarray(b, dtype=float)
    return -np.log(b / B_CRIT - 1.0) + math.log(216.0 * (7.0 - 4.0 * math.sqrt(3.0))) - math.pi


# ----------------------------------------------------------------------------
# a camera at finite radius
# ----------------------------------------------------------------------------
def impact_from_angle(delta, r_cam):
    """Impact parameter of a ray seen by a static observer at r_cam at angle
    delta from the outward radial direction: b = r sin(delta)/sqrt(1-2/r)."""
    return r_cam * np.sin(delta) / math.sqrt(1.0 - 2.0 / r_cam)


def shadow_angle(r_cam):
    """Angular radius of the shadow, measured from the direction to the hole,
    for a static observer at r_cam (Synge 1966)."""
    s = B_CRIT / r_cam * math.sqrt(1.0 - 2.0 / r_cam)
    if r_cam >= 3.0:
        return math.asin(s)
    return math.pi - math.asin(s)


def _sweep_outward(b, u_c):
    """int_0^{u_c} du / sqrt(P(u)) for a ray with no turning point beyond
    the camera, by quadrature (P > 0 on [0, u_c))."""
    P = lambda u: 1.0 / b ** 2 - u * u + 2.0 * u ** 3
    if b == 0.0:
        return 0.0
    # substitution near u_c in case P(u_c) is small
    Pc = P(u_c)
    if Pc > 1e-6:
        val, _ = integrate.quad(lambda u: 1.0 / math.sqrt(P(u)), 0.0, u_c,
                                epsabs=1e-13, epsrel=1e-12, limit=500)
        return val
    s_max = math.sqrt(u_c)
    val, _ = integrate.quad(lambda s: 2.0 * s / math.sqrt(max(P(u_c - s * s), 1e-300)),
                            0.0, s_max, epsabs=1e-13, epsrel=1e-12, limit=500)
    return val


def sky_angle(delta, r_cam):
    """Total angle Phi swept by the backward ray seen at angle delta (from
    the outward radial direction) by a static camera at r_cam > 3.

    The sky point it shows lies at angle Phi from the camera's radial
    direction, as seen from the hole. Returns nan if the ray is captured."""
    u_c = 1.0 / r_cam
    b = float(impact_from_angle(delta, r_cam))
    if delta <= 0.5 * math.pi:
        # heading outwards: straight to infinity
        return _sweep_outward(b, u_c)
    # heading inwards: escapes only if b exceeds the critical value
    if b <= B_CRIT:
        return math.nan
    full = float(sweep_elliptic(b))
    to_cam = float(sweep_elliptic(b, u_c))
    # in to the turning point, then all the way out
    return to_cam + full


def sky_angle_table(r_cam, n=4000):
    """Tabulate Phi(delta) on a grid refined towards the shadow edge, where
    Phi diverges logarithmically. Returns (delta, Phi)."""
    dc = shadow_angle(r_cam)
    outer = math.pi - dc            # angle from the outward direction
    # uniform away from the edge, geometric close to it
    d1 = np.linspace(0.0, 0.5 * math.pi, n // 4)
    gap = outer - 0.5 * math.pi
    d2 = 0.5 * math.pi + gap * (1.0 - np.geomspace(1.0, 1e-12, 3 * n // 4))
    delta = np.unique(np.concatenate([d1, d2]))
    Phi = np.array([sky_angle(d, r_cam) for d in delta])
    ok = np.isfinite(Phi)
    return delta[ok], Phi[ok]


def aberrate(N, beta):
    """Camera-frame directions N (forward, right, up) of a camera moving at
    speed beta along +right, re-expressed in the frame of an observer at
    rest (a static or ZAMO observer) at the same point. Light travels along
    n = -N; its velocity transforms by the relativistic addition law."""
    g = 1.0 / math.sqrt(1.0 - beta * beta)
    n = -np.asarray(N)
    den = 1.0 + beta * n[..., 1]
    ns = np.stack([n[..., 0] / (g * den), (n[..., 1] + beta) / den, n[..., 2] / (g * den)], -1)
    return -ns


def lens_directions(N, r_cam, theta_cam=0.5 * math.pi, table=None):
    """Ray-trace a static camera by table lookup.

    N are camera-frame unit vectors (forward = towards the hole, right =
    +phi, up = -theta), as from kerr.Camera.directions. Returns the sky
    direction each pixel sees as Cartesian unit vectors in the hole's frame
    (camera at azimuth phi = 0), and a mask of captured rays.

    By spherical symmetry the whole image needs only Phi(delta): the ray
    seen at angle delta from the outward radial direction leaves for the
    sky at angle Phi from that direction, in the plane containing both."""
    if table is None:
        table = sky_angle_table(r_cam)
    d_tab, P_tab = table
    st, ct = math.sin(theta_cam), math.cos(theta_cam)
    e_r = np.array([st, 0.0, ct])
    e_th = np.array([ct, 0.0, -st])
    e_ph = np.array([0.0, 1.0, 0.0])
    Nw = (-N[..., 0:1] * e_r + N[..., 1:2] * e_ph - N[..., 2:3] * e_th)
    cosd = np.clip(Nw @ e_r, -1, 1)
    delta = np.arccos(cosd)
    perp = Nw - cosd[..., None] * e_r
    nrm = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp = np.where(nrm > 1e-15, perp / np.maximum(nrm, 1e-300), e_ph)
    captured = delta > d_tab[-1]
    Phi = np.interp(delta, d_tab, P_tab)
    v = np.cos(Phi)[..., None] * e_r + np.sin(Phi)[..., None] * perp
    return v, captured


def einstein_ring(r_cam, table=None):
    """Angular radius (from the direction of the hole) of the Einstein ring,
    the image of the point directly behind the hole: Phi(delta) = pi."""
    d_tab, P_tab = table if table is not None else sky_angle_table(r_cam)
    i = np.searchsorted(P_tab, math.pi)
    d = np.interp(math.pi, P_tab[i - 1:i + 1], d_tab[i - 1:i + 1])
    return math.pi - d


def point_images(stars_v, r_cam, theta_cam=0.5 * math.pi, table=None, n_max=2):
    """All images of point stars for a static camera, by inverting Phi.

    A star at angle Psi from the camera's outward radial direction (seen
    from the hole) has images where Phi(delta) = Psi + 2 pi k (same side)
    or Phi(delta) = 2 pi (k + 1) - Psi (opposite side). Magnification is
    the ratio of solid angles, mu = sin(delta) d(delta) / (sin(Psi) dPsi).
    Returns star index, camera-frame direction N, signed magnification
    and image order (1 = primary, 2 = secondary, ...)."""
    d_tab, P_tab = table if table is not None else sky_angle_table(r_cam)
    st, ct = math.sin(theta_cam), math.cos(theta_cam)
    e_r = np.array([st, 0.0, ct])
    e_th = np.array([ct, 0.0, -st])
    e_ph = np.array([0.0, 1.0, 0.0])
    cosP = np.clip(stars_v @ e_r, -1, 1)
    Psi = np.arccos(cosP)
    perp = stars_v - cosP[:, None] * e_r
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-300)
    dPhi = np.gradient(P_tab, d_tab)
    out = []
    for k in range(n_max):
        for side, target in ((+1, Psi + 2 * math.pi * k), (-1, 2 * math.pi * (k + 1) - Psi)):
            ok = target <= P_tab[-1]
            delta = np.interp(target[ok], P_tab, d_tab)
            mu = np.sin(delta) / (np.sin(Psi[ok]) * np.interp(delta, d_tab, dPhi))
            p = side * perp[ok]
            Nw = np.cos(delta)[:, None] * e_r + np.sin(delta)[:, None] * p
            # world -> camera frame (forward, right, up)
            N = np.stack([-(Nw @ e_r), Nw @ e_ph, -(Nw @ e_th)], -1)
            order = 2 * k + (1 if side > 0 else 2)
            out.append((np.nonzero(ok)[0], N, side * mu, np.full(ok.sum(), order)))
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))
