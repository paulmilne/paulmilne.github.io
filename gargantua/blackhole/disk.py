"""A geometrically thin accretion disk in the equatorial plane of a Kerr hole.

Gas moves on prograde circular geodesics (Bardeen, Press and Teukolsky 1972)
down to the innermost stable circular orbit, and radiates the flux of the
Novikov-Thorne / Page-Thorne (1974) model. G = c = M = 1.
"""
import math

import numpy as np
from scipy import integrate


def isco(a):
    """Radius of the innermost stable circular (prograde) orbit."""
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = math.sqrt(3 * a * a + z1 * z1)
    return 3 + z2 - math.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def omega(r, a):
    """Angular velocity d(phi)/dt of a prograde circular orbit."""
    return 1.0 / (r ** 1.5 + a)


def energy(r, a):
    """Specific energy E = -u_t of a prograde circular orbit."""
    return (r ** 1.5 - 2 * r ** 0.5 + a) / (r ** 0.75 * np.sqrt(r ** 1.5 - 3 * r ** 0.5 + 2 * a))


def ang_mom(r, a):
    """Specific angular momentum L = u_phi of a prograde circular orbit."""
    return (r * r - 2 * a * r ** 0.5 + a * a) / (r ** 0.75 * np.sqrt(r ** 1.5 - 3 * r ** 0.5 + 2 * a))


def u_t(r, a):
    """u^t = dt/dtau of a prograde circular orbit."""
    return (r ** 1.5 + a) / (r ** 0.75 * np.sqrt(r ** 1.5 - 3 * r ** 0.5 + 2 * a))


def flux_numeric(r, a, r_in=None):
    """Page-Thorne flux per unit accretion rate,

        F(r) = -(1 / 4 pi sqrt(-g)) Omega_,r / (E - Omega L)^2
                  * int_{r_in}^r (E - Omega L) L_,r dr,

    with sqrt(-g) = r on the equator, by quadrature."""
    r_in = isco(a) if r_in is None else r_in
    h = 1e-6

    def d(f, x):
        return (f(x + h, a) - f(x - h, a)) / (2 * h)

    def integrand(x):
        return (energy(x, a) - omega(x, a) * ang_mom(x, a)) * d(ang_mom, x)

    out = []
    for x in np.atleast_1d(r):
        if x <= r_in:
            out.append(0.0)
            continue
        I, _ = integrate.quad(integrand, r_in, x, epsabs=1e-14, epsrel=1e-12, limit=200)
        EmOL = energy(x, a) - omega(x, a) * ang_mom(x, a)
        out.append(-d(omega, x) / EmOL ** 2 * I / (4 * math.pi * x))
    return np.array(out)


def flux(r, a):
    """Closed-form Page-Thorne flux per unit accretion rate (Page and Thorne
    1974): with x = sqrt(r), x0 = sqrt(r_isco) and x1, x2, x3 the roots of
    x^3 - 3x + 2a = 0,

    F = 3 / (8 pi) / (x^4 (x^3 - 3x + 2a)) * [ x - x0 - (3/2) a ln(x/x0)
          - sum_i 3 (x_i - a)^2 / (x_i prod_{j != i}(x_i - x_j)) ln((x - x_i)/(x0 - x_i)) ],

    zero inside the ISCO. It tends to 3 / (8 pi r^3) far out."""
    r = np.asarray(r, dtype=float)
    x = np.sqrt(r)
    x0 = math.sqrt(isco(a))
    acs = math.acos(a)
    x1 = 2 * math.cos((acs - math.pi) / 3)
    x2 = 2 * math.cos((acs + math.pi) / 3)
    x3 = -2 * math.cos(acs / 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        bracket = (x - x0 - 1.5 * a * np.log(x / x0)
                   - 3 * (x1 - a) ** 2 / (x1 * (x1 - x2) * (x1 - x3)) * np.log((x - x1) / (x0 - x1))
                   - 3 * (x2 - a) ** 2 / (x2 * (x2 - x1) * (x2 - x3)) * np.log((x - x2) / (x0 - x2))
                   - 3 * (x3 - a) ** 2 / (x3 * (x3 - x1) * (x3 - x2)) * np.log((x - x3) / (x0 - x3)))
        F = 3.0 / (8 * math.pi) / (x ** 4 * (x ** 3 - 3 * x + 2 * a)) * bracket
    return np.where(r > x0 * x0, F, 0.0)


def flux_newtonian(r, r_in):
    """Shakura-Sunyaev flux 3 / (8 pi r^3) (1 - sqrt(r_in / r)) per unit
    accretion rate: the large-r limit of the relativistic flux."""
    r = np.asarray(r, dtype=float)
    return np.where(r > r_in, 3 / (8 * math.pi * r ** 3) * (1 - np.sqrt(r_in / r)), 0.0)


def temperature(r, a, t_max=1.0):
    """Effective temperature T proportional to F^(1/4), scaled so that its
    maximum over the disk is t_max."""
    rr = np.geomspace(isco(a) * 1.0001, 50, 4000)
    Fmax = flux(rr, a).max()
    return t_max * (np.clip(flux(r, a), 0, None) / Fmax) ** 0.25


def redshift(r, b, E_cam, a):
    """g = nu_observed / nu_emitted for a photon of axial angular momentum b
    (unit energy at infinity) leaving a circular-orbit emitter at r and
    reaching a camera that measures energy E_cam."""
    return E_cam / (u_t(r, a) * (1.0 - omega(r, a) * b))


def streaks(r, phi, seed=3, n_modes=90, contrast=0.35):
    """A procedural texture of spiral streaks, stretched along the orbit,
    for the look of turbulent gas. Mean 1, values >= 0."""
    rng = np.random.default_rng(seed)
    out = np.zeros(np.shape(r))
    L = np.log(np.maximum(r, 1e-9))
    tot = 0.0
    for _ in range(n_modes):
        m = rng.integers(1, 7)
        k = rng.uniform(4, 60)
        amp = 1.0 / math.sqrt(k)
        out += amp * np.cos(k * L + m * phi + rng.uniform(0, 2 * math.pi))
        tot += amp * amp
    out /= math.sqrt(tot / 2)
    return np.clip(1.0 + contrast * out, 0.0, None)
