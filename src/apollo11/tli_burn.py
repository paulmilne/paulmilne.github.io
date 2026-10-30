"""Reconstruct the S-IVB translunar injection burn.

The 1969 tables give the parking-orbit state, the burn's start and end times
and the state 10 s after cutoff. They do not give the steering. We model the
burn as a constant-thrust rocket (mass falling linearly, so acceleration
rises as a / (1 - a t / ve)) whose thrust direction is pitched and yawed off
the velocity vector by angles that vary linearly in time, and solve for the
five unknowns (initial acceleration, pitch and yaw offsets and rates) that
best reproduce the documented TLI state. The solver is damped Gauss-Newton
with a finite-difference Jacobian, so it needs only numpy.
"""
import numpy as np
import dynamics as D

VE = 4.16   # km/s, J-2 effective exhaust velocity (Isp ~ 424 s)


def thrust_fn(p, t_ign):
    a0, pb, pk, yb, yk = p

    def thrust(t, r, v):
        tau = t - t_ign
        a = a0 / (1.0 - a0 * tau / VE)
        uv = v / np.linalg.norm(v)
        h = np.cross(r, v)
        h /= np.linalg.norm(h)
        n = np.cross(h, uv)
        th, ps = pb + pk * tau, yb + yk * tau
        return a * (np.cos(th) * np.cos(ps) * uv + np.sin(th) * n + np.cos(th) * np.sin(ps) * h)
    return thrust


def fly(p, t_ign, y_ign, t_cut, t_tli, t_out=None):
    """Burn from ignition to cutoff, then coast to the TLI epoch."""
    burn_times = [t_cut] if t_out is None else list(t_out) + [t_cut]
    yb = D.integrate(D.make_rhs(thrust_fn(p, t_ign)), t_ign, y_ign, burn_times, h_max=5.0)
    y_tli = D.integrate(D.make_rhs(), t_cut, yb[-1], [t_tli])[0]
    return yb, y_tli


def fit(t_ign, y_ign, t_cut, t_tli, y_target, p0=None, iters=12):
    p = np.array(p0 if p0 is not None else [0.007, 0.0, 0.0, 0.0, 0.0])
    w = np.r_[np.ones(3), np.ones(3) * 1000.0]      # 1 km ~ 1 m/s

    def resid(q):
        return (fly(q, t_ign, y_ign, t_cut, t_tli)[1] - y_target) * w

    lam = 1e-3
    res = resid(p)
    for _ in range(iters):
        J = np.empty((6, 5))
        for j in range(5):
            dp = np.zeros(5)
            dp[j] = 1e-6 if j == 0 else 1e-5
            J[:, j] = (resid(p + dp) - res) / dp[j]
        step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J)), -J.T @ res)
        trial = resid(p + step)
        if np.linalg.norm(trial) < np.linalg.norm(res):
            p, res, lam = p + step, trial, lam * 0.3
        else:
            lam *= 10
    return p, res / w
