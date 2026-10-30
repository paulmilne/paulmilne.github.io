"""Powered-descent model for Apollo 11: the LM's P63/P64 guidance (after Klumpp, R-695)
flown in a Moon-centred simulation, plus helpers for the landing-site frame.

Units inside this module: km, s, kg; guidance targets are converted from feet.
"""
import numpy as np
import dynamics as D
import moon_frame as MF
import anchors as A

FT = 0.3048e-3
NMI = 1.852
RM = MF.R_MOON_REF
R_SITE = RM + A.SITE_ALT_FT * FT          # radius of the landing site, km
G0 = 9.80665e-3                           # km/s^2


def site_pos(t, lat=None, lon=None):
    """Moon-relative working-frame position of a point fixed on the Moon (default: the landing site)."""
    lat = A.LM_SITE[0] if lat is None else lat
    lon = A.LM_SITE[1] if lon is None else lon
    return MF.from_selenographic(t, lat, lon, R_SITE)


def moon_omega(t, dt=30.0):
    """Angular velocity vector of the Moon's body frame in the working frame (rad/s)."""
    M0, M1 = MF.body_to_work(t - dt), MF.body_to_work(t + dt)
    W = (M1 - M0) / (2 * dt) @ MF.body_to_work(t).T
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def guidance_frame(r_site, r_lm):
    """Rows: X up at the site, Y crossrange, Z downrange (horizontal, from the LM towards the site)."""
    X = r_site / np.linalg.norm(r_site)
    h = r_lm - (r_lm @ X) * X
    Z = -h / np.linalg.norm(h)
    Y = np.cross(Z, X)
    return np.array([X, Y, Z])


def ttg_cubic(jz, az, vtz, vgz, dz):
    """Klumpp eq. 12: J T^3 + 6 A T^2 + (18 VT + 6 VG) T + 24 (RT - RG) = 0, T < 0. Returns T."""
    roots = np.roots([jz, 6 * az, 18 * vtz + 6 * vgz, 24 * dz])
    neg = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real < 0]
    return max(neg) if neg else None


def acg(T, rt, vt, at, rg, vg):
    """Klumpp eq. 7: AC = 12 (RT - RG)/T^2 + 6 (VT + VG)/T + AT (all guidance-frame vectors)."""
    return 12 * (rt - rg) / T**2 + 6 * (vt + vg) / T + at


class Descent:
    """Simulate P63 (and optionally P64) from a PDI state.

    targets: dict with r, v, a (3-vectors, km units) and jz for each phase.
    """

    def __init__(self, t0, y0, mass0, target_ll, p63, p64, t_p64, t_end, ve=A.DPS_VE, dt=2.0, aim_shift=None):
        self.t0, self.y0, self.m0 = t0, np.array(y0, float), mass0
        self.target_ll = target_ll       # (lat, lon) of the point the guidance flies to, at R_SITE
        self.p63, self.p64, self.t_p64, self.t_end = p63, p64, t_p64, t_end
        self.ve, self.dt = ve, dt
        self.aim_shift = aim_shift       # optional f(t) -> feet the aim point is moved downrange

    def run(self):
        t, y, m = self.t0, self.y0.copy(), self.m0
        rated = A.DPS_RATED_N * 1e-3          # kN -> kg km/s^2 when divided by kg: N = kg m/s^2
        rows = []
        recovered = False
        t_recover = None
        while t < self.t_end - 1e-9:
            rm, vm = y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)
            rs = site_pos(t, *self.target_ll)
            C = guidance_frame(rs, rm)
            if self.aim_shift is not None:
                rs = rs + C[2] * self.aim_shift(t) * FT
                C = guidance_frame(rs, rm)
            w = moon_omega(t)
            vrel = vm - np.cross(w, rm)
            rg, vg = C @ (rm - rs), C @ vrel
            # Leadtime: predict the state when the command takes effect.
            phase = self.p63 if t < self.t_p64 else self.p64
            rt, vt, at, jz = phase["r"], phase["v"], phase["a"], phase["jz"]
            rgl = rg + vg * A.LEADTIME
            T = ttg_cubic(jz, at[2], vt[2], vg[2], rt[2] - rgl[2])
            g_vec = -D.MU_MOON * rm / np.linalg.norm(rm) ** 3
            gg = C @ g_vec
            if T is None or T > -1.0:
                T = -1.0
            ac = acg(T, rt, vt, at, rgl, vg)
            af = ac - gg                            # thrust acceleration demanded, guidance frame
            gtc = np.linalg.norm(af) * m            # kg km/s^2 (= kN)
            frac_dem = gtc / rated
            el = t - self.t0
            if el < A.THROTTLE_HOLD_S:
                frac = A.DPS_MIN
            elif not recovered and frac_dem > A.THROTTLE_RECOVERY:
                frac = A.DPS_FTP
            else:
                if not recovered:
                    recovered, t_recover = True, t
                frac = min(A.DPS_THROTTLE_MAX, max(A.DPS_THROTTLE_MIN, frac_dem))
            F = frac * rated
            u = C.T @ (af / np.linalg.norm(af))     # working-frame unit thrust direction
            rows.append(dict(t=t, rm=rm, vm=vm, vrel=vrel, rg=rg, vg=vg, T=T, frac=frac, frac_dem=frac_dem,
                             m=m, u=u, phase=1 if t < self.t_p64 else 2))
            dt = min(self.dt, self.t_end - t)

            def thrust(tt, r, v, F=F, u=u, m=m, t1=t):
                return u * F / (m - F * (tt - t1) / self.ve)
            y = D.integrate(D.make_rhs(thrust), t, y, [t + dt], h_max=dt)[-1]
            m -= F * dt / self.ve
            t += dt
        rm, vm = y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)
        self.final = dict(t=t, y=y, m=m)
        self.rows = rows
        self.t_recover = t_recover
        return rows


def ft_vec(v):
    return np.array(v, float) * FT
