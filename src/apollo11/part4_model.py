"""Models for Part 4: lunar ascent, rendezvous, transearth injection and coast, entry.

Units: km, s, kg inside; documented values are converted from feet and nautical miles.
The working frame is the geocentric MOD-1969 frame used throughout the series.
"""
import json
from pathlib import Path

import numpy as np

import anchors as A
import burns as B
import descent as DS
import dynamics as D
import frames as F
import lunar as L
import moon_frame as MF

FT, NMI = F.FT, F.NMI
R_SITE = DS.R_SITE          # landing-site radius: the reference for the Part 4 lunar heights
G0 = 9.80665e-3


def moon_state(t, y):
    return y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)


def lvlh(r, v):
    rh = r / np.linalg.norm(r)
    x = v - (v @ rh) * rh
    x /= np.linalg.norm(x)
    return x, np.cross(-rh, x), -rh


def rot(axis, ang, v):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


def coast(t0, y0, ts, **kw):
    ts = list(ts)
    if not ts:
        return np.zeros((0, 6))
    return D.integrate(D.make_rhs(), t0, y0, ts, **kw)


def lunar_table(t, y, R=R_SITE):
    """Height (n mi above R), speed (ft/s), flight-path angle (deg) relative to the Moon."""
    r, v = moon_state(t, y)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    return (rn - R) / NMI, vn / FT, np.degrees(np.arcsin(r @ v / rn / vn))


def lunar_orbit(t, y, R=R_SITE):
    e = L.elements(*moon_state(t, y), R=R)
    return e["apo"] / NMI, e["peri"] / NMI


# ---------------------------------------------------------------------------
# CSM in lunar orbit: a prior from the Part 3 reconstruction

def csm_prior(part3_json):
    """Geocentric CSM state at the end of the Part 3 track (5-point velocity stencil)."""
    d = json.loads(Path(part3_json).read_text())
    c = np.array(d["csm"])
    k = len(c) - 3
    t = c[k, 0]
    h = c[k + 1, 0] - c[k, 0]
    p = c[k - 2:k + 3, 1:4]
    v = (p[0] - 8 * p[1] + 8 * p[3] - p[4]) / (12 * h)
    return t, np.r_[p[2] + D.moon_pos(t), v + D.moon_vel(t)]


def state_in_plane(t, y_ref, theta_deg, alt_nmi, v_fps, fpa_deg, R=R_SITE):
    """A lunar-orbit state in the plane of y_ref, rotated theta along the orbit from y_ref's
    position, with documented height, speed and flight-path angle."""
    r0, v0 = moon_state(t, y_ref)
    hn = np.cross(r0, v0)
    hn /= np.linalg.norm(hn)
    rh = rot(hn, np.radians(theta_deg), r0 / np.linalg.norm(r0))
    th = np.cross(hn, rh)
    g = np.radians(fpa_deg)
    r = rh * (R + alt_nmi * NMI)
    v = v_fps * FT * (np.cos(g) * th + np.sin(g) * rh)
    return np.r_[r + D.moon_pos(t), v + D.moon_vel(t)]


# ---------------------------------------------------------------------------
# Earth-side helpers

def geodetic_lat(geoc_deg):
    return np.degrees(np.arctan(np.tan(np.radians(geoc_deg)) / (1 - F.FLAT) ** 2))


def geocentric_lat(geod_deg):
    return np.degrees(np.arctan(np.tan(np.radians(geod_deg)) * (1 - F.FLAT) ** 2))


def earth_alt(r):
    lat = np.arcsin(r[2] / np.linalg.norm(r))
    return np.linalg.norm(r) - F.ellipsoid_radius(lat)


def earth_table(t, y):
    tb = F.table_from_state(t, y[:3], y[3:])
    tb["geod_lat"] = geodetic_lat(tb["lat"])
    return tb


def find_alt_crossing(t0, y0, alt_km, t_lo, t_hi, step=20.0):
    """First time after t_lo at which the geodetic height falls through alt_km."""
    y = coast(t0, y0, [t_lo])[-1]
    t = t_lo
    while t < t_hi:
        y1 = coast(t, y, [t + step], h_max=step)[-1]
        if earth_alt(y1[:3]) < alt_km:
            lo, hi, ylo = t, t + step, y
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                ym = coast(lo, ylo, [mid], h_max=step)[-1]
                if earth_alt(ym[:3]) < alt_km:
                    hi = mid
                else:
                    lo, ylo = mid, ym
                if hi - lo < 1e-3:
                    break
            return lo, ylo
        t, y = t + step, y1
    return None, None


# ---------------------------------------------------------------------------
# Transearth: TEI as a finite SPS burn from the CSM's lunar orbit, MCC-5 as flown,
# coast to entry interface.

TEI_DIR = np.array(A.TEI["pad"]) / np.linalg.norm(A.TEI["pad"])


def tei_ignition_state(y_prior, theta):
    a = A.TEI
    return state_in_plane(a["tig"], y_prior, theta, a["alt_ign"], a["v_ign"], a["fpa_ign"])


def fly_tei(y_ign, pitch_deg, yaw_deg, dv_fps, t_out=None):
    a = A.TEI
    d = B.rotate_dir(TEI_DIR, np.radians(pitch_deg), np.radians(yaw_deg))
    return B.burn(a["tig"], y_ign, a["dur"], d, dv_fps * FT, body="moon", t_out=t_out)


def mcc5(y):
    return B.impulsive(A.MCC5["tig"] + A.MCC5["dur"] / 2, y, A.MCC5["pad"], body="earth")


def transearth(p, y_prior, ei_alt_km=None):
    """p = (theta deg, pitch deg, yaw deg, dv ft/s). Returns dict with TEI cutoff state,
    post-MCC-5 state and the entry-interface time and state."""
    theta, pitch, yaw, dv = p
    y_ign = tei_ignition_state(y_prior, theta)
    y_co = fly_tei(y_ign, pitch, yaw, dv)[-1]
    t_m = A.MCC5["tig"] + A.MCC5["dur"] / 2
    y_m0 = coast(A.TEI["tco"], y_co, [t_m])[-1]
    y_m1 = mcc5(y_m0)
    alt = (A.EI["alt_ft"] * FT) if ei_alt_km is None else ei_alt_km
    t_ei, y_ei = find_alt_crossing(t_m, y_m1, alt, A.EI["get"] - 3 * 3600, A.EI["get"] + 3 * 3600, step=60.0)
    return dict(y_ign=y_ign, y_co=y_co, t_m=t_m, y_m0=y_m0, y_m1=y_m1, t_ei=t_ei, y_ei=y_ei)


def ei_residuals(sol):
    e = A.EI
    if sol["t_ei"] is None:
        return np.full(7, 1e3)
    tb = earth_table(sol["t_ei"], sol["y_ei"])
    dlon = (tb["lon"] - e["lon"] + 180) % 360 - 180
    h, v, g = lunar_table(A.TEI["tco"], sol["y_co"])
    return np.array([(sol["t_ei"] - e["get"]) / 2.0,
                     (tb["geod_lat"] - e["geod_lat"]) / 0.01,
                     dlon / 0.01,
                     (tb["fpa"] - e["fpa"]) / 0.01,
                     (h - A.TEI["alt_co"]) / 0.3,
                     (v - A.TEI["v_co"]) / 3.0,
                     (g - A.TEI["fpa_co"]) / 0.1])



# ---------------------------------------------------------------------------
# Lunar ascent: constant APS thrust, 10 s vertical rise, pitchover, then a linear-tangent
# steering law in the plane through the site parallel to the CSM's orbit.

ASC_PITCHOVER_S = 8.0     # time taken to pitch from vertical onto the steering law (assumed)


def ascent_attitude(tau, a, b, yaw_deg=0.0):
    """Thrust elevation above the local horizontal (rad) tau s after liftoff, and yaw (rad)."""
    tv = A.APS_VERTICAL_S
    if tau < tv:
        return np.pi / 2, 0.0
    th_lt = np.arctan(a - b * (tau - tv))
    if tau < tv + ASC_PITCHOVER_S:
        f = (tau - tv) / ASC_PITCHOVER_S
        return np.pi / 2 + (th_lt - np.pi / 2) * f, np.radians(yaw_deg) * f
    return th_lt, np.radians(yaw_deg)


def ascent_thrust(t0, m0, ve, a, b, h_csm, yaw_deg=0.0, F=A.APS_THRUST_N * 1e-3):
    def thrust(t, r, v):
        tau = t - t0
        if tau < 0:
            return np.zeros(3)
        rm = r - D.moon_pos(t)
        rh = rm / np.linalg.norm(rm)
        d = np.cross(h_csm, rh)
        d /= np.linalg.norm(d)
        th, ps = ascent_attitude(tau, a, b, yaw_deg)
        u = np.cos(th) * (np.cos(ps) * d + np.sin(ps) * h_csm) + np.sin(th) * rh
        return u * F / (m0 - F * tau / ve)
    return thrust


def liftoff_state(t):
    rm = DS.site_pos(t)
    w = DS.moon_omega(t)
    return np.r_[rm + D.moon_pos(t), np.cross(w, rm) + D.moon_vel(t)]


def fly_ascent(p, h_csm, t_out=None, yaw_deg=0.0):
    """p = (a, b, m0 kg, ve km/s). Returns states at t_out (plus cutoff last)."""
    a, b, m0, ve = p
    t0 = A.LIFTOFF_GET
    tco = A.INSERTION["get"]
    rhs = D.make_rhs(ascent_thrust(t0, m0, ve, a, b, h_csm, yaw_deg))
    ts = ([] if t_out is None else list(t_out)) + [tco]
    return D.integrate(rhs, t0, liftoff_state(t0), ts, h_max=1.0, rtol=1e-10)


def lunar_hdot_hvel(t, y, R=R_SITE):
    r, v = moon_state(t, y)
    rn = np.linalg.norm(r)
    hd = r @ v / rn
    return (rn - R) / FT, hd / FT, np.sqrt(max(v @ v - hd * hd, 0)) / FT



# ---------------------------------------------------------------------------
# Rendezvous burns, flown with the LM's RCS as finite burns in the LM's LVLH frame.

VE_RCS = 2.84      # km/s, LM RCS (Isp about 290 s, assumed)


def lvlh_burn(t_ign, y, dur, comp_fps, ve=VE_RCS, t_out=None):
    c = np.asarray(comp_fps, float)
    mag = np.linalg.norm(c)
    return B.burn(t_ign, y, dur, c / mag, mag * FT, body="moon", t_out=t_out, ve=ve)


def fit_orbit_burn(t_ign, y, dur, apo, peri, c0, y_comp=0.0):
    """Solve the X and Z components of a burn so the following orbit has the given
    apolune and perilune (n mi above the landing-site radius)."""
    def res(p):
        yy = lvlh_burn(t_ign, y, dur, (p[0], y_comp, p[1]))[-1]
        a, q = lunar_orbit(t_ign + dur, yy)
        return np.array([a - apo, q - peri])
    p, r = L.solve(res, c0, [0.01, 0.01], iters=20)
    return np.array([p[0], y_comp, p[1]]), r


def relative_lvlh(t, y_lm, y_csm):
    """LM position relative to the CSM: curvilinear downrange (km, + ahead of the CSM),
    out of plane (km), radial (km, + above the CSM)."""
    rl, _ = moon_state(t, y_lm)
    rc, vc = moon_state(t, y_csm)
    hn = np.cross(rc, vc)
    hn /= np.linalg.norm(hn)
    rln = np.linalg.norm(rl)
    rcn = np.linalg.norm(rc)
    ip = rl - (rl @ hn) * hn
    ang = np.arctan2(np.cross(rc, ip) @ hn, rc @ ip)
    return np.array([rcn * ang, rl @ hn, rln - rcn])


def fit_intercept(t_ign, y, dur, t_arr, csm_pos, c0):
    """Solve the three LVLH components of a burn that puts the LM on the CSM at t_arr."""
    def res(p):
        yy = lvlh_burn(t_ign, y, dur, p)[-1]
        ya = coast(t_ign + dur, yy, [t_arr])[-1]
        return (ya[:3] - csm_pos) / 0.01
    p, r = L.solve(res, c0, [0.01, 0.01, 0.01], iters=20)
    return p, r



# ---------------------------------------------------------------------------
# Entry: 3-degree-of-freedom flight of the command module through a rotating
# US 1976 standard atmosphere, lift steered by a fitted bank-angle history.

_US76 = [  # base altitude km, base temperature K, lapse K/km, base pressure Pa
    (0.0, 288.15, -6.5, 101325.0), (11.0, 216.65, 0.0, 22632.06), (20.0, 216.65, 1.0, 5474.889),
    (32.0, 228.65, 2.8, 868.0187), (47.0, 270.65, 0.0, 110.9063), (51.0, 270.65, -2.8, 66.93887),
    (71.0, 214.65, -2.0, 3.956420)]
_THERMO = [(86.0, 6.958e-6), (91.0, 2.860e-6), (100.0, 5.604e-7), (110.0, 9.708e-8),
           (120.0, 2.222e-8), (130.0, 8.152e-9), (150.0, 2.076e-9)]
_LOG_T = np.array([[h, np.log(r)] for h, r in _THERMO])


def density(h_km):
    """US Standard Atmosphere 1976 density, kg/m^3 (geometric altitude treated as geopotential
    below 86 km; log-linear in the published table above)."""
    if h_km >= 86.0:
        return float(np.exp(np.interp(h_km, _LOG_T[:, 0], _LOG_T[:, 1])))
    h = 6356.766 * h_km / (6356.766 + h_km)
    for hb, tb, lr, pb in reversed(_US76):
        if h >= hb:
            break
    g0, M, R = 9.80665, 0.0289644, 8.31432
    T = tb + lr * (h - hb)
    if lr == 0.0:
        p = pb * np.exp(-g0 * M * (h - hb) * 1e3 / (R * tb))
    else:
        p = pb * (tb / T) ** (g0 * M / (R * lr * 1e-3))
    return p * M / (R * T)


CM_MASS = A.CM_MASS_LB * 0.45359237
CM_AREA = np.pi * (3.9116 / 2) ** 2       # m^2, 154 in diameter heat shield
CM_CD = 1.29                              # hypersonic drag coefficient at trim (assumed)
BETA = CM_MASS / (CM_CD * CM_AREA)        # kg/m^2
OMEGA = np.array([0.0, 0.0, F.OMEGA_EARTH])


def earth_gravity(r):
    rn = np.linalg.norm(r)
    z2 = (r[2] / rn) ** 2
    k = 1.5 * D.J2 * D.MU_EARTH * D.R_J2**2 / rn**5
    return -D.MU_EARTH * r / rn**3 + k * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])


def aero(r, v, bank_rad, beta=BETA, ld=A.CM_LD):
    """Aerodynamic acceleration (km/s^2) and its magnitude in g."""
    vr = v - np.cross(OMEGA, r)
    h = earth_alt(r)
    rho = density(h)
    vn = np.linalg.norm(vr)
    if rho < 1e-14 or vn < 1e-6:
        return np.zeros(3), 0.0
    dmag = 0.5 * rho * (vn * 1e3) ** 2 / beta * 1e-3          # km/s^2
    vh = vr / vn
    up = r / np.linalg.norm(r)
    upp = up - (up @ vh) * vh
    upp /= np.linalg.norm(upp)
    side = np.cross(vh, upp)
    ldir = np.cos(bank_rad) * upp + np.sin(bank_rad) * side
    a = dmag * (-vh + ld * ldir)
    return a, np.linalg.norm(a) / G0


def fly_entry(t0, y0, bank_fn, t_end, dt=0.5, alt_stop_km=None, beta=BETA, ld=A.CM_LD):
    """RK4 from the entry-interface state. bank_fn(tau) -> bank (rad), tau = t - t0.
    Returns rows (t, y, g) until t_end or until the altitude falls below alt_stop_km."""
    t, y = t0, np.array(y0, float)

    def f(t, y):
        a, g = aero(y[:3], y[3:], bank_fn(t - t0), beta, ld)
        return np.r_[y[3:], earth_gravity(y[:3]) + a], g
    rows = []
    while t < t_end:
        k1, g = f(t, y)
        rows.append((t, y.copy(), g))
        k2, _ = f(t + dt / 2, y + dt / 2 * k1)
        k3, _ = f(t + dt / 2, y + dt / 2 * k2)
        k4, _ = f(t + dt, y + dt * k3)
        yn = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if alt_stop_km is not None and earth_alt(yn[:3]) < alt_stop_km:
            h0, h1 = earth_alt(y[:3]), earth_alt(yn[:3])
            f_ = (h0 - alt_stop_km) / (h0 - h1)
            rows.append((t + f_ * dt, y + f_ * (yn - y), g))
            return rows
        t, y = t + dt, yn
    rows.append((t, y.copy(), f(t, y)[1]))
    return rows


def lat_lon(t, r):
    tb = F.table_from_state(t, r, np.zeros(3))
    return geodetic_lat(tb["lat"]), tb["lon"]


BANK_KNOTS = np.array([0, 40, 70, 95, 120, 160, 200, 250, 300, 360, 420, 480, 560], float)


def bank_profile(p_deg):
    p = np.radians(np.asarray(p_deg, float))
    return lambda tau: float(np.interp(tau, BANK_KNOTS, p))



# ---------------------------------------------------------------------------
# CSM lunar orbit from 124 h to TEI: one state, fitted, carried through the final
# separation burn (130:30, +X in the CSM's LVLH frame).

def csm_states(y_ep, ep, ts):
    """CSM states at the times ts (any order), from the state y_ep at epoch ep (> separation)."""
    fs = A.FINAL_SEP
    ts_sep = fs["tig"] + fs["dur"] / 2
    yb = coast(ep, y_ep, [ts_sep])[-1]
    yb = B.impulsive(ts_sep, yb, (-fs["dv_fps"], 0.0, 0.0), body="moon")
    ts = sorted(set(float(t) for t in ts))
    out = {}
    fwd = [t for t in ts if t >= ep]
    mid = [t for t in ts if ts_sep <= t < ep][::-1]
    early = [t for t in ts if t < ts_sep][::-1]
    for grp, t0, y0 in ((fwd, ep, y_ep), (mid, ep, y_ep), (early, ts_sep, yb)):
        if grp:
            for t, y in zip(grp, coast(t0, y0, grp)):
                out[t] = y
    return out


def bank_fn(q, kn, sign, t_flip, reversals=(376.0, 436.0, 468.0)):
    """Entry bank angle (rad) against time from entry interface: lift up to 70 s, rolling to the
    displayed -15.2 deg by 90 s, then a vertical lift fraction tanh(q) at the knots kn, with the
    lift's sideways sign reversed at t_flip and at the documented lateral switches."""
    u = np.tanh(np.asarray(q, float))

    def f(tau):
        if tau < 70:
            return 0.0
        if tau < 90:
            return np.radians(-15.2) * (tau - 70) / 20
        s = sign if tau < t_flip else -sign
        for ts in reversals:
            if tau > ts:
                s = -s
        return s * np.arccos(np.interp(tau, kn, u))
    return f


def fly_chutes(t0, y0, t_main, t_end, beta_d, beta_m, dt=0.5):
    """Drogues from (t0, y0) to t_main, then mains until the ground. Drag only."""
    n = max(1, int(np.ceil((t_main - t0) / dt)))
    rows = fly_entry(t0, y0, lambda tau: 0.0, t_main, dt=(t_main - t0) / n, beta=beta_d, ld=0.0)
    y1 = rows[-1][1]
    rows2 = fly_entry(t_main, y1, lambda tau: 0.0, t_end, dt=dt, alt_stop_km=0.0, beta=beta_m, ld=0.0)
    return rows[:-1] + rows2
