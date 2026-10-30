"""Part 3: undocking, descent orbit insertion, powered descent and landing.

Pipeline:
  1. The LM's orbital plane is the CSM's (from the Part 2 lunar-orbit reconstruction, which
     passes 0.03 n mi from the site). The point the guidance was flying to is placed
     1,100 ft uprange and 400 ft left of where Eagle landed (MR, BENNETT).
  2. PDI state: documented time, altitude and speed (ABTN); its position along the orbit,
     its flight-path angle and the illegible braking-phase downrange velocity target are
     solved for, so that (a) throttle recovery comes 386 s after ignition (MR),
     (b) at P64 (102:41:32) the LM is at 7,129 ft (MR) and (c) traced back to DOI cutoff
     the orbit is at the documented 57.8 n mi (ABTN).
  3. P63 braking phase: Klumpp's quadratic guidance with the Luminary 99 targets, a
     constant-thrust engine with the documented throttle schedule.
  4. From P64 to touchdown the path is fitted to the crew's altitude, descent-rate and
     forward-speed callouts (a smoothing fit), ending at the landing site at 102:45:39.9.
     The automatic P64 guidance is also continued on its own, to show where the
     computer would have landed.
  5. Back from PDI through DOI (76.4 ft/s, retrograde) to undocking; the CSM separates
     with its 2.7 ft/s radial burn.
"""
import json
import sys
from pathlib import Path

import numpy as np

import anchors as A
import descent as DS
import dynamics as D
import lunar as L
import moon_frame as MF

FT, NMI, RM = DS.FT, DS.NMI, DS.RM
T_START = A.LM_UNDOCK["get"] - 12 * 60          # 100:00:00
T_END = A.TOUCHDOWN_GET + 1230                # 103:06:10
SRC = dict(documented=0, interpolated=1, propagated=2, fitted=3)


def log(*a):
    print(*a, flush=True)


def rot(axis, ang, v):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


def lvlh(r, v):
    rh = r / np.linalg.norm(r)
    x = v - (v @ rh) * rh
    x /= np.linalg.norm(x)
    return x, np.cross(-rh, x), -rh


def moon_state(t, y):
    return y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)


# ---------------------------------------------------------------------------
# 1. Plane and aim point

def csm_plane(part2_json):
    d = json.loads(Path(part2_json).read_text())
    tr = np.array([r[:7] for r in d["track"]])
    i = np.searchsorted(tr[:, 0], 355000)
    h = np.cross(tr[i, 4:7], tr[i + 1, 4:7] - tr[i, 4:7])
    return h / np.linalg.norm(h)


def aim_point(h):
    t = A.TOUCHDOWN_GET
    rs = DS.site_pos(t)
    rh = rs / np.linalg.norm(rs)
    vh = np.cross(h, rh)
    vh -= (vh @ rh) * rh
    vh /= np.linalg.norm(vh)
    right = np.cross(vh, rh)
    c = rs - A.LONG_FT * FT * vh - A.CROSS_FT * FT * right
    lat, lon, _ = MF.to_selenographic(t, c)
    return lat, lon


def pdi_state(h, aim, theta_deg, gamma_deg):
    t = A.PDI["tig"]
    c = MF.from_selenographic(t, aim[0], aim[1], 1.0)
    c = c - (c @ h) * h
    c /= np.linalg.norm(c)
    r = rot(h, -np.radians(theta_deg), c) * (RM + A.PDI["alt"] * NMI)
    rh = r / np.linalg.norm(r)
    th = np.cross(h, rh)
    g = np.radians(gamma_deg)
    v = A.PDI["v"] * FT * (np.cos(g) * th + np.sin(g) * rh)
    return np.r_[r + D.moon_pos(t), v + D.moon_vel(t)]


def targets(vbz_ft):
    p63 = dict(r=DS.ft_vec(A.P63_TARGETS["r"]), v=DS.ft_vec([A.P63_TARGETS["vx"], 0, vbz_ft]),
               a=DS.ft_vec(A.P63_TARGETS["a"]), jz=A.P63_TARGETS["jz"] * FT)
    p64 = dict(r=DS.ft_vec(A.P64_TARGETS["r"]), v=DS.ft_vec(A.P64_TARGETS["v"]),
               a=DS.ft_vec(A.P64_TARGETS["a"]), jz=A.P64_TARGETS["jz"] * FT)
    return p63, p64


MASS_PDI = A.LM_MASS_DOI_KG * np.exp(-A.DOI["dv_fps"] * FT / A.DPS_VE) - 20.0   # DOI propellant, ~20 kg RCS


def simulate(h, aim, p, t_end, extend_ft=0.0):
    """extend_ft: how far the aim point is moved downrange, ramped in from attitude hold (102:41:53)
    to P66 - Armstrong's extension of the automatic approach."""
    p63, p64 = targets(p[2] * 100)
    t1, t2 = A.ATT_HOLD_GET, A.P66["get"]
    shift = (lambda t: extend_ft * min(1.0, max(0.0, (t - t1) / (t2 - t1)))) if extend_ft else None
    sim = DS.Descent(A.PDI["tig"], pdi_state(h, aim, p[0], p[1]), MASS_PDI, aim, p63, p64, A.P64["get"], t_end,
                     aim_shift=shift)
    sim.run()
    return sim


def recovery_time(sim):
    rs = sim.rows
    for a, b in zip(rs, rs[1:]):
        if b["t"] - A.PDI["tig"] > 30 and a["frac_dem"] > A.THROTTLE_RECOVERY >= b["frac_dem"]:
            f = (a["frac_dem"] - A.THROTTLE_RECOVERY) / (a["frac_dem"] - b["frac_dem"])
            return a["t"] + (b["t"] - a["t"]) * f - A.PDI["tig"]
    return 600.0


def alt_hdot(t, y):
    rm, vm = moon_state(t, y)
    rn = np.linalg.norm(rm)
    return (rn - DS.R_SITE) / FT, (vm @ rm) / rn / FT


def doi_cutoff(h, aim, p):
    t = A.DOI["tco"]
    y = D.integrate(D.make_rhs(), A.PDI["tig"], pdi_state(h, aim, p[0], p[1]), [t])[-1]
    rm, vm = moon_state(t, y)
    return (np.linalg.norm(rm) - RM) / NMI, np.linalg.norm(vm) / FT


def fit_pdi(h, aim):
    def res(p):
        sim = simulate(h, aim, p, A.P64["get"])
        alt, hd = alt_hdot(sim.final["t"], sim.final["y"])
        co = doi_cutoff(h, aim, p)
        return np.array([(recovery_time(sim) - A.THROTTLE_DOWN_S) / 2, (alt - A.P64["alt_ft"]) / 100,
                         (hd - A.P64["hdot_fps"]) / 5, (co[0] - A.DOI["alt_co"]) / 0.1])
    p, r = L.solve(res, [16.19, 0.24, 1.23], [0.01, 0.01, 0.05], iters=15,
                   log=lambda it, p, r: log("  fit", it, np.round(p, 4), np.round(r, 3)))
    return p


# ---------------------------------------------------------------------------
# 4. Final approach fitted to the callouts

def whittaker(tgrid, obs_val, obs_der, lam, hard=()):
    """Smooth f on tgrid minimising sum w (f - v)^2 + sum w (f' - d)^2 + lam * sum (third difference)^2.
    obs_val/obs_der: lists of (t, value, weight). hard: (index, value) equality-ish constraints."""
    n, dt = len(tgrid), tgrid[1] - tgrid[0]
    rows, rhs = [], []

    def interp_row(t):
        k = min(n - 2, max(0, int((t - tgrid[0]) / dt)))
        f = (t - tgrid[k]) / dt
        r = np.zeros(n)
        r[k], r[k + 1] = 1 - f, f
        return r

    for t, v, w in obs_val:
        rows.append(np.sqrt(w) * interp_row(t))
        rhs.append(np.sqrt(w) * v)
    for t, d, w in obs_der:
        k = min(n - 2, max(0, int(round((t - tgrid[0]) / dt))))
        r = np.zeros(n)
        k = min(k, n - 2)
        r[k], r[k + 1] = -1 / dt, 1 / dt
        rows.append(np.sqrt(w) * r)
        rhs.append(np.sqrt(w) * d)
    for k, v in hard:
        r = np.zeros(n)
        r[k] = 1
        rows.append(1e4 * r)
        rhs.append(1e4 * v)
    D3 = np.zeros((n - 3, n))
    for i in range(n - 3):
        D3[i, i:i + 4] = [-1, 3, -3, 1]
    rows = np.vstack(rows + list(np.sqrt(lam) * D3 / dt**3))
    rhs = np.r_[rhs, np.zeros(n - 3)]
    return np.linalg.lstsq(rows, rhs, rcond=None)[0]


def site_frame(t, approach_dir):
    """Unit vectors at the actual landing site: up, downrange (horizontal, along approach_dir), right."""
    rs = DS.site_pos(t)
    up = rs / np.linalg.norm(rs)
    z = approach_dir - (approach_dir @ up) * up
    z /= np.linalg.norm(z)
    return rs, up, z, np.cross(z, up)


def from_site(t, s_ft, c_ft, h_ft, approach_body):
    """Position (Moon-relative working frame, km) of a point s ft downrange, c ft right, h ft up of the site."""
    M = MF.body_to_work(t)
    rs = DS.site_pos(t)
    up = rs / np.linalg.norm(rs)
    z = M @ approach_body
    z = z - (z @ up) * up
    z /= np.linalg.norm(z)
    right = np.cross(z, up)
    a_s, a_c = s_ft * FT / DS.R_SITE, c_ft * FT / DS.R_SITE
    d = up * np.cos(a_s) + z * np.sin(a_s)
    d = d * np.cos(a_c) + right * np.sin(a_c)
    return d / np.linalg.norm(d) * (DS.R_SITE + h_ft * FT)


def to_site(t, rm, approach_body):
    M = MF.body_to_work(t)
    rs = DS.site_pos(t)
    up = rs / np.linalg.norm(rs)
    z = M @ approach_body
    z = z - (z @ up) * up
    z /= np.linalg.norm(z)
    right = np.cross(z, up)
    u = rm / np.linalg.norm(rm)
    s = np.arctan2(u @ z, u @ up) * DS.R_SITE / FT
    c = np.arcsin(np.clip(u @ right, -1, 1)) * DS.R_SITE / FT
    return s, c, (np.linalg.norm(rm) - DS.R_SITE) / FT


def pchip(x, y, xq):
    """Monotone cubic (Fritsch-Carlson) interpolation."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    k = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[k]) / h[k]
    h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
    return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]


def final_approach(sim_auto, approach_body):
    """P66: from the automatic P64 state at 102:43:22, a rate-command controller of the kind P66 used
    (Klumpp: 1.5 s time constant, thrust within 20 degrees of vertical) follows the crew's callouts
    of altitude, descent rate and forward speed, and ends at the landing site at contact light.
    Returns a 0.5 s grid of (t, s ft downrange of the site, c ft right, h ft up)."""
    t66, tTD = A.P66["get"], A.TOUCHDOWN_GET
    t_stop = A.PDI["tco"]
    rows = sorted(sim_auto.rows, key=lambda r: r["t"])
    near = [r for r in rows if abs(r["t"] - t66) <= 2.01]
    pts = np.array([to_site(r["t"], r["rm"], approach_body) for r in near])
    ts = np.array([r["t"] for r in near])
    s0, c0, h0 = [np.interp(t66, ts, pts[:, i]) for i in range(3)]
    sd0, cd0, hd0 = [np.polyfit(ts - t66, pts[:, i], 1)[0] for i in range(3)]

    # References from the callouts.
    ha = [(t, a) for t, a, _, _ in A.CALLOUTS if a is not None and t > t66]
    ht = [t66] + [t for t, _ in ha] + [tTD, t_stop + 1.0]
    hv = [h0] + [a for _, a in ha] + [5.6, 0.0]
    fa = [(t, f) for t, _, _, f in A.CALLOUTS if f is not None and t > t66]
    ft_ = [t66] + [t for t, _ in fa] + [t_stop]
    fv = [sd0] + [f for _, f in fa] + [0.0]
    ra = [(t, r) for t, _, r, _ in A.CALLOUTS if r is not None and t > t66]

    dt = 0.05
    tt = np.arange(t66, t_stop + 1.0 + 1e-9, dt)
    # Forward speed: the calls, plus an excess that starts at the automatic P64 speed and decays
    # with a time constant chosen so that the path reaches the site (the calls alone fall short).
    u_call = pchip(ft_[1:], fv[1:], np.clip(tt, ft_[1], None))
    excess0 = sd0 - u_call[0]
    kTD = np.searchsorted(tt, tTD)

    def dist_for(tau):
        return np.sum((u_call + excess0 * np.exp(-(tt - t66) / tau))[:kTD]) * dt
    lo, hi = 0.5, 600.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dist_for(mid) < -s0:
            lo = mid
        else:
            hi = mid
    tau_x = 0.5 * (lo + hi)
    u_ref = u_call + excess0 * np.exp(-(tt - t66) / tau_x)
    scale = tau_x
    s_ref = s0 + np.cumsum(u_ref) * dt
    # Altitude: the calls (extrapolated back to the handover), plus the difference from the automatic
    # path's altitude at the handover, fading with a 15 s time constant.
    htc, hvc = ht[1:], hv[1:]
    hc = pchip(htc, hvc, np.clip(tt, htc[0], None))
    slope0 = (hvc[1] - hvc[0]) / (htc[1] - htc[0])
    hc = np.where(tt < htc[0], hvc[0] + slope0 * (tt - htc[0]), hc)
    h_ref = hc + (h0 - hc[0]) * np.exp(-(tt - t66) / 15.0)
    hd_ref = np.gradient(h_ref, dt)
    c_ref = c0 * (0.5 + 0.5 * np.cos(np.pi * np.clip((tt - t66) / (tTD - t66), 0, 1)))

    g = D.MU_MOON / DS.R_SITE**2 / FT        # ft/s^2
    s, c, h, u, w, v = s0, c0, h0, sd0, cd0, hd0
    out = []
    tau_v, tau_h, tau_p = 1.5, 3.0, 6.0
    tilt_max = np.radians(20.0)
    for k, t in enumerate(tt):
        if k % 10 == 0:
            out.append((t, s, c, h))
        if t >= t_stop:                      # engine off: settle on the pads
            h = max(0.0, h + v * dt)
            v = v - g * dt if h > 0 else 0.0
            u = w = 0.0
            continue
        v_cmd = hd_ref[k] + (h_ref[k] - h) / tau_p
        av = (v_cmd - v) / tau_v
        u_cmd = u_ref[k] + (s_ref[k] - s) / tau_p
        w_cmd = (c_ref[k] - c) / tau_p
        a_s, a_c = (u_cmd - u) / tau_h, (w_cmd - w) / tau_h
        lim = np.tan(tilt_max) * (g + av)
        hh = np.hypot(a_s, a_c)
        if hh > lim > 0:
            a_s, a_c = a_s * lim / hh, a_c * lim / hh
        # The engine's permitted range (11-65% of rated thrust) bounds the total acceleration.
        a_max = A.DPS_THROTTLE_MAX * A.DPS_RATED_N / 7700.0 / 0.3048
        a_min = A.DPS_THROTTLE_MIN * A.DPS_RATED_N / 7700.0 / 0.3048
        tot = np.sqrt(a_s**2 + a_c**2 + (g + av)**2)
        if tot > a_max:
            k_ = a_max / tot
            a_s, a_c, av = a_s * k_, a_c * k_, (g + av) * k_ - g
        elif tot < a_min:
            av = np.sqrt(max(a_min**2 - a_s**2 - a_c**2, 0)) - g
        u, w, v = u + a_s * dt, w + a_c * dt, v + av * dt
        s, c, h = s + u * dt, c + w * dt, max(0.0, h + v * dt)
    out = np.array(out)
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3], scale


# ---------------------------------------------------------------------------

def main(out_path, part2_json):
    checks = []

    def add(sec, label, doc, model, unit, note=""):
        checks.append([sec, label, None if doc is None else float(doc), float(model), unit, note])

    h = csm_plane(part2_json)
    aim = aim_point(h)
    log("aim point", aim, "site", A.LM_SITE)
    log("Fitting PDI position, flight-path angle and braking target")
    p = fit_pdi(h, aim)
    log("p =", p)

    # Full automatic run (P63 then P64 to its target, then P65-like vertical descent).
    sim0 = simulate(h, aim, p, A.TOUCHDOWN_GET + 60)
    t_rec = recovery_time(sim0)
    rows_auto = sim0.rows
    # Find where P64 reaches its target point (T close to zero).
    auto_end = next((r for r in rows_auto if r["phase"] == 2 and r["T"] > -8), rows_auto[-1])

    # Approach direction, fixed in the Moon's body frame (from the LM's velocity at P64 entry).
    r64 = next(r for r in rows_auto if r["t"] >= A.P64["get"] - 1e-6)
    approach_body = MF.body_to_work(r64["t"]).T @ (r64["vrel"] / np.linalg.norm(r64["vrel"]))

    # Armstrong's range extension: move the aim point downrange during P64 until the callouts'
    # forward speeds, integrated through P66, carry the LM exactly to the landing site.
    extend_ft = 0.0
    sim = simulate(h, aim, p, A.P66["get"] + 4, extend_ft=extend_ft)
    grid, ss, cc, hh, u_scale = final_approach(sim, approach_body)
    rows_ext = sim.rows

    # --- Assemble the LM track -------------------------------------------------
    # Coast states before PDI: back-propagate from PDI to DOI, undo DOI, back to the start.
    y_pdi = pdi_state(h, aim, p[0], p[1])
    t_doi_mid = A.DOI["tig"] + A.DOI["dur"] / 2
    t_back = list(np.arange(A.PDI["tig"] - 30, t_doi_mid, -30.0)) + [t_doi_mid]
    Yb = D.integrate(D.make_rhs(), A.PDI["tig"], y_pdi, t_back)
    y_mid = Yb[-1]
    rm, vm = moon_state(t_doi_mid, y_mid)
    x, _, _ = lvlh(rm, vm)
    y_pre = np.r_[y_mid[:3], y_mid[3:] + x * A.DOI["dv_fps"] * FT]
    t_back2 = list(np.arange(t_doi_mid - 30, T_START, -30.0)) + [T_START]
    Yb2 = D.integrate(D.make_rhs(), t_doi_mid, y_pre, t_back2)
    lm_orbit = sorted([(t, y) for t, y in zip(t_back, Yb)] + [(t, y) for t, y in zip(t_back2, Yb2)] +
                      [(A.PDI["tig"], y_pdi)], key=lambda q: q[0])

    # CSM: same state as the LM until the separation burn (radially down, 2.7 ft/s), then its own path.
    y_undock = D.integrate(D.make_rhs(), t_doi_mid, y_pre, [A.LM_UNDOCK["get"]])[-1]
    t_sep = A.CSM_SEP["tig"] + A.CSM_SEP["dur"] / 2
    y_sep0 = D.integrate(D.make_rhs(), A.LM_UNDOCK["get"], y_undock, [t_sep])[-1]
    rm, vm = moon_state(t_sep, y_sep0)
    _, _, zdn = lvlh(rm, vm)
    y_sep = np.r_[y_sep0[:3], y_sep0[3:] + zdn * A.CSM_SEP["dv_fps"] * FT]
    t_csm_a = list(np.arange(T_START, t_sep, 30.0))
    t_csm_b = list(np.arange(np.ceil(t_sep / 30) * 30, T_END + 1, 30.0))
    Ya = D.integrate(D.make_rhs(), A.LM_UNDOCK["get"], y_undock, [t for t in t_csm_a if t >= A.LM_UNDOCK["get"]])
    Ya0 = D.integrate(D.make_rhs(), A.LM_UNDOCK["get"], y_undock, [t for t in t_csm_a if t < A.LM_UNDOCK["get"]][::-1])[::-1]
    Yc = D.integrate(D.make_rhs(), t_sep, y_sep, t_csm_b)
    csm = list(zip([t for t in t_csm_a if t < A.LM_UNDOCK["get"]], Ya0)) + \
        list(zip([t for t in t_csm_a if t >= A.LM_UNDOCK["get"]], Ya)) + [(t_sep, y_sep)] + list(zip(t_csm_b, Yc))

    # Powered descent rows from the simulation (P63), then the fitted final approach.
    track = []
    thrust_dirs = []

    def lm_row(t, rm, vrel, frac, mass, src, phase):
        lat, lon, _ = MF.to_selenographic(t, rm)
        s, c, alt = to_site(t, rm, approach_body)
        rn = np.linalg.norm(rm)
        hdot = (vrel @ rm) / rn / FT
        hvel = np.linalg.norm(vrel - (vrel @ rm) / rn**2 * rm) / FT
        return [round(t, 2), *np.round(rm, 4), round(alt, 1), round(hdot, 2), round(hvel, 2),
                round(-s, 1), round(c, 1), round(lat, 5), round(lon, 5), round(frac * 100, 1),
                round(mass, 1), src, phase]

    for t, y in lm_orbit:
        if t >= A.PDI["tig"]:
            break
        rm, vm = moon_state(t, y)
        w = DS.moon_omega(t)
        src = SRC["propagated"]
        track.append(lm_row(t, rm, vm - np.cross(w, rm), 0.0, MASS_PDI if t > t_doi_mid else
                            A.LM_MASS_DOI_KG, src, 0))
    for r in rows_ext:
        if r["t"] >= A.P66["get"] - 1e-6:
            break
        track.append(lm_row(r["t"], r["rm"], r["vrel"], r["frac"], r["m"], SRC["fitted"],
                            63 if r["t"] < A.P64["get"] - 1e-6 else 64))
        thrust_dirs.append([round(r["t"], 2), *np.round(r["u"], 5)])

    # Kinematics of the fitted approach -> thrust and mass.
    pos = np.array([from_site(t, s, c, hh_, approach_body) for t, s, c, hh_ in zip(grid, ss, cc, hh)])
    dtg = grid[1] - grid[0]
    vel = np.gradient(pos, dtg, axis=0)
    acc = np.gradient(vel, dtg, axis=0)
    r66 = next(r for r in rows_ext if r["t"] >= A.P66["get"] - 1e-6)
    m = r66["m"] - r66["frac"] * A.DPS_RATED_N * 1e-3 * (A.P66["get"] - r66["t"]) / A.DPS_VE
    m64 = next(r for r in rows_ext if r["t"] >= A.P64["get"] - 1e-6)["m"]
    thrust_acc, masses, fracs, tdirs = [], [], [], []
    for t, rr, aa in zip(grid, pos, acc):
        g = -D.MU_MOON * rr / np.linalg.norm(rr) ** 3 + D.lunar_harmonics(t, rr)
        ta = aa - g
        thrust_acc.append(np.linalg.norm(ta))
        tdirs.append(ta / np.linalg.norm(ta))
        F = np.linalg.norm(ta) * m
        masses.append(m)
        fracs.append(F / (A.DPS_RATED_N * 1e-3))
        if t < A.PDI["tco"]:
            m -= F * dtg / A.DPS_VE
    for k in range(0, len(grid), 2):
        t = grid[k]
        if t > A.PDI["tco"] + 1:
            break
        w = DS.moon_omega(t)
        vrel = vel[k] - np.cross(w, pos[k])
        track.append(lm_row(t, pos[k], vrel, fracs[k] if t < A.PDI["tco"] else 0.0, masses[k], SRC["fitted"], 66))
        thrust_dirs.append([round(t, 2), *np.round(tdirs[k], 5)])
    # On the surface to the end.
    for t in np.arange(np.ceil(grid[-1]) + 1, T_END + 1, 30.0):
        rm = from_site(t, ss[-1], cc[-1], 0, approach_body)
        track.append(lm_row(t, rm, np.zeros(3), 0.0, m, SRC["documented"], 68))

    # --- Checks --------------------------------------------------------------------
    def nm_v(t, y):
        rm, vm = moon_state(t, y)
        return (np.linalg.norm(rm) - RM) / NMI, np.linalg.norm(vm) / FT

    lm_at = lambda t: D.integrate(D.make_rhs(), t_doi_mid, y_pre, [t])[-1] if t < t_doi_mid else \
        D.integrate(D.make_rhs(), t_doi_mid, y_mid, [t])[-1]
    a_, v_ = nm_v(A.LM_UNDOCK["get"], D.integrate(D.make_rhs(), t_doi_mid, y_pre, [A.LM_UNDOCK["get"]])[-1])
    add("orbit", "Undocking, 100:12:00: height", A.LM_UNDOCK["alt"], a_, "n mi")
    add("orbit", "Undocking, 100:12:00: speed", A.LM_UNDOCK["v"], v_, "ft/s")
    a_, v_ = nm_v(A.CSM_SEP["tig"], D.integrate(D.make_rhs(), t_doi_mid, y_pre, [A.CSM_SEP["tig"]])[-1])
    add("orbit", "Separation burn ignition: height", A.CSM_SEP["alt_ign"], a_, "n mi")
    add("orbit", "Separation burn ignition: speed", A.CSM_SEP["v_ign"], v_, "ft/s")
    yco = D.integrate(D.make_rhs(), t_sep, y_sep, [A.CSM_SEP["tco"] + 600])[-1]
    e = L.elements(*moon_state(A.CSM_SEP["tco"] + 600, yco))
    add("orbit", "CSM orbit after separation: apolune", A.CSM_SEP["apo"], e["apo"] / NMI, "n mi")
    add("orbit", "CSM orbit after separation: perilune", A.CSM_SEP["peri"], e["peri"] / NMI, "n mi")
    a_, v_ = nm_v(A.DOI["tig"], D.integrate(D.make_rhs(), t_doi_mid, y_pre, [A.DOI["tig"]])[-1])
    add("doi", "DOI ignition: height", A.DOI["alt_ign"], a_, "n mi")
    add("doi", "DOI ignition: speed", A.DOI["v_ign"], v_, "ft/s")
    yco = D.integrate(D.make_rhs(), t_doi_mid, y_mid, [A.DOI["tco"]])[-1]
    a_, v_ = nm_v(A.DOI["tco"], yco)
    add("doi", "DOI cutoff: height", A.DOI["alt_co"], a_, "n mi", "fitted")
    add("doi", "DOI cutoff: speed", A.DOI["v_co"], v_, "ft/s")
    e = L.elements(*moon_state(A.DOI["tco"], yco))
    dsite = (RM - DS.R_SITE) / NMI
    add("doi", "Descent orbit: apolune; above the landing site", A.DOI["orbit_onboard"][0], e["apo"] / NMI + dsite, "n mi",
        "1969: onboard computer")
    add("doi", "Descent orbit: perilune; above the landing site", A.DOI["orbit_onboard"][1], e["peri"] / NMI + dsite, "n mi",
        "1969: onboard computer; 8.5 predicted")
    rm, vm = moon_state(A.PDI["tig"], y_pdi)
    add("pdi", "PDI: height above the reference sphere", A.PDI["alt"], (np.linalg.norm(rm) - RM) / NMI, "n mi", "documented anchor")
    add("pdi", "PDI: height above the landing site", A.PDI["alt_est_ft"], (np.linalg.norm(rm) - DS.R_SITE) / FT, "ft",
        "1969: crew estimate from landmark sightings")
    add("pdi", "PDI: ground range to the landing site", None,
        np.arccos(np.clip(rm @ DS.site_pos(A.PDI["tig"]) / np.linalg.norm(rm) / DS.R_SITE, -1, 1)) * DS.R_SITE / NMI,
        "n mi", "BENNETT: about 260 n mi")
    add("pdi", "PDI: flight-path angle", None, p[1], "deg", "fitted")
    add("pdi", "Braking-phase downrange velocity target", None, p[2] * 100, "ft/s", "fitted; illegible in the scan")
    add("p63", "Throttle recovery after ignition", A.THROTTLE_DOWN_S, t_rec, "s", "fitted")
    alt_p64, hd_p64 = alt_hdot(A.P64["get"], simulate(h, aim, p, A.P64["get"]).final["y"])
    add("p63", "Altitude at P64, 102:41:32", A.P64["alt_ft"], alt_p64, "ft", "fitted")
    add("p63", "Descent rate at P64", -A.P64["hdot_fps"], -hd_p64, "ft/s")
    for t_p, q_, v_ in A.PAO_DESCENT:
        rr = min(rows_auto, key=lambda q: abs(q["t"] - t_p))
        if q_ == "altitude_ft":
            mv = (np.linalg.norm(rr["rm"]) - DS.R_SITE) / FT
            lab = "Altitude"
        else:
            mv = np.linalg.norm(rr["vrel"]) / FT
            lab = "Speed over the surface"
        add("pao", f"{lab} at {int(t_p // 3600)}:{int(t_p % 3600 // 60):02d}:{int(t_p % 60):02d}", v_, mv,
            "ft" if q_ == "altitude_ft" else "ft/s", "PAO commentary, rounded")
    r64p = next(r for r in rows_auto if r["t"] >= A.P64["get"] - 1e-6)
    r63l = [r for r in rows_auto if r["t"] < A.P64["get"] - 1e-6][-1]
    add("p63", "Time-to-go to the braking-phase target at P64", None, -r63l["T"], "s", "BENNETT: targets about 60 s beyond high gate")
    add("p63", "Braking phase delta-v, PDI to P64", None,
        sum(r["frac"] * A.DPS_RATED_N * 1e-3 / r["m"] * 2.0 for r in rows_auto if r["t"] < A.P64["get"] - 1e-6) / FT, "ft/s")
    add("p63", "Range to the landing site at P64", None, -to_site(r64p["t"], r64p["rm"], approach_body)[0], "ft",
        "BENNETT: about 4.5 n mi (27,000 ft) at high gate")
    add("p63", "Forward speed at P64", None, np.linalg.norm(r64p["vg"][[1, 2]]) / FT, "ft/s", "BENNETT: about 500")

    # Callout residuals against the fitted approach, and against the automatic P64.
    def model_at(t):
        """(altitude ft, vertical rate ft/s, forward speed ft/s, unit thrust, LM position) at time t."""
        if t < grid[0]:
            near = sorted(rows_ext, key=lambda q: abs(q["t"] - t))[:2]
            near.sort(key=lambda q: q["t"])
            a0, a1 = near
            f = (t - a0["t"]) / (a1["t"] - a0["t"])
            rm = a0["rm"] + f * (a1["rm"] - a0["rm"])
            vr = a0["vrel"] + f * (a1["vrel"] - a0["vrel"])
            rn = np.linalg.norm(rm)
            hd = vr @ rm / rn
            fwd = np.linalg.norm(vr - hd * rm / rn)
            return (rn - DS.R_SITE) / FT, hd / FT, fwd / FT, a0["u"], rm
        k = int(round((t - grid[0]) / dtg))
        k1, k0 = min(k + 1, len(grid) - 1), max(k - 1, 0)
        return (hh[k], (hh[k1] - hh[k0]) / (grid[k1] - grid[k0]), (ss[k1] - ss[k0]) / (grid[k1] - grid[k0]),
                tdirs[k], pos[k])

    callout_rows = []
    for t, a, rr, f in A.CALLOUTS:
        ma, mr, mf, _, _ = model_at(t)
        auto = min(rows_auto, key=lambda q: abs(q["t"] - t))
        sa, ca, aa = to_site(auto["t"], auto["rm"], approach_body)
        callout_rows.append([t, a, rr, f, round(ma, 1), round(mr, 2), round(mf, 2), round(aa, 1)])
    res_h = [c[4] - c[1] for c in callout_rows if c[1] is not None]
    res_r = [c[5] - c[2] for c in callout_rows if c[2] is not None]
    res_f = [c[6] - c[3] for c in callout_rows if c[3] is not None]
    add("callouts", "Altitude callouts: rms difference", None, np.sqrt(np.mean(np.square(res_h))), "ft", f"{len(res_h)} calls")
    add("callouts", "Descent-rate callouts: rms difference", None, np.sqrt(np.mean(np.square(res_r))), "ft/s", f"{len(res_r)} calls")
    add("callouts", "Forward-speed callouts: rms difference", None, np.sqrt(np.mean(np.square(res_f))), "ft/s", f"{len(res_f)} calls")
    add("callouts", "Forward speed above the calls: decay time", None, u_scale, "s",
        "starts at the automatic P64 speed; sized so the path ends at the site")

    # Landing point designator angles: the line of sight to the aim point below the LM's body
    # horizontal plane (Klumpp: arcsin of the sight line on the thrust axis), against the calls.
    lpd_rows = []
    for t_c, ang in A.LPD_CALLS:
        _, _, _, u_t, p_t = model_at(t_c)
        aimv = MF.from_selenographic(t_c, aim[0], aim[1], DS.R_SITE)
        los = aimv - p_t
        los /= np.linalg.norm(los)
        lpd = np.degrees(np.arcsin(np.clip(-(los @ u_t), -1, 1)))
        lpd_rows.append([t_c, ang, round(lpd, 1)])
        add("lpd", f"LPD angle at {int(t_c // 3600)}:{int(t_c % 3600 // 60):02d}:{int(t_c % 60):02d}", ang, lpd, "deg")

    # Delta-v and propellant.
    dv_p63 = sum(r["frac"] * A.DPS_RATED_N * 1e-3 / r["m"] * min(2.0, A.P66["get"] - r["t"])
                 for r in rows_ext if r["t"] < A.P66["get"] - 1e-6)
    dv_fit = sum(a * dtg for a, t in zip(thrust_acc, grid) if t < A.PDI["tco"])
    add("total", "Powered descent delta-v", A.PDI["dv_fps"], (dv_p63 + dv_fit) / FT, "ft/s")
    used = MASS_PDI - masses[-1]
    lb = 0.45359237
    usable_kg = A.DPS_USABLE_LB * lb - (A.LM_MASS_DOI_KG - MASS_PDI - 20)   # less DOI propellant
    g_moon = D.MU_MOON / DS.R_SITE**2
    hover_kgps = masses[-1] * g_moon / A.DPS_VE
    rem = usable_kg - used
    add("total", "Descent propellant used, PDI to touchdown", None, used / lb, "lb")
    add("total", "Usable descent propellant left at touchdown", None, rem / lb, "lb", "BENNETT: 40-50 s of hover")
    add("total", "Hover time that propellant represents", None, rem / hover_kgps, "s")
    add("total", "Mass at touchdown", None, masses[-1] / lb, "lb")
    sa, ca, _ = to_site(auto_end["t"], auto_end["rm"], approach_body)
    add("landing", "Automatic P64 reaches its target, GET", None, auto_end["t"] / 3600, "h")

    # --- Tables for the page ----------------------------------------------------------
    moon_tab = [[round(t), *np.round(D.moon_pos(t), 1)] for t in np.arange(T_START - 600, T_END + 601, 600.0)]
    sun = []
    for t in np.arange(T_START - 3600, T_END + 3601, 1800.0):
        s_ = D.sun_pos(t) - D.moon_pos(t)
        slat, slon, _ = MF.to_selenographic(t, s_)
        elat, elon, _ = MF.to_selenographic(t, -D.moon_pos(t))
        sun.append([round(t), *np.round(s_ / np.linalg.norm(s_), 6), round(slat, 4), round(slon, 4), round(elat, 4), round(elon, 4)])
    orient = [[round(t), *np.round(MF.body_to_work(t).ravel(), 8)] for t in np.arange(T_START - 1800, T_END + 1801, 900.0)]
    csm_rows = []
    for t, y in csm:
        rm, vm = moon_state(t, y)
        lat, lon, _ = MF.to_selenographic(t, rm)
        csm_rows.append([round(t, 2), *np.round(rm, 3), round((np.linalg.norm(rm) - RM) / NMI, 3),
                         round(np.linalg.norm(vm) / FT, 1), round(lat, 4), round(lon, 4)])
    auto_rows = []
    for r in rows_auto:
        if r["t"] < A.P64["get"] - 1e-6:
            continue
        sa, ca, aa = to_site(r["t"], r["rm"], approach_body)
        auto_rows.append([round(r["t"], 1), round(-sa, 1), round(ca, 1), round(aa, 1)])
        if r is auto_end:
            break
    events = [
        ["Undocking", A.LM_UNDOCK["get"]], ["CSM separation burn", A.CSM_SEP["tig"]],
        ["DOI ignition", A.DOI["tig"]], ["DOI cutoff", A.DOI["tco"]],
        ["PDI", A.PDI["tig"]], ["Full throttle", A.PDI["tig"] + A.THROTTLE_HOLD_S],
        ["Yaw to windows up", A.YAW[0]], ["Throttle down", A.PDI["tig"] + A.THROTTLE_DOWN_S],
        ["P64, high gate", A.P64["get"]], ["P66, manual", A.P66["get"]],
        ["Contact light", A.TOUCHDOWN_GET], ["Engine stop", A.PDI["tco"]],
    ] + [[f"Program alarm {code}", t] for t, code in A.ALARMS] + [[f"Bingo call: {w}", t] for t, w in A.BINGO_CALLS]

    lm_orbit_tab = []
    for t, y in lm_orbit:
        rm, vm = moon_state(t, y)
        lm_orbit_tab.append([round(t, 1), round((np.linalg.norm(rm) - RM) / NMI, 3)])

    out = dict(
        meta=dict(
            title="Apollo 11 Part 3: descent and landing",
            frame="Moon-centred, mean equator and equinox of 16 July 1969, km",
            columns=["get", "x", "y", "z", "alt_ft", "hdot_fps", "hvel_fps", "range_ft", "cross_ft",
                     "slat", "slon", "throttle_pct", "mass_kg", "src", "prog"],
            csm_columns=["get", "x", "y", "z", "alt_nmi", "v_fps", "slat", "slon"],
            src_codes=SRC, r_moon_ref=RM, r_site=DS.R_SITE, landing_site=list(A.LM_SITE), aim_point=list(aim),
            approach_body=list(np.round(approach_body, 7)), t_start=T_START, t_end=T_END,
            pdi=A.PDI["tig"], touchdown=A.TOUCHDOWN_GET, engine_stop=A.PDI["tco"], doi=[A.DOI["tig"], A.DOI["tco"]],
            sep=[A.CSM_SEP["tig"], A.CSM_SEP["tco"]], p64=A.P64["get"], p66=A.P66["get"],
            yaw=list(A.YAW), throttle_down=A.PDI["tig"] + A.THROTTLE_DOWN_S, rated_n=A.DPS_RATED_N,
            usable_kg=usable_kg, mass_pdi=MASS_PDI),
        track=track, csm=csm_rows, auto=auto_rows, moon=moon_tab, sun=sun, moon_orient=orient,
        events=events, alarms=[[t, c] for t, c in A.ALARMS], bingo=[[t, w] for t, w in A.BINGO_CALLS],
        callouts=callout_rows, thrust=thrust_dirs, lpd=lpd_rows, checks=[[s_, l_, d_ if d_ is None else round(d_, 4), round(m_, 4), u_, n_]
                                        for s_, l_, d_, m_, u_, n_ in checks],
        fit=dict(theta_deg=p[0], gamma_deg=p[1], vbz_fps=p[2] * 100, mass_pdi_kg=MASS_PDI, m64=m64),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for s_, l_, d_, m_, u_, n_ in checks:
        log(f"{s_:8s} {l_:58s} doc={'' if d_ is None else round(d_, 2)!s:<9} model={m_:.2f} {u_}  {n_}")
    for c in callout_rows:
        log("callout", c)
    log("rows", len(track), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "part3.json",
         sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "../../post2/data/part2.json"))
