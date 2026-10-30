"""Build the data file for Part 2: translunar coast and lunar orbit (GET 2:50 to 100:00).

    python3 build_part2.py  [output.json]

Pipeline (each stage starts from a documented anchor):
  1. From the documented TLI state, coast with the documented evasive burn
     (PAD vector, ABTN magnitude) to MCC-2.
  2. Re-anchor at MCC-2 ignition to the ABTN altitude, speed, flight-path angle
     and heading (the table gives no latitude or longitude, so the model's are kept).
  3. MCC-2 as an impulse in the Earth LVLH frame, solved by damped least squares
     so the trajectory arrives with the documented LOI-1 ignition altitude and
     speed, loss of signal time and predicted pericynthion; the PAD vector is
     the prior.
  4. LOI-1 and LOI-2 as finite SPS burns in the lunar LVLH frame, solved jointly
     against the ABTN cutoff states and orbits and the landing-site plane.
  5. Lunar orbit propagated with the Moon's J2 and C22 (no mascons) to 100 h.
Everything is checked against the documented values that were not used in the fit.
"""
import json
import sys
from pathlib import Path
import numpy as np

import anchors as A
import burns as B
import dynamics as D
import frames as F
import lunar as L
import moon_frame as MF
from pao_callouts import PAO

NMI, FT, RM = F.NMI, F.FT, L.RM
SRC = dict(documented=0, interpolated=1, propagated=2, fitted=3)
T_END = 100 * 3600.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- stage 1-2
def coast_to_mcc():
    T = A.TLI
    r, v = F.state_from_table(T["get"], T["lat"], T["lon"], 0, T["v_kms"], T["fpa"], T["head"],
                              radius_km=T["radius_km"])
    y = np.r_[r, v]
    ev = A.EVASIVE
    te = ev["tig"] + ev["dur"] / 2
    y_ev = D.integrate(D.make_rhs(), T["get"], y, [te])[0]
    dv_ev = np.array(ev["pad"]) * ev["dv_fps"] / np.linalg.norm(ev["pad"])
    y_ev2 = B.impulsive(te, y_ev, dv_ev)
    name, tt, alt, vv, fpa, hd = A.COAST_STATES[4]
    ym = D.integrate(D.make_rhs(), te, y_ev2, [tt])[0]
    q = F.table_from_state(tt, ym[:3], ym[3:])
    ra, va = F.state_from_table(tt, q["lat"], q["lon"], alt * NMI, vv * FT, fpa, hd)
    shift = (np.linalg.norm(ra - ym[:3]), np.linalg.norm(va - ym[3:]) * 1e3)
    return dict(y_tli=y, t_tli=T["get"], te=te, y_ev_pre=y_ev, y_ev=y_ev2, dv_ev=dv_ev,
                t_mcc_ign=tt, y_mcc_model=ym, y_mcc_anchor=np.r_[ra, va], shift=shift, q_mcc=q)


# ---------------------------------------------------------------- stage 3
def fit_mcc(c):
    tt = c["t_mcc_ign"]
    tm = A.MCC2["tig"] + A.MCC2["dur"] / 2
    y_m = D.integrate(D.make_rhs(), tt, c["y_mcc_anchor"], [tm])[0]
    tl = A.LOI1["tig"]
    tlos = A.LOS_AOS[0][2]
    p_pad = np.array(A.MCC2["pad"]) * A.MCC2["dv_fps"] / A.MCC2["pad_dv"]

    def arrive(dv):
        y2 = B.impulsive(tm, y_m, dv)
        ts = [tlos - 1500 + 15 * k for k in range(int((tl - tlos + 1500) // 15))] + [tl]
        Y = D.integrate(D.make_rhs(), tm, y2, ts)
        rel, vrel = L.moon_rel(tl, Y[-1])
        cr = [x for x in L.crossings(ts[:-1], Y[:-1]) if x[1] == "LOS"]
        t_los = cr[0][0] if cr else tlos + 1000
        rp, vp = L.moon_rel(ts[0], Y[0])
        peri = L.elements(rp, vp)["peri"] / NMI
        return dict(alt=(np.linalg.norm(rel) - RM) / NMI, v=np.linalg.norm(vrel) / FT, los=t_los,
                    peri=peri, y_ign=Y[-1])

    def res(p):
        a = arrive(p)
        return np.r_[a["alt"] - A.LOI1["alt_ign"], a["v"] - A.LOI1["v_ign"], (a["los"] - tlos) / 10.0,
                     a["peri"] - 61.5, (p - p_pad) / 2.0]

    pad_arrival = arrive(p_pad)
    p, r = L.solve(res, p_pad, [0.02] * 3, iters=15,
                   log=lambda i, p, r: log("  MCC", i, np.round(p, 3), np.round(r[:4], 3)))
    fit = arrive(p)
    none = arrive(np.zeros(3))
    return dict(tm=tm, y_m=y_m, p=p, p_pad=p_pad, pad_arrival=pad_arrival, fit=fit, none=none,
                y_post=B.impulsive(tm, y_m, p))


# ---------------------------------------------------------------- stage 4
def fit_loi(m):
    L1, L2 = A.LOI1, A.LOI2
    y_l1 = m["fit"]["y_ign"]
    pad1 = np.array(L1["pad"]) / np.linalg.norm(L1["pad"])
    pad2 = np.array(L2["pad"]) / np.linalg.norm(L2["pad"])
    lat, lon = A.LM_SITE
    W = np.array([1, 2, 1, 1, 2, 4, 1, 2, 1, 1, 0.5])

    def run(p, full=False):
        p1, y1, s1, p2, y2, s2, pr1 = p
        d1 = B.rotate_dir(pad1, p1, y1)
        yb = B.burn(L1["tig"], y_l1, L1["dur"], d1, L1["dv_fps"] * FT * (1 + s1), pitch_rate=pr1 * 1e-3)[-1]
        t1 = L1["tig"] + L1["dur"]
        yc, yi = D.integrate(D.make_rhs(), t1, yb, [L1["tco"], L2["tig"]])
        rc, vc = L.moon_rel(L1["tco"], yc)
        e1 = L.elements(rc, vc)
        ri, vi = L.moon_rel(L2["tig"], yi)
        d2 = B.rotate_dir(pad2, p2, y2)
        yb2 = B.burn(L2["tig"], yi, L2["dur"], d2, L2["dv_fps"] * FT * (1 + s2))[-1]
        t2 = L2["tig"] + L2["dur"]
        yc2, yland = D.integrate(D.make_rhs(), t2, yb2, [L2["tco"], A.LANDING_GET])
        rc2, vc2 = L.moon_rel(L2["tco"], yc2)
        e2 = L.elements(rc2, vc2)
        raw = np.array([
            (np.linalg.norm(rc) - RM) / NMI - L1["alt_co"], np.linalg.norm(vc) / FT - L1["v_co"],
            e1["apo"] / NMI - L1["apo"], e1["peri"] / NMI - L1["peri"],
            (np.linalg.norm(ri) - RM) / NMI - L2["alt_ign"], np.linalg.norm(vi) / FT - L2["v_ign"],
            (np.linalg.norm(rc2) - RM) / NMI - L2["alt_co"], np.linalg.norm(vc2) / FT - L2["v_co"],
            e2["apo"] / NMI - L2["apo"], e2["peri"] / NMI - L2["peri"],
            L.site_offset_km(A.LANDING_GET, yland, lat, lon) / NMI])
        prior = np.r_[np.degrees(p[0]) / 10, np.degrees(p[[1, 3, 4]]) / 3.0, p[2] / 0.01, p[5] / 0.01, p[6]]
        res = np.r_[raw / W, prior]
        if full:
            return res, dict(raw=raw, e1=e1, e2=e2, d1=d1, d2=d2, yb=yb, yb2=yb2, yi=yi, rc=rc, vc=vc,
                             rc2=rc2, vc2=vc2)
        return res

    p, r = L.solve(run, np.zeros(7), [1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3], iters=14,
                   log=lambda i, p, r: log("  LOI", i, np.round(r[:11] * W, 2)))
    _, info = run(p, True)
    return dict(p=p, info=info, y_l1=y_l1, pad1=pad1, pad2=pad2)


# ---------------------------------------------------------------- helpers
def row(t, y, src):
    rm, vm = L.moon_rel(t, y)
    lat, lon, _ = MF.to_selenographic(t, rm)
    q = F.table_from_state(t, y[:3], y[3:])
    return [round(t, 1), *np.round(y[:3], 2), *np.round(rm, 3), round(q["alt"], 2),
            round(np.linalg.norm(rm) - RM, 3), round(q["v"], 5), round(np.linalg.norm(vm), 5),
            round(lat, 3), round(lon, 3), SRC[src]]


def seg_times(t0, t1, step):
    ts = list(np.arange(t0, t1, step))
    if ts and ts[0] == t0:
        ts = ts[1:]
    return ts + [t1]


def main(out_path):
    log("Stage 1-2: coast to MCC-2")
    c = coast_to_mcc()
    log("Stage 3: MCC-2 fit")
    m = fit_mcc(c)
    log("Stage 4: LOI fit")
    lo = fit_loi(m)
    p = lo["p"]
    L1, L2 = A.LOI1, A.LOI2

    # ---- assemble the reconstructed trajectory, segment by segment
    rows = []
    T0 = c["t_tli"]
    rows.append(row(T0, c["y_tli"], "documented"))

    def coast(t0, y0, t1, step, src="propagated"):
        ts = seg_times(t0, t1, step)
        Y = D.integrate(D.make_rhs(), t0, y0, ts)
        for t, y in zip(ts, Y):
            rows.append(row(t, y, src))
        return Y[-1]

    coast(T0, c["y_tli"], c["te"], 60.0)
    y = coast(c["te"], c["y_ev"], 6 * 3600.0, 60.0)
    y = coast(6 * 3600.0, y, c["t_mcc_ign"], 300.0)
    rows.append(row(c["t_mcc_ign"], c["y_mcc_anchor"], "documented"))
    coast(c["t_mcc_ign"], c["y_mcc_anchor"], m["tm"], 1.0)
    y = coast(m["tm"], m["y_post"], 70 * 3600.0, 300.0, "fitted")
    y = coast(70 * 3600.0, y, L1["tig"], 60.0, "fitted")
    # LOI-1 burn
    d1 = lo["info"]["d1"]
    tb = seg_times(L1["tig"], L1["tig"] + L1["dur"], 10.0)
    Yb = B.burn(L1["tig"], y, L1["dur"], d1, L1["dv_fps"] * FT * (1 + p[2]), t_out=tb[:-1], pitch_rate=p[6] * 1e-3)
    for t, yy in zip(tb, Yb):
        rows.append(row(t, yy, "fitted"))
    y = coast(tb[-1], Yb[-1], L2["tig"], 40.0)
    d2 = lo["info"]["d2"]
    tb2 = seg_times(L2["tig"], L2["tig"] + L2["dur"], 4.0)
    Yb2 = B.burn(L2["tig"], y, L2["dur"], d2, L2["dv_fps"] * FT * (1 + p[5]), t_out=tb2[:-1])
    for t, yy in zip(tb2, Yb2):
        rows.append(row(t, yy, "fitted"))
    y_after2 = Yb2[-1]
    t_after2 = tb2[-1]
    y = coast(t_after2, y_after2, T_END, 40.0)

    # ---- arrays used by the checks
    tr = np.array([r_[0] for r_ in rows])

    def state_at(t):
        """Re-integrate from the nearest earlier row (exact, not interpolated)."""
        i = int(np.searchsorted(tr, t, side="right") - 1)
        r_ = rows[i]
        y0 = np.r_[r_[1:4], np.zeros(3)]
        return None if r_ is None else (i, y0)

    checks = []

    def add(section, label, doc, model, unit="", note=""):
        checks.append([section, label, None if doc is None else float(doc), float(model), unit, note])

    # coast states
    for name, tt, alt, vv, fpa, hd in A.COAST_STATES[:4]:
        yy = D.integrate(D.make_rhs(), c["te"] if tt > c["te"] else T0,
                         c["y_ev"] if tt > c["te"] else c["y_tli"], [tt])[0]
        q = F.table_from_state(tt, yy[:3], yy[3:])
        add("coast", f"{name}: altitude", alt, q["alt"] / NMI, "n mi")
        add("coast", f"{name}: speed", vv, q["v"] / FT, "ft/s")
    q = c["q_mcc"]
    add("coast", "MCC-2 ignition, before re-anchoring: altitude", A.COAST_STATES[4][2], q["alt"] / NMI, "n mi")
    add("coast", "MCC-2 ignition, before re-anchoring: speed", A.COAST_STATES[4][3], q["v"] / FT, "ft/s")
    add("coast", "Re-anchoring shift at MCC-2: position", None, c["shift"][0], "km")
    add("coast", "Re-anchoring shift at MCC-2: velocity", None, c["shift"][1], "m/s")
    yco = D.integrate(D.make_rhs(), m["tm"], m["y_post"], [A.COAST_STATES[5][1]])[0]
    q = F.table_from_state(A.COAST_STATES[5][1], yco[:3], yco[3:])
    add("coast", "MCC-2 cutoff: altitude", A.COAST_STATES[5][2], q["alt"] / NMI, "n mi")
    add("coast", "MCC-2 cutoff: speed", A.COAST_STATES[5][3], q["v"] / FT, "ft/s")
    add("coast", "MCC-2 cutoff: flight-path angle", A.COAST_STATES[5][4], q["fpa"], "deg")

    # MCC
    for k, lab in enumerate(["X (forward)", "Y (cross-track)", "Z (down)"]):
        add("mcc", f"MCC-2 Noun 81 {lab}", m["p_pad"][k], m["p"][k], "ft/s",
            "PAD vector scaled to the achieved 20.9 ft/s")
    add("mcc", "MCC-2 total delta-v", A.MCC2["dv_fps"], np.linalg.norm(m["p"]), "ft/s")
    for lab, key in [("pad_arrival", "With the PAD burn as flown"), ("none", "With no MCC-2 at all")]:
        a = m[lab]
        add("mcc", f"{key}: height at LOI-1 ignition time", A.LOI1["alt_ign"], a["alt"], "n mi")
    a = m["fit"]
    add("arrival", "Height above the Moon at LOI-1 ignition", A.LOI1["alt_ign"], a["alt"], "n mi")
    add("arrival", "Moon-relative speed at LOI-1 ignition", A.LOI1["v_ign"], a["v"], "ft/s")
    add("arrival", "Loss of signal behind the Moon (GET, s after 75:41:00)", A.LOS_AOS[0][2] - (75 * 3600 + 41 * 60),
        a["los"] - (75 * 3600 + 41 * 60), "s")
    add("arrival", "Pericynthion of the approach hyperbola", 61.5, a["peri"], "n mi")

    # PAO series
    pao = []
    for gmin, dist, vel, ref in PAO:
        t = gmin * 60.0
        if t < T0 or t > L1["tig"]:
            continue
        i = int(np.searchsorted(tr, t))
        r_ = rows[min(i, len(rows) - 1)]
        if abs(r_[0] - t) > 400:
            continue
        # interpolate between neighbouring rows
        a_, b_ = rows[i - 1], rows[i]
        f = (t - a_[0]) / (b_[0] - a_[0])
        lerp = lambda j: a_[j] + f * (b_[j] - a_[j])
        if ref == "earth":
            pao.append([gmin, ref, dist, round(lerp(7) / NMI, 1), vel, None if vel is None else round(lerp(9) / FT, 1)])
        else:
            pao.append([gmin, ref, dist, round(lerp(8) / NMI, 1), vel, None if vel is None else round(lerp(10) / FT, 1)])

    # SOI
    ts_soi = A.SOI["get"]
    i = int(np.searchsorted(tr, ts_soi))
    a_, b_ = rows[i - 1], rows[i]
    f = (ts_soi - a_[0]) / (b_[0] - a_[0])
    lerp = lambda j: a_[j] + f * (b_[j] - a_[j])
    dm = np.linalg.norm(D.moon_pos(ts_soi))
    r_soi = dm * (D.MU_MOON / D.MU_EARTH) ** 0.4
    add("soi", "Height above Earth", A.SOI["earth_dist_nmi"], lerp(7) / NMI, "n mi")
    add("soi", "Height above the Moon", A.SOI["moon_dist_nmi"], lerp(8) / NMI, "n mi")
    add("soi", "Speed relative to Earth", A.SOI["v_earth_fps"], lerp(9) / FT, "ft/s")
    add("soi", "Speed relative to the Moon", A.SOI["v_moon_fps"], lerp(10) / FT, "ft/s")
    add("soi", "Laplace sphere-of-influence radius, a (m/M)^(2/5)", None, r_soi / NMI, "n mi",
        "centre distance; Earth-Moon distance at that moment")
    add("soi", "Moon centre distance at the documented crossing", None, (lerp(8) + RM) / NMI, "n mi")

    # LOI
    info = lo["info"]
    raw = info["raw"]
    add("loi1", "Cutoff height", L1["alt_co"], L1["alt_co"] + raw[0], "n mi")
    add("loi1", "Cutoff speed", L1["v_co"], L1["v_co"] + raw[1], "ft/s")
    add("loi1", "Apolune", L1["apo"], L1["apo"] + raw[2], "n mi")
    add("loi1", "Perilune", L1["peri"], L1["peri"] + raw[3], "n mi")
    add("loi1", "Orbital period", A.LOI1_PERIOD / 60, info["e1"]["period"] / 60, "min")
    add("loi1", "Delta-v", L1["dv_fps"], L1["dv_fps"] * (1 + p[2]), "ft/s")
    ang = np.degrees(np.arccos(np.clip(d1 @ lo["pad1"], -1, 1)))
    add("loi1", "Thrust direction at ignition, angle from PAD vector", None, ang, "deg")
    add("loi1", "Thrust pitch rate during the burn", None, np.degrees(p[6] * 1e-3), "deg/s")
    add("loi2", "Ignition height", L2["alt_ign"], L2["alt_ign"] + raw[4], "n mi")
    add("loi2", "Ignition speed", L2["v_ign"], L2["v_ign"] + raw[5], "ft/s")
    add("loi2", "Cutoff height", L2["alt_co"], L2["alt_co"] + raw[6], "n mi")
    add("loi2", "Cutoff speed", L2["v_co"], L2["v_co"] + raw[7], "ft/s")
    add("loi2", "Apolune", L2["apo"], L2["apo"] + raw[8], "n mi")
    add("loi2", "Perilune", L2["peri"], L2["peri"] + raw[9], "n mi")
    add("loi2", "Delta-v", L2["dv_fps"], L2["dv_fps"] * (1 + p[5]), "ft/s")
    ang2 = np.degrees(np.arccos(np.clip(d2 @ lo["pad2"], -1, 1)))
    add("loi2", "Thrust direction, angle from PAD vector", None, ang2, "deg")
    add("loi2", "Landing site distance from the orbit plane at 102:45:40", 0.0, raw[10], "n mi")

    # LOS/AOS through the lunar phase (Earth's centre, lunar limb at 1738 km)
    lun = [r_ for r_ in rows if r_[0] >= 75 * 3600]
    yl = [np.r_[r_[1:4], 0, 0, 0] for r_ in lun]
    cr = L.crossings([r_[0] for r_ in lun], yl)
    losaos = []
    for name, kind, tdoc in A.LOS_AOS:
        cand = [t for t, k in cr if k == kind]
        tm_ = min(cand, key=lambda t: abs(t - tdoc)) if cand else np.nan
        losaos.append([name, kind, tdoc, round(tm_, 1)])
    # AOS if LOI-1 had not been done
    ts_nb = seg_times(L1["tig"], L1["tig"] + 3600, 10.0)
    Ynb = D.integrate(D.make_rhs(), L1["tig"], lo["y_l1"], ts_nb)
    aos_nb = [t for t, k in L.crossings(ts_nb, Ynb) if k == "AOS"]
    losaos.append(["AOS if LOI-1 had not happened (PAD prediction)", "AOS", A.AOS_NO_BURN,
                   round(aos_nb[0], 1) if aos_nb else None])

    # Orbit evolution after LOI-2: osculating apolune/perilune every 10 min
    evo = []
    ts_e = seg_times(t_after2, 102.8 * 3600, 600.0)
    Ye = D.integrate(D.make_rhs(), t_after2, y_after2, ts_e)
    for t, yy in zip(ts_e, Ye):
        rr, vv = L.moon_rel(t, yy)
        e = L.elements(rr, vv)
        evo.append([round(t), round(e["apo"] / NMI, 2), round(e["peri"] / NMI, 2)])
    # same, but a spherical Moon (no J2/C22)
    D.LUNAR_HARMONICS = False
    Ye0 = D.integrate(D.make_rhs(), t_after2, y_after2, ts_e)
    D.LUNAR_HARMONICS = True
    evo0 = []
    for t, yy in zip(ts_e, Ye0):
        rr, vv = L.moon_rel(t, yy)
        e = L.elements(rr, vv)
        evo0.append([round(t), round(e["apo"] / NMI, 2), round(e["peri"] / NMI, 2)])
    yun = D.integrate(D.make_rhs(), t_after2, y_after2, [A.UNDOCK["get"]])[0]
    ru, vu = L.moon_rel(A.UNDOCK["get"], yun)
    add("evolution", "Height at undocking, 100:12:00", A.UNDOCK["alt"], (np.linalg.norm(ru) - RM) / NMI, "n mi")
    add("evolution", "Speed at undocking, 100:12:00", A.UNDOCK["v"], np.linalg.norm(vu) / FT, "ft/s")
    doc_orbits = [[t, a_, p_, "PAO"] for t, a_, p_ in A.PAO_ORBITS] + [
        [L2["tco"], L2["apo"], L2["peri"], "ABTN"],
        [93 * 3600 + 28 * 60, 64.0, 55.5, "MSC 'Apollo 11 Adventure'"],
        [A.SEP_ORBIT["get"], A.SEP_ORBIT["apo"], A.SEP_ORBIT["peri"], "ABTN, after 2.7 ft/s separation burn"]]

    # free-return paths for display: no MCC-2 at all, and no LOI-1
    t_fr = seg_times(m["tm"], 160 * 3600.0, 900.0)
    Yfr = D.integrate(D.make_rhs(), m["tm"], B.impulsive(m["tm"], m["y_m"], np.zeros(3)), t_fr)
    rr_fr = [np.linalg.norm(yy[:3] - D.moon_pos(t)) for t, yy in zip(t_fr, Yfr)]
    k = int(np.argmin(rr_fr))
    add("freereturn", "No MCC-2: closest approach to the Moon (height)", None, (rr_fr[k] - RM) / NMI, "n mi",
        "coarse, 15 min sampling")
    t_nl = seg_times(L1["tig"], 150 * 3600.0, 600.0)
    Ynl = D.integrate(D.make_rhs(), L1["tig"], lo["y_l1"], t_nl)
    re_nl = [np.linalg.norm(yy[:3]) for yy in Ynl]
    ent = next((i for i, r_ in enumerate(re_nl) if r_ < F.RE_EQ + 121.9), None)
    add("freereturn", "No LOI-1: back at Earth's entry interface (GET h)", None,
        t_nl[ent] / 3600 if ent is not None else np.nan, "h", "10 min sampling")

    # Moon, Sun and libration tables for the page
    moon = [[round(t), *np.round(D.moon_pos(t), 1)] for t in np.arange(2 * 3600, 102 * 3600 + 1, 600.0)]
    sun = []
    for t in np.arange(2 * 3600, 102 * 3600 + 1, 3600.0):
        s = D.sun_pos(t)
        s_m = s - D.moon_pos(t)
        slat, slon, _ = MF.to_selenographic(t, s_m)
        elat, elon, _ = MF.to_selenographic(t, -D.moon_pos(t))
        sun.append([round(t), *np.round(s / np.linalg.norm(s), 6), round(slat, 3), round(slon, 3),
                    round(elat, 3), round(elon, 3)])
    # Moon orientation (body -> working frame) every hour, as a flat 3x3 for the 3D view
    orient = [[round(t), *np.round(MF.body_to_work(t).ravel(), 7)] for t in np.arange(2 * 3600, 102 * 3600 + 1, 3600.0)]

    events = [
        ["Translunar injection", A.TLI["get"]],
        ["CSM separates from S-IVB", A.CSM_SEPARATION_GET],
        ["CSM docks with LM", A.DOCKING_GET],
        ["Evasive burn", A.EVASIVE["tig"]],
        ["Midcourse correction 2", A.MCC2["tig"]],
        ["Enter lunar sphere of influence", A.SOI["get"]],
        ["Loss of signal behind the Moon", A.LOS_AOS[0][2]],
        ["LOI-1 ignition", L1["tig"]],
        ["LOI-1 cutoff", L1["tco"]],
        ["Acquisition of signal, in lunar orbit", A.LOS_AOS[1][2]],
        ["LOI-2 ignition", L2["tig"]],
        ["LOI-2 cutoff", L2["tco"]],
    ] + [[f"TV: {name}", t] for t, dur, name in A.TV if t > A.TLI["get"]]

    out = dict(
        meta=dict(
            title="Apollo 11 Part 2: translunar coast and lunar orbit",
            frame="Geocentric, mean equator and equinox of 16 July 1969, km",
            columns=["get", "x", "y", "z", "mx", "my", "mz", "alt_e_km", "alt_m_km", "v_e_kms", "v_m_kms",
                     "slat", "slon", "src"],
            src_codes=SRC, r_moon_ref=RM, range_zero_utc="1969-07-16T13:32:00Z",
            gmst0_rad=float(F.gmst_rad(0.0)), omega_earth=F.OMEGA_EARTH,
            landing_site=list(A.LM_SITE), t_start=A.TLI["get"], t_end=T_END),
        track=rows,
        moon=moon, sun=sun, moon_orient=orient,
        events=events,
        checks=[[s_, l_, None if d_ is None else round(d_, 4), round(m_, 4), u_, n_] for s_, l_, d_, m_, u_, n_ in checks],
        pao=pao,
        losaos=losaos,
        evolution=dict(model=evo, spherical=evo0, documented=doc_orbits),
        free_return=dict(no_mcc=[[round(t), *np.round(yy[:3], 0)] for t, yy in zip(t_fr, Yfr)],
                         no_loi=[[round(t), *np.round(yy[:3], 0)] for t, yy in zip(t_nl, Ynl)]),
        tv=[[t, dur, name] for t, dur, name in A.TV],
        burns=dict(mcc=dict(get=m["tm"], lvlh=list(np.round(m["p"], 3)), pad=list(np.round(m["p_pad"], 3))),
                   loi1=dict(tig=L1["tig"], dur=L1["dur"], dir=list(np.round(d1, 5)), pitch_rate=float(p[6] * 1e-3)),
                   loi2=dict(tig=L2["tig"], dur=L2["dur"], dir=list(np.round(d2, 5))),
                   evasive=dict(tig=A.EVASIVE["tig"], dur=A.EVASIVE["dur"])),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for s_, l_, d_, m_, u_, n_ in checks:
        log(f"{s_:10s} {l_:62s} doc={'' if d_ is None else round(d_, 2)!s:<10} model={m_:.2f} {u_}")
    for r_ in losaos:
        log("LOS/AOS", r_)
    log("rows", len(rows), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "part2.json")
