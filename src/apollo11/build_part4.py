"""Part 4: lunar ascent, rendezvous, transearth injection, the coast home and entry.

    python3 build_part4.py out.json part3.json

Pipeline:
  1. Columbia's lunar orbit, 124-135 h. Starting from the Part 3 orbit carried forward, one state
     (position along the orbit, height, speed, flight-path angle) is fitted to the times Mission
     Control gave for loss and acquisition of signal, the ABTN heights and speeds at docking,
     jettison, final separation and TEI, the orbit after the separation burn, and the TPI burn
     the crew read out (the rendezvous pins down where Columbia was along its orbit far better
     than a signal time does).
  2. Ascent: APS at 3,500 lbf from the landing site at the documented liftoff time, 10 s vertical
     rise, then a linear-tangent steering law; two steering constants, the liftoff mass and the
     exhaust velocity are fitted to the insertion state the crew read from the DSKY, the ABTN speed
     and the ABTN delta-v.
  3. Rendezvous: CSI and CDH as finite RCS burns fitted to the documented orbits after each; TPI
     solved so Eagle meets Columbia at the end of TPF; braking and docking follow the range and
     range-rate calls.
  4. TEI: a finite SPS burn from Columbia's fitted orbit; its pitch, yaw and size are fitted so that,
     with MCC-5 as flown, the trajectory crosses 400,000 ft at the documented entry-interface time,
     place and flight-path angle.
  5. Entry: the command module flown through the US 1976 atmosphere with L/D 0.282, its bank angle
     fitted to the load-factor history, the drogue time and the onboard splash point; then
     drogues and mains to splashdown.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import anchors as A
import burns as B
import descent as DS
import dynamics as D
import frames as F
import lunar as L
import moon_frame as MF
import part4_model as P

FT, NMI = P.FT, P.NMI
RM = MF.R_MOON_REF
g = A._get
T_START = g(124, 0, 0)
SPLASH_T = A.SPLASH["get"]
T_END = SPLASH_T + 300
EP = g(131, 0, 0)
# Mission Control's signal times for Columbia (TEC): "about a minute to LOS" and so on.
LOS = [g(125, 6, 57), g(129, 3, 15), g(131, 1, 37), g(133, 0, 2), g(134, 57, 30)]
AOS = [g(127, 51, 0), g(131, 48, 0), g(133, 46, 0)]
SRC = dict(documented=0, interpolated=1, propagated=2, fitted=3)
LB = 0.45359237


def log(*a):
    print(*a, flush=True)


def gs(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def main(out_path, part3_json):
    t_clock = time.time()
    checks = []

    def add(sec, label, doc, model, unit, note=""):
        checks.append([sec, label, None if doc is None else float(doc), float(model), unit, note])

    # ------------------------------------------------------------------ prior
    t0, y0 = P.csm_prior(part3_json)
    y_prior_tei = P.coast(t0, y0, [A.TEI["tig"]])[-1]
    y_ep_prior = P.coast(A.TEI["tig"], y_prior_tei, [EP])[-1]
    yl = P.coast(EP, y_ep_prior, [A.LIFTOFF_GET])[-1]
    r_, v_ = P.moon_state(A.LIFTOFF_GET, yl)
    h_csm = np.cross(r_, v_)
    h_csm /= np.linalg.norm(h_csm)

    # ------------------------------------------------------------------ ascent
    log("Ascent fit")
    def res_asc(p):
        y = P.fly_ascent(p, h_csm)[-1]
        t = A.INSERTION["get"]
        alt, hd, _ = P.lunar_hdot_hvel(t, y)
        _, v, _ = P.lunar_table(t, y)
        m1 = p[2] - A.APS_THRUST_N * 1e-3 * A.INSERTION["dur"] / p[3]
        dv = p[3] * np.log(p[2] / m1) / FT
        return np.array([(alt - A.INSERTION["dsky_alt_ft"]) / 100, hd - A.INSERTION["dsky_hdot"],
                         v - A.INSERTION["v"], dv - A.INSERTION["dv_fps"]])
    p_asc, _ = L.solve(res_asc, [0.75, 0.00195, 4890, 3.03], [1e-3, 1e-5, 1.0, 1e-3], iters=30)
    log("  p_asc", p_asc)
    t_asc = list(np.arange(A.LIFTOFF_GET, A.INSERTION["get"], 1.0))
    Y_asc = P.fly_ascent(p_asc, h_csm, t_asc)
    y_ins = Y_asc[-1]
    m_ins = p_asc[2] - A.APS_THRUST_N * 1e-3 * A.INSERTION["dur"] / p_asc[3]

    # ------------------------------------------------------------------ CSI, CDH
    log("CSI / CDH")
    y_csi0 = P.coast(A.INSERTION["get"], y_ins, [A.CSI["tig"]])[-1]
    c_csi, _ = P.fit_orbit_burn(A.CSI["tig"], y_csi0, A.CSI["dur"], A.CSI["apo"], A.CSI["peri"], [51.5, 0.0])
    y_csi1 = P.lvlh_burn(A.CSI["tig"], y_csi0, A.CSI["dur"], c_csi)[-1]
    y_cdh0 = P.coast(A.CSI["tco"], y_csi1, [A.CDH["tig"]])[-1]
    c_cdh, _ = P.fit_orbit_burn(A.CDH["tig"], y_cdh0, A.CDH["dur"], A.CDH["apo"], A.CDH["peri"],
                                [-8.1, -18.2], y_comp=A.CDH["noun81"][1])
    y_cdh1 = P.lvlh_burn(A.CDH["tig"], y_cdh0, A.CDH["dur"], c_cdh)[-1]
    y_tpi0 = P.coast(A.CDH["tco"], y_cdh1, [A.TPI["tig"]])[-1]

    # ------------------------------------------------------------------ Columbia's orbit
    log("Columbia's orbit fit")
    grid = list(np.arange(g(124, 50, 0), g(135, 23, 30), 30.0))
    pts = [A.DOCK["get"], A.JETTISON["get"], A.FINAL_SEP["tig"], A.TEI["tig"], A.TPF["tco"],
           A.FINAL_SEP["tco"] + 300, A.PAO_TPI["get"]]
    y_pao = P.coast(A.CDH["tco"], y_cdh1, [A.PAO_TPI["get"]])[-1]

    def csm_fit_res(p, verbose=False):
        y_ep = P.state_in_plane(EP, y_ep_prior, *p)
        tr = P.csm_states(y_ep, EP, grid + pts)
        cr = L.crossings(np.array(grid), [tr[t] for t in grid])
        r = []
        for tl in LOS:
            c = min([c for c in cr if c[1] == "LOS"], key=lambda c: abs(c[0] - tl))
            r.append((c[0] - tl) / 30)
        for ta in AOS:
            c = min([c for c in cr if c[1] == "AOS"], key=lambda c: abs(c[0] - ta))
            r.append((c[0] - ta) / 60)
        for t, (a_, v_) in [(A.DOCK["get"], (A.DOCK["alt"], A.DOCK["v"])),
                            (A.JETTISON["get"], (A.JETTISON["alt"], A.JETTISON["v"])),
                            (A.FINAL_SEP["tig"], (A.FINAL_SEP["alt_ign"], A.FINAL_SEP["v_ign"])),
                            (A.TEI["tig"], (A.TEI["alt_ign"], A.TEI["v_ign"]))]:
            hh, vv, _ = P.lunar_table(t, tr[t])
            r += [hh - a_, (vv - v_) / 3.0]
        r.append((P.lunar_table(A.TEI["tig"], tr[A.TEI["tig"]])[2] - A.TEI["fpa_ign"]) / 0.2)
        ap, pe = P.lunar_orbit(A.FINAL_SEP["tco"] + 300, tr[A.FINAL_SEP["tco"] + 300])
        r += [ap - A.FINAL_SEP["apo"], pe - A.FINAL_SEP["peri"]]
        c, _ = P.fit_intercept(A.TPI["tig"], y_tpi0, A.TPI["dur"], A.TPF["tco"], tr[A.TPF["tco"]][:3],
                               list(A.TPI["noun81"]))
        r += list((c - np.array(A.TPI["noun81"])) / 2.0)
        tp = A.PAO_TPI["get"]
        d0 = y_pao[:3] - tr[tp][:3]
        dv_ = y_pao[3:] - tr[tp][3:]
        rng = np.linalg.norm(d0)
        r += [(rng / NMI - A.PAO_TPI["range_nmi"]) / 0.5, ((d0 @ dv_) / rng / FT - A.PAO_TPI["rdot_fps"]) / 10.0]
        return np.array(r), cr, tr, c

    p_csm, r_csm = L.solve(lambda p: csm_fit_res(p)[0], [14.67, 58.53, 5353.07, -0.197],
                           [0.01, 0.01, 0.01, 0.01], iters=10)
    log("  p_csm", p_csm, "cost", r_csm @ r_csm)
    log("  residuals", np.round(r_csm, 2))
    y_ep = P.state_in_plane(EP, y_ep_prior, *p_csm)
    _, crossings, _, c_tpi = csm_fit_res(p_csm)

    # ------------------------------------------------------------------ TPI to docking
    log("TPI, braking, docking")
    y_tpi1 = P.lvlh_burn(A.TPI["tig"], y_tpi0, A.TPI["dur"], c_tpi)[-1]
    t_lm_coast = list(np.arange(np.ceil(A.TPI["tco"]), A.TPF["tco"], 5.0)) + [A.TPF["tco"]]
    Y_tpi = P.coast(A.TPI["tco"], y_tpi1, t_lm_coast)

    # Columbia: one table of states on a 30 s grid plus the burn times.
    csm_t = sorted(set(list(np.arange(T_START, A.TEI["tig"], 30.0)) + [A.TEI["tig"]] +
                       list(np.arange(A.LIFTOFF_GET, A.INSERTION["get"] + 1, 5.0)) +
                       list(np.arange(A.TPI["tig"] - 60, g(128, 5, 0), 5.0)) + t_lm_coast))
    csm = P.csm_states(y_ep, EP, csm_t)
    y_csm_tei = csm[A.TEI["tig"]]

    # ------------------------------------------------------------------ TEI and the coast home
    log("TEI fit")
    def te_sol(q):
        y_co = P.fly_tei(y_csm_tei, *q)[-1]
        t_m = A.MCC5["tig"] + A.MCC5["dur"] / 2
        y_m0 = P.coast(A.TEI["tco"], y_co, [t_m])[-1]
        y_m1 = P.mcc5(y_m0)
        t_ei, y_ei = P.find_alt_crossing(t_m, y_m1, A.EI["alt_ft"] * FT, A.EI["get"] - 3 * 3600,
                                         A.EI["get"] + 3 * 3600, step=60.0)
        return dict(y_co=y_co, t_m=t_m, y_m0=y_m0, y_m1=y_m1, t_ei=t_ei, y_ei=y_ei)
    best = None
    for pitch in np.arange(-3.0, 3.01, 0.5):
        for dv in (3270.0, 3280.0, 3290.0, 3300.0):
            r = P.ei_residuals(te_sol([pitch, -2.3, dv]))
            if r[0] < 999 and (best is None or r @ r < best[1]):
                best = ([pitch, -2.3, dv], r @ r)
    log("  TEI start", best)
    q_tei, _ = L.solve(lambda q: P.ei_residuals(te_sol(q)), best[0], [0.01, 0.01, 0.1], iters=20)
    log("  q_tei", q_tei)
    te = te_sol(q_tei)
    t_ei, y_ei = te["t_ei"], te["y_ei"]

    t_tei = list(np.arange(A.TEI["tig"], A.TEI["tco"], 2.0))
    Y_tei = P.fly_tei(y_csm_tei, *q_tei, t_out=t_tei)
    tco_out = sorted(set(list(np.arange(np.ceil(A.TEI["tco"] / 60) * 60, te["t_m"], 120.0))))
    Y_c1 = P.coast(A.TEI["tco"], te["y_co"], tco_out)
    tco2 = list(np.arange(np.ceil(te["t_m"] / 120) * 120, t_ei, 120.0))
    Y_c2 = P.coast(te["t_m"], te["y_m1"], tco2)
    te_rows = [(t, y) for t, y in zip(t_tei, Y_tei)] + [(A.TEI["tco"], te["y_co"])] + \
        list(zip(tco_out, Y_c1)) + [(te["t_m"], te["y_m1"])] + list(zip(tco2, Y_c2))

    # ------------------------------------------------------------------ entry
    log("Entry fit")
    KN = np.array([90, 115, 140, 170, 200, 240, 280, 330, 380, 440, 520.0])
    q0 = np.array([-0.24, 0.03, 0.64, 1.0, 0.01, 0.68, -0.15, 0.11, 0.81, -0.53, 1.0])
    q0 = np.r_[np.arctanh(np.clip(q0, -0.9, 0.9)), 123.5]
    E = A.ENTRY_G
    t_drog = A.DROGUE_GET - t_ei

    def entry_rows(qq, dt=1.0):
        return P.fly_entry(t_ei, y_ei, P.bank_fn(qq[:-1], KN, -1, qq[-1]), t_ei + 900, dt=dt,
                           alt_stop_km=24000 * FT)

    def series(rows):
        tau = np.array([x[0] - t_ei for x in rows])
        gg = np.array([x[2] for x in rows])
        V = np.array([np.linalg.norm(x[1][3:]) for x in rows]) / FT
        return tau, gg, V

    aim = np.array([A.SPLASH["lat"], A.SPLASH["lon"]])

    def res_entry(qq):
        rows = entry_rows(qq)
        tau, gg, V = series(rows)
        gi = lambda t: np.interp(t, tau, gg, right=0)
        Vi = lambda t: np.interp(t, tau, V, right=0)
        tv = np.interp(-E["vmin"], -V, tau)
        lat, lon = P.lat_lon(rows[-1][0], rows[-1][1][:3])
        return np.array([(Vi(E["t_final"]) - E["v_final"]) / 100, (gi(tv) - E["gmin"]) / 0.05,
                         (gi(262) - 0.60) / 0.1, (gi(E["t2"]) - E["g2"]) / 0.1,
                         (gi(E["t2"] + 3) - gi(E["t2"] - 3)) / 0.05, (Vi(E["t2"]) - E["v2"]) / 150,
                         (tau[-1] - t_drog) / 3, (lat - aim[0]) / 0.02, (lon - aim[1]) / 0.02])
    q_ent, r_ent = L.solve(res_entry, q0, [0.02] * len(KN) + [1.0], iters=25)
    log("  entry residuals", np.round(r_ent, 2))
    for pass_ in range(2):
        rows_e = entry_rows(q_ent, dt=0.5)
        bankf = P.bank_fn(q_ent[:-1], KN, -1, q_ent[-1])
        t_d, y_d = rows_e[-1][0], rows_e[-1][1]
        t_main = t_d + A.MAIN_DELAY_S
        if pass_ == 1:
            break
        # Drift under the parachutes: aim the drogue point so the splash lands on the documented point.
        rows_tmp = P.fly_chutes(t_d, y_d, t_main, t_main + 1500, 195.0, 6.0)
        sp = np.array(P.lat_lon(rows_tmp[-1][0], rows_tmp[-1][1][:3]))
        dp = np.array(P.lat_lon(t_d, y_d[:3]))
        aim = np.array([A.SPLASH["lat"], A.SPLASH["lon"]]) - (sp - dp)
        log("  chute drift (deg)", sp - dp)
        q_ent, r_ent = L.solve(res_entry, q_ent, [0.02] * len(KN) + [1.0], iters=12)
        log("  entry residuals, pass 2", np.round(r_ent, 2))

    def chute_rows(bd, bm):
        return P.fly_chutes(t_d, y_d, t_main, t_main + 1500, bd, bm)

    def res_ch(p):
        rows = chute_rows(np.exp(p[0]), np.exp(p[1]))
        i = min(range(len(rows)), key=lambda k: abs(rows[k][0] - t_main))
        return np.array([(P.earth_alt(rows[i][1][:3]) / FT - 10000) / 100, (rows[-1][0] - SPLASH_T) / 2])
    pch, _ = L.solve(res_ch, [np.log(60.0), np.log(5.0)], [0.01, 0.01], iters=20)
    beta_d, beta_m = np.exp(pch)
    rows_ch = chute_rows(beta_d, beta_m)
    log("  chutes beta", beta_d, beta_m)

    # ------------------------------------------------------------------ LM track (Moon-centred)
    log("Assembling")
    lm = []          # get, x, y, z (Moon-rel km), alt_ft (site), hdot fps, hvel fps, slat, slon, mass kg, src, phase

    def lm_row(t, y, mass, src, phase):
        r, v = P.moon_state(t, y)
        alt, hd, hv = P.lunar_hdot_hvel(t, y)
        lat, lon, _ = MF.to_selenographic(t, r)
        return [round(t, 2), *np.round(r, 4), round(alt, 1), round(hd, 2), round(hv, 1), round(lat, 4),
                round(lon, 4), round(mass, 1), src, phase]

    rs = DS.site_pos(T_START)
    for t in np.arange(T_START, A.LIFTOFF_GET, 60.0):
        lm.append(lm_row(t, P.liftoff_state(t), p_asc[2], SRC["documented"], 0))
    F_ = A.APS_THRUST_N * 1e-3
    for t, y in zip(t_asc + [A.INSERTION["get"]], list(Y_asc)):
        lm.append(lm_row(t, y, p_asc[2] - F_ * (t - A.LIFTOFF_GET) / p_asc[3], SRC["fitted"], 1))
    seg = [(A.INSERTION["get"], y_ins, A.CSI["tig"], None, None),
           (A.CSI["tig"], y_csi0, A.CSI["tco"], c_csi, A.CSI["dur"]),
           (A.CSI["tco"], y_csi1, A.CDH["tig"], None, None),
           (A.CDH["tig"], y_cdh0, A.CDH["tco"], c_cdh, A.CDH["dur"]),
           (A.CDH["tco"], y_cdh1, A.TPI["tig"], None, None),
           (A.TPI["tig"], y_tpi0, A.TPI["tco"], c_tpi, A.TPI["dur"])]
    mass = m_ins
    for ta, ya, tb, comp, dur in seg:
        ts = list(np.arange(ta + (2.0 if comp is not None else 30.0), tb, 2.0 if comp is not None else 30.0))
        if comp is None:
            Ys = P.coast(ta, ya, ts) if ts else []
            for t, y in zip(ts, Ys):
                lm.append(lm_row(t, y, mass, SRC["propagated"], 2))
        else:
            Ys = P.lvlh_burn(ta, ya, dur, comp, t_out=ts)[:-1] if ts else []
            m0 = mass
            dvk = np.linalg.norm(comp) * FT
            m1 = m0 * np.exp(-dvk / P.VE_RCS)
            for t, y in zip(ts, Ys):
                lm.append(lm_row(t, y, m0 - (m0 - m1) * (t - ta) / dur, SRC["fitted"], 3))
            mass = m1
    # TPI coast, then braking along the line of sight, stationkeeping and docking.
    rel_rows = []

    def rel_of(t, y_l):
        return P.relative_lvlh(t, y_l, csm[t])
    for t, y in zip(t_lm_coast, Y_tpi):
        if t < A.BRAKING_GET:
            lm.append(lm_row(t, y, mass, SRC["propagated"], 2))
    # Range profile through the calls (feet), monotone cubic in time.
    tb = A.BRAKING_GET
    k_b = int(np.argmin([abs(t - tb) for t in t_lm_coast]))
    y_b = Y_tpi[k_b]
    rvec_b = y_b[:3] - csm[t_lm_coast[k_b]][:3]
    R_b = np.linalg.norm(rvec_b) / FT
    Rd_b = ((Y_tpi[k_b + 1][:3] - csm[t_lm_coast[k_b + 1]][:3]) - (Y_tpi[k_b - 1][:3] - csm[t_lm_coast[k_b - 1]][:3]))
    Rd_b = (Rd_b @ rvec_b / np.linalg.norm(rvec_b)) / 10.0 / FT
    knots = [(t_lm_coast[k_b], R_b)] + [(t, r) for t, r, _ in A.RANGE_CALLS if r is not None and t < A.TPF["tig"]] + \
        [(A.TPF["tig"], 1200.0), (A.TPF["tco"], 150.0), (A.STATIONKEEP_GET, 100.0)] + \
        [(t, r) for t, r, _ in A.RANGE_CALLS if r is not None and t > A.TPF["tig"]] + [(A.DOCK["get"] - 150, 0.0)]
    knots.sort()
    kt, kr = np.array([k[0] for k in knots]), np.array([k[1] for k in knots])
    from scipy.interpolate import PchipInterpolator
    range_ft = PchipInterpolator(kt, kr)
    dirs = {t: (y[:3] - csm[t][:3]) for t, y in zip(t_lm_coast, Y_tpi)}
    u_last = None
    t_dock_contact = A.DOCK["get"] - 150
    for t in [t for t in csm_t if t_lm_coast[k_b] <= t <= g(128, 5, 0)]:
        if t in dirs and t < A.TPF["tco"] - 30:
            u = dirs[t] / np.linalg.norm(dirs[t])
            u_last = u
        rr = float(range_ft(min(t, t_dock_contact))) * FT
        y_l = np.r_[csm[t][:3] + u_last * rr, csm[t][3:]]
        if t > t_lm_coast[k_b] + 1:
            lm.append(lm_row(t, y_l, mass, SRC["fitted"], 4 if t < t_dock_contact else 5))
    # Docked, then the ascent stage on its own after jettison (it stayed in orbit).
    for t in [t for t in csm_t if g(128, 5, 0) < t <= A.JETTISON["get"]]:
        lm.append(lm_row(t, csm[t], mass, SRC["propagated"], 5))
    y_jet = csm[min(csm_t, key=lambda t: abs(t - A.JETTISON["get"]))]
    t_after = list(np.arange(np.ceil(A.JETTISON["get"] / 30) * 30 + 30, A.TEI["tco"] + 3600, 60.0))
    t_jet0 = min(csm_t, key=lambda t: abs(t - A.JETTISON["get"]))
    for t, y in zip(t_after, P.coast(t_jet0, y_jet, t_after)):
        lm.append(lm_row(t, y, mass, SRC["propagated"], 6))
    lm.sort(key=lambda r: r[0])

    # Relative motion, insertion to docking: Eagle relative to Columbia (km).
    lm_arr = np.array([r[:4] for r in lm])
    for t in [t for t in csm_t if A.INSERTION["get"] <= t <= A.DOCK["get"]]:
        k = np.searchsorted(lm_arr[:, 0], t)
        if k >= len(lm_arr):
            continue
        if abs(lm_arr[k, 0] - t) > 1e-6:
            if k == 0:
                continue
            a, b = lm_arr[k - 1], lm_arr[k]
            f = (t - a[0]) / (b[0] - a[0])
            rl = a[1:4] + f * (b[1:4] - a[1:4])
        else:
            rl = lm_arr[k, 1:4]
        rc, vc = P.moon_state(t, csm[t])
        y_l = np.r_[rl + D.moon_pos(t), vc + D.moon_vel(t)]
        rel = P.relative_lvlh(t, y_l, csm[t])
        rel_rows.append([round(t, 1), *np.round(rel, 4)])
    rel_rows = rel_rows[::1]

    # Columbia rows (Moon-centred) until 2 h after TEI.
    csm_rows = []

    def csm_row(t, y, src=SRC["fitted"]):
        r, v = P.moon_state(t, y)
        lat, lon, _ = MF.to_selenographic(t, r)
        return [round(t, 2), *np.round(r, 3), round((np.linalg.norm(r) - P.R_SITE) / NMI, 3),
                round(np.linalg.norm(v) / FT, 1), round(lat, 4), round(lon, 4), src]
    for t in csm_t:
        csm_rows.append(csm_row(t, csm[t]))
    for t, y in te_rows:
        if t > A.TEI["tco"] + 2.5 * 3600:
            break
        csm_rows.append(csm_row(t, y))

    # Earth-centred transearth rows: get, x, y, z (km), alt n mi, v ft/s, distance from the Moon km.
    earth_rows = []
    for t, y in te_rows:
        if t < A.TEI["tig"]:
            continue
        tb_ = P.earth_table(t, y)
        earth_rows.append([round(t, 1), *np.round(y[:3], 1), round(tb_["alt"] / NMI, 1), round(tb_["v"] / FT, 1),
                           round(float(np.linalg.norm(y[:3] - D.moon_pos(t))), 1)])

    # Entry rows: get, x, y, z (km), alt ft, inertial v ft/s, Earth-relative v ft/s, g, bank deg, lat, lon, fpa, phase
    entry_rows_out = []
    t_pre = list(np.arange(t_ei - 1200, t_ei, 20.0))
    Y_pre = P.coast(t_ei, y_ei, t_pre[::-1])[::-1]
    allrows = [(t, y, 0.0, 0) for t, y in zip(t_pre, Y_pre)] + \
        [(t, y, gg, 1) for t, y, gg in rows_e[::2]] + [(t, y, gg, 2) for t, y, gg in rows_ch[::2]]
    for t, y, gg, ph in allrows:
        tb_ = P.earth_table(t, y)
        bank = np.degrees(bankf(t - t_ei)) if ph == 1 else 0.0
        entry_rows_out.append([round(t, 1), *np.round(y[:3], 3), round(P.earth_alt(y[:3]) / FT, 0),
                               round(tb_["v"] / FT, 1), round(tb_["v_rel"] / FT, 1), round(gg, 3), round(bank, 1),
                               round(tb_["geod_lat"], 4), round(tb_["lon"], 4), round(tb_["fpa"], 3), ph])
    splash_ll = P.lat_lon(rows_ch[-1][0], rows_ch[-1][1][:3])

    # ------------------------------------------------------------------ checks
    log("Checks")
    # Which radius are the ABTN lunar heights measured from?
    rows_ref = [(10.0, 5537.9, 48.0, 9.4), (48.4, 5376.6, 49.3, 45.7), (44.1, 5391.5, 47.4, 42.1),
                (44.0, 5413.2, 61.7, 43.7), (62.7, 5326.9, 62.7, 54.0)]

    def visviva_rms(R):
        d = []
        for h, v, ap, pe in rows_ref:
            r = R + h * NMI
            a = R + (ap + pe) / 2 * NMI
            d.append(np.sqrt(D.MU_MOON * (2 / r - 1 / a)) / FT - v)
        return np.sqrt(np.mean(np.square(d)))
    Rs = np.arange(1733.0, 1740.0, 0.01)
    R_best = Rs[np.argmin([visviva_rms(R) for R in Rs])]
    add("ref", "Speed misfit, heights from the 1738.09 km sphere", None, visviva_rms(RM), "ft/s", "rms over five ABTN rows")
    add("ref", "Speed misfit, heights from the landing-site radius", None, visviva_rms(P.R_SITE), "ft/s", "rms over five ABTN rows")
    add("ref", "Radius that fits best", None, R_best, "km", f"landing site: {P.R_SITE:.2f} km (TRW)")

    yins = y_ins
    alt_i, hd_i, hv_i = P.lunar_hdot_hvel(A.INSERTION["get"], yins)
    h_i, v_i, _ = P.lunar_table(A.INSERTION["get"], yins)
    add("ascent", "Insertion, 124:29:15.7: height", A.INSERTION["dsky_alt_ft"], alt_i, "ft", "fitted; DSKY readout")
    add("ascent", "Insertion: altitude rate", A.INSERTION["dsky_hdot"], hd_i, "ft/s", "fitted; DSKY readout")
    add("ascent", "Insertion: speed", A.INSERTION["v"], v_i, "ft/s", "fitted; DSKY 5,537.3")
    add("ascent", "Ascent delta-v", A.INSERTION["dv_fps"], p_asc[3] * np.log(p_asc[2] / m_ins) / FT, "ft/s", "fitted")
    ap, pe = P.lunar_orbit(A.INSERTION["get"] + 60, P.coast(A.INSERTION["get"], yins, [A.INSERTION["get"] + 60])[-1])
    add("ascent", "Insertion orbit: apolune", A.INSERTION["apo"], ap, "n mi", "onboard 47.3, AGS 46.6")
    add("ascent", "Insertion orbit: perilune", A.INSERTION["peri"], pe, "n mi", "onboard and AGS 9.5")
    add("ascent", "Liftoff mass", None, p_asc[2] / LB, "lb", "fitted")
    add("ascent", "Propellant burned", A.APS_PROPELLANT_LB, (p_asc[2] - m_ins) / LB, "lb", "BENNETT: loaded, approximate")
    add("ascent", "Exhaust velocity", None, p_asc[3] * 1000 / 9.80665, "s", "specific impulse; fitted")
    rl, vl = P.moon_state(A.INSERTION["get"], yins)
    rc_, vc_ = P.moon_state(A.INSERTION["get"], csm[min(csm_t, key=lambda t: abs(t - A.INSERTION["get"]))])
    hc = np.cross(rc_, vc_)
    hc /= np.linalg.norm(hc)
    add("ascent", "Out-of-plane rate relative to Columbia's orbit", A.INSERTION["ydot_fps"], vl @ hc / FT, "ft/s",
        "1969: Collins's estimate, read up at 124:58")
    add("ascent", "Out-of-plane distance at insertion", None, (rl @ hc) / NMI, "n mi")
    asc_calls = []
    for tc, hv, hd, al in A.ASCENT_CALLS:
        y = P.fly_ascent(p_asc, h_csm, [tc])[0]
        a_, d_, v_ = P.lunar_hdot_hvel(tc, y)
        asc_calls.append([tc, hv, hd, al, round(v_, 1), round(d_, 1), round(a_, 0)])
        lab = gs(tc)
        if hv is not None:
            add("calls", f"Horizontal speed called at {lab}", hv, v_, "ft/s", "")
        if hd is not None:
            add("calls", f"Climb rate called at {lab}", hd, d_, "ft/s", "")
        if al is not None:
            add("calls", f"Altitude called at {lab}", al, a_, "ft", "")
    for tc, vgo in A.ASCENT_VGO:
        y = P.fly_ascent(p_asc, h_csm, [tc])[0]
        _, vnow, _ = P.lunar_table(tc, y)
        add("calls", f"Velocity to go called at {gs(tc)}", vgo, A.INSERTION["v"] - vnow, "ft/s",
            "speed still to gain, from the model")
    rel_ins = P.relative_lvlh(A.INSERTION["get"], yins, csm[min(csm_t, key=lambda t: abs(t - A.INSERTION["get"]))])
    add("ascent", "Columbia ahead of Eagle at insertion", None, rel_ins[0] / NMI * -1, "n mi", "along the orbit")

    add("rdv", "CSI: delta-v", A.CSI["dv_fps"], np.linalg.norm(c_csi), "ft/s", "orbit after fitted: 49.3 x 45.7")
    add("rdv", "CSI: forward component", None, c_csi[0], "ft/s")
    add("rdv", "CSI: radial component (down +)", None, c_csi[2], "ft/s")
    h_, v_, _ = P.lunar_table(A.CSI["tig"], y_csi0)
    add("rdv", "CSI ignition: height", A.CSI["alt_ign"], h_, "n mi")
    add("rdv", "CSI ignition: speed", A.CSI["v_ign"], v_, "ft/s")
    h_, v_, _ = P.lunar_table(A.CSI["tco"], y_csi1)
    add("rdv", "CSI cutoff: height", A.CSI["alt_co"], h_, "n mi")
    add("rdv", "CSI cutoff: speed", A.CSI["v_co"], v_, "ft/s")
    for k, lab in enumerate(["X (forward)", "Y (right)", "Z (down)"]):
        add("rdv", f"CDH Noun 81 {lab}", A.CDH["noun81"][k], c_cdh[k], "ft/s", "Y taken from the readout" if k == 1 else "")
    add("rdv", "CDH: delta-v", A.CDH["dv_fps"], np.linalg.norm(c_cdh), "ft/s", "orbit after fitted: 47.4 x 42.1")
    dh = -P.relative_lvlh(A.CDH["tco"] + 600, P.coast(A.CDH["tco"], y_cdh1, [A.CDH["tco"] + 600])[-1],
                          P.csm_states(y_ep, EP, [A.CDH["tco"] + 600])[A.CDH["tco"] + 600])[2] / NMI
    add("rdv", "Height difference after CDH (Delta H)", A.DELTA_H_AGS, dh, "n mi", "AGS figure read out at 124:39")
    h_, v_, _ = P.lunar_table(A.TPI["tig"], y_tpi0)
    add("rdv", "TPI ignition: height", A.TPI["alt_ign"], h_, "n mi")
    add("rdv", "TPI ignition: speed", A.TPI["v_ign"], v_, "ft/s")
    rel_tpi = P.relative_lvlh(A.TPI["tig"], y_tpi0, csm[min(csm_t, key=lambda t: abs(t - A.TPI["tig"]))])
    add("rdv", "Range to Columbia at TPI", None, np.linalg.norm(rel_tpi) / NMI, "n mi")
    yp = P.coast(A.CDH["tco"], y_cdh1, [A.PAO_TPI["get"]])[-1]
    cp = P.csm_states(y_ep, EP, [A.PAO_TPI["get"]])[A.PAO_TPI["get"]]
    d0, dv_ = yp[:3] - cp[:3], yp[3:] - cp[3:]
    add("rdv", "Range, 6½ minutes before TPI", A.PAO_TPI["range_nmi"], np.linalg.norm(d0) / NMI, "n mi", "PAO commentary; used in the fit")
    add("rdv", "Range rate, 6½ minutes before TPI", A.PAO_TPI["rdot_fps"], (d0 @ dv_) / np.linalg.norm(d0) / FT, "ft/s", "PAO commentary; used in the fit")
    add("rdv", "Line-of-sight elevation at TPI", None, np.degrees(np.arctan2(-rel_tpi[2], -rel_tpi[0])), "deg")
    for k, lab in enumerate(["X (forward)", "Y (right)", "Z (down)"]):
        add("rdv", f"TPI Noun 81 {lab}", A.TPI["noun81"][k], c_tpi[k], "ft/s", "solved so Eagle meets Columbia at the end of TPF")
    add("rdv", "TPI: delta-v", A.TPI["dv_fps"], np.linalg.norm(c_tpi), "ft/s")
    h_, v_, _ = P.lunar_table(A.TPI["tco"], y_tpi1)
    add("rdv", "TPI cutoff: height", A.TPI["alt_co"], h_, "n mi")
    add("rdv", "TPI cutoff: speed", A.TPI["v_co"], v_, "ft/s")
    ya = Y_tpi[-1]
    add("rdv", "Closing speed if Eagle had not braked", None, np.linalg.norm(ya[3:] - csm[A.TPF["tco"]][3:]) / FT,
        "ft/s", f"braking and TPF total in ABTN: {A.TPF['dv_fps']} ft/s for TPF alone")
    for tc, rr, rd in A.RANGE_CALLS[:2]:
        k = int(np.argmin([abs(t - tc) for t in t_lm_coast]))
        rng = np.linalg.norm(Y_tpi[k][:3] - csm[t_lm_coast[k]][:3]) / FT
        add("rdv", f"Range at {gs(tc)}, without braking", rr, rng, "ft", "the flown path is drawn through the calls")

    for t, (a_, v_), lab in [(A.DOCK["get"], (A.DOCK["alt"], A.DOCK["v"]), "Docking"),
                             (A.JETTISON["get"], (A.JETTISON["alt"], A.JETTISON["v"]), "Ascent stage jettison"),
                             (A.FINAL_SEP["tig"], (A.FINAL_SEP["alt_ign"], A.FINAL_SEP["v_ign"]), "Final separation"),
                             (A.TEI["tig"], (A.TEI["alt_ign"], A.TEI["v_ign"]), "TEI ignition")]:
        hh, vv, _ = P.lunar_table(t, P.csm_states(y_ep, EP, [t])[t])
        add("csm", f"{lab}, {gs(t)}: height", a_, hh, "n mi")
        add("csm", f"{lab}: speed", v_, vv, "ft/s")
    tt_ = A.FINAL_SEP["tco"] + 300
    ap, pe = P.lunar_orbit(tt_, P.csm_states(y_ep, EP, [tt_])[tt_])
    add("csm", "Orbit after final separation: apolune", A.FINAL_SEP["apo"], ap, "n mi")
    add("csm", "Orbit after final separation: perilune", A.FINAL_SEP["peri"], pe, "n mi")
    for tl in LOS:
        c = min([c for c in crossings if c[1] == "LOS"], key=lambda c: abs(c[0] - tl))
        add("signal", f"Loss of signal near {gs(tl)}", 0.0, c[0] - tl, "s", "time difference; 1969 times from Mission Control's calls")
    for ta in AOS:
        c = min([c for c in crossings if c[1] == "AOS"], key=lambda c: abs(c[0] - ta))
        add("signal", f"Acquisition of signal near {gs(ta)}", 0.0, c[0] - ta, "s", "")

    h_, v_, g_ = P.lunar_table(A.TEI["tco"], te["y_co"])
    add("tei", "TEI delta-v", A.TEI["dv_fps"], q_tei[2], "ft/s", f"fitted; PAD {np.linalg.norm(A.TEI['pad']):.1f}, PAO 3,283.5")
    add("tei", "TEI pitch from the PAD direction", None, q_tei[0], "deg", "fitted")
    add("tei", "TEI yaw from the PAD direction", None, q_tei[1], "deg", "fitted")
    add("tei", "TEI cutoff: height", A.TEI["alt_co"], h_, "n mi")
    add("tei", "TEI cutoff: speed", A.TEI["v_co"], v_, "ft/s")
    add("tei", "TEI cutoff: flight-path angle", A.TEI["fpa_co"], g_, "deg")
    tb_ = P.earth_table(A.MCC5["tig"], P.coast(A.TEI["tco"], te["y_co"], [A.MCC5["tig"]])[-1])
    add("tei", "MCC-5 ignition: height", A.MCC5["alt_ign"], tb_["alt"] / NMI, "n mi")
    add("tei", "MCC-5 ignition: speed", A.MCC5["v_ign"], tb_["v"] / FT, "ft/s")
    add("tei", "MCC-5 ignition: flight-path angle", A.MCC5["fpa_ign"], tb_["fpa"], "deg")
    ysep = P.coast(te["t_m"], te["y_m1"], [A.CM_SM_SEP["get"]])[-1]
    tb_ = P.earth_table(A.CM_SM_SEP["get"], ysep)
    add("tei", "CM/SM separation: height", A.CM_SM_SEP["alt"], tb_["alt"] / NMI, "n mi")
    add("tei", "CM/SM separation: speed", A.CM_SM_SEP["v"], tb_["v"] / FT, "ft/s")
    add("tei", "CM/SM separation: flight-path angle", A.CM_SM_SEP["fpa"], tb_["fpa"], "deg")
    tb_ = P.earth_table(t_ei, y_ei)
    add("ei", "Entry interface time", A.EI["get"] / 3600, t_ei / 3600, "h", "fitted")
    add("ei", "Entry interface: latitude", A.EI["geod_lat"], tb_["geod_lat"], "deg", "fitted")
    add("ei", "Entry interface: longitude", A.EI["lon"], tb_["lon"], "deg", "fitted")
    add("ei", "Entry interface: flight-path angle", A.EI["fpa"], tb_["fpa"], "deg", "fitted")
    add("ei", "Entry interface: speed", A.EI["v"], tb_["v"] / FT, "ft/s")
    add("ei", "Entry interface: azimuth", A.EI["az"], tb_["head"], "deg")

    tau, gg, V = series(rows_e)
    hh = np.array([P.earth_alt(x[1][:3]) for x in rows_e]) / FT
    w = (tau < 150)
    i1 = int(np.argmax(np.where(w, gg, -1)))
    add("entry", "First load-factor peak", E["g1"], gg[i1], "g", "MR: about 6.5")
    add("entry", "  time after entry interface", E["t1"], tau[i1], "s")
    add("entry", "  speed", E["v1"], V[i1], "ft/s")
    w2 = (tau > 60) & (tau < 150)
    ih = int(np.argmin(np.where(w2, hh, 1e9)))
    add("entry", "Lowest point of the first dip", E["hmin_ft"], hh[ih], "ft")
    add("entry", "  time after entry interface", E["t_hmin"], tau[ih], "s")
    i005 = int(np.argmax(gg > 0.05))
    add("entry", "0.05 g reached", E["t_005g"], tau[i005], "s", "after entry interface")
    add("entry", "  altitude", E["h_005g"], hh[i005], "ft")
    tv = np.interp(-E["vmin"], -V, tau)
    add("entry", f"Load factor as speed falls through {E['vmin']:,.0f} ft/s", E["gmin"], np.interp(tv, tau, gg), "g", "fitted")
    add("entry", "Speed at the start of the final phase (200 s)", E["v_final"], np.interp(200, tau, V), "ft/s", "fitted")
    w3 = (tau > 300)
    i2 = int(np.argmax(np.where(w3, gg, -1)))
    add("entry", "Second load-factor peak", E["g2"], gg[i2], "g", "fitted")
    add("entry", "  time after entry interface", E["t2"], tau[i2], "s", "fitted")
    add("entry", "  speed", E["v2"], V[i2], "ft/s", "fitted")
    add("entry", "Drogues, 24,000 ft", A.DROGUE_GET - t_ei, t_d - t_ei, "s", "after entry interface; fitted")
    up = (tau > 90) & (tau < 200)
    add("entry", "Highest point of the loft", None, hh[up].max(), "ft")
    lat_d, lon_d = P.lat_lon(t_d, y_d[:3])
    def gc_nmi(a, b):
        p1, l1, p2, l2 = map(np.radians, (a[0], a[1], b[0], b[1]))
        return np.degrees(np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(l2 - l1), -1, 1))) * 60
    ei_ll = (P.earth_table(t_ei, y_ei)["geod_lat"], P.earth_table(t_ei, y_ei)["lon"])
    add("entry", "Ground range, entry interface to splashdown", 1497.0, gc_nmi(ei_ll, splash_ll), "n mi", "ENTRY: relative range at entry")
    add("splash", "Splashdown latitude", A.SPLASH["lat"], splash_ll[0], "deg", "onboard computer; fitted at drogue deploy")
    add("splash", "Splashdown longitude", A.SPLASH["lon"], splash_ll[1], "deg", "")
    add("splash", "Distance from the target point", 1.7, gc_nmi(A.ENTRY_TARGET, splash_ll), "n mi", "ENTRY: onboard, 1.7")
    add("splash", "Splashdown time", SPLASH_T / 3600, rows_ch[-1][0] / 3600, "h", "fitted")
    add("splash", "Drogue ballistic coefficient", None, beta_d, "kg/m²", "fitted: mains at 10,000 ft 50 s after drogues")
    add("splash", "Main-chute ballistic coefficient", None, beta_m, "kg/m²", "fitted to the splashdown time")
    vz_end = np.array([P.earth_alt(x[1][:3]) for x in rows_ch[-6:]])
    add("splash", "Descent rate at splashdown", None, -(vz_end[-1] - vz_end[0]) / (rows_ch[-1][0] - rows_ch[-6][0]) / FT, "ft/s",
        "about 31 ft/s on three mains (design)")
    tc_ = np.array([x[0] for x in rows_ch])
    hc_ = np.array([P.earth_alt(x[1][:3]) for x in rows_ch]) / FT
    for tcall, alt in A.CHUTE_CALLS:
        add("splash", f"Altitude called at {gs(tcall)}", alt, np.interp(tcall + A.CHUTE_CALL_SHIFT_S, tc_, hc_), "ft",
            "transcript time moved 17 s to match MR's splashdown")
    add("entry", "Ballistic coefficient used", None, P.BETA, "kg/m²", "CM 12,148 lb, CD 1.29 (assumed), 154 in shield")

    # ------------------------------------------------------------------ tables for the page
    moon_tab = [[round(t), *np.round(D.moon_pos(t), 1)] for t in np.arange(T_START - 1200, T_END + 1201, 600.0)]
    sun = []
    for t in np.arange(T_START - 3600, T_END + 3601, 1800.0):
        s_ = D.sun_pos(t) - D.moon_pos(t)
        slat, slon, _ = MF.to_selenographic(t, s_)
        sg = D.sun_pos(t)
        sun.append([round(t), *np.round(s_ / np.linalg.norm(s_), 6), round(slat, 4), round(slon, 4),
                    *np.round(sg / np.linalg.norm(sg), 6)])
    orient = [[round(t), *np.round(MF.body_to_work(t).ravel(), 8)] for t in np.arange(T_START - 1800, T_END + 1801, 900.0)]
    events = [
        ["Liftoff from Tranquility Base", A.LIFTOFF_GET], ["Orbit insertion", A.INSERTION["get"]],
        ["CSI", A.CSI["tig"]], ["CDH", A.CDH["tig"]], ["TPI", A.TPI["tig"]], ["Braking", A.BRAKING_GET],
        ["TPF", A.TPF["tig"]], ["Stationkeeping", A.STATIONKEEP_GET], ["Docked", A.DOCK["get"]],
        ["Eagle jettisoned", A.JETTISON["get"]], ["Final separation", A.FINAL_SEP["tig"]],
        ["TEI", A.TEI["tig"]], ["TEI cutoff", A.TEI["tco"]], ["MCC-5", A.MCC5["tig"]],
        ["CM/SM separation", A.CM_SM_SEP["get"]], ["Entry interface", t_ei], ["Drogues", t_d],
        ["Mains", t_main], ["Splashdown", rows_ch[-1][0]],
    ]
    burns = [["CSI", A.CSI["tig"], A.CSI["dur"], *np.round(c_csi, 2)], ["CDH", A.CDH["tig"], A.CDH["dur"], *np.round(c_cdh, 2)],
             ["TPI", A.TPI["tig"], A.TPI["dur"], *np.round(c_tpi, 2)]]
    sgn = lambda t: float(np.mod(F.gmst_rad(t), 2 * np.pi))
    out = dict(
        meta=dict(
            title="Apollo 11 Part 4: rendezvous, return and entry",
            frame="Moon-centred (lunar tables) and Earth-centred (Earth tables), mean equator and equinox of 16 July 1969, km",
            lm_columns=["get", "x", "y", "z", "alt_ft", "hdot_fps", "hvel_fps", "slat", "slon", "mass_kg", "src", "phase"],
            csm_columns=["get", "x", "y", "z", "alt_nmi", "v_fps", "slat", "slon", "src"],
            earth_columns=["get", "x", "y", "z", "alt_nmi", "v_fps", "moon_km"],
            entry_columns=["get", "x", "y", "z", "alt_ft", "v_fps", "vrel_fps", "g", "bank_deg", "lat", "lon", "fpa", "phase"],
            rel_columns=["get", "downrange_km", "cross_km", "radial_km"],
            src_codes=SRC, r_moon_ref=RM, r_site=P.R_SITE, landing_site=list(A.LM_SITE), t_start=T_START, t_end=T_END,
            liftoff=A.LIFTOFF_GET, insertion=A.INSERTION["get"], csi=[A.CSI["tig"], A.CSI["tco"]],
            cdh=[A.CDH["tig"], A.CDH["tco"]], tpi=[A.TPI["tig"], A.TPI["tco"]], braking=A.BRAKING_GET,
            tpf=[A.TPF["tig"], A.TPF["tco"]], stationkeep=A.STATIONKEEP_GET, dock=A.DOCK["get"],
            dock_contact=t_dock_contact, jettison=A.JETTISON["get"],
            final_sep=[A.FINAL_SEP["tig"], A.FINAL_SEP["tco"]], tei=[A.TEI["tig"], A.TEI["tco"]],
            mcc5=A.MCC5["tig"], cm_sm_sep=A.CM_SM_SEP["get"], ei=t_ei, drogue=t_d, main=t_main,
            splash=rows_ch[-1][0], splash_ll=list(splash_ll), target_ll=list(A.ENTRY_TARGET),
            gmst0=sgn(0.0), omega_earth=F.OMEGA_EARTH, re_eq=F.RE_EQ, aps_n=A.APS_THRUST_N,
            mass_liftoff=p_asc[2], ve_aps=p_asc[3]),
        lm=lm, csm=csm_rows, rel=rel_rows, earth=earth_rows, entry=entry_rows_out, moon=moon_tab, sun=sun,
        moon_orient=orient, events=events, burns=burns, ascent_calls=asc_calls,
        range_calls=[[t, r, rd] for t, r, rd in A.RANGE_CALLS],
        los=[[round(c[0], 1), c[1]] for c in crossings],
        checks=[[s_, l_, d_ if d_ is None else round(d_, 4), round(m_, 4), u_, n_] for s_, l_, d_, m_, u_, n_ in checks],
        fit=dict(p_asc=list(p_asc), p_csm=list(p_csm), q_tei=list(q_tei), q_entry=list(q_ent), bank_knots=list(KN),
                 beta=P.BETA, beta_drogue=beta_d, beta_main=beta_m),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for s_, l_, d_, m_, u_, n_ in checks:
        log(f"{s_:7s} {l_:60s} doc={'' if d_ is None else round(d_, 3)!s:<10} model={m_:.3f} {u_}  {n_}")
    log("rows lm", len(lm), "csm", len(csm_rows), "rel", len(rel_rows), "earth", len(earth_rows), "entry",
        len(entry_rows_out), "->", out_path, f"{time.time() - t_clock:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "part4.json",
         sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "../../post3/data/part3.json"))
