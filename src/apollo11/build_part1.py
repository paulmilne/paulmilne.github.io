"""Build the data file for Part 1: launch to translunar injection.

    python3 build_part1.py  [output.json]

Pipeline (each stage starts from a documented anchor):
  1. Ascent, 0 to 709 s: the 1969 ascent table, interpolated (PCHIP).
  2. Parking orbit: propagated from the documented insertion state with
     J2, Moon and Sun, plus a small along-track acceleration solved so the
     speed at TLI ignition matches the documented value.
  3. TLI burn: a fitted constant-thrust burn (tli_burn.py) from the
     propagated ignition state to the documented TLI state; the leftover
     miss is spread linearly across the burn.
  4. Translunar coast: propagated from the documented TLI state.
"""
import json
import sys
from pathlib import Path
import numpy as np

import anchors as A
import dynamics as D
import frames as F
import tli_burn as B

SRC = dict(documented=0, interpolated=1, propagated=2, fitted=3)


def pchip(x, y, xq):
    """Monotone cubic interpolation (Fritsch-Carlson), numpy only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[i]) / h[i]
    return ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * h[i] * m[i]
            + (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * h[i] * m[i + 1])


def row(t, r, v, src):
    q = F.table_from_state(t, r, v)
    rm = np.linalg.norm(r - D.moon_pos(t))
    return [round(t, 2), *np.round(r, 2), round(q["lat"], 4), round(q["lon"], 4),
            round(q["alt"], 3), round(q["v"], 5), round(q["v_rel"], 5), round(q["fpa"], 3),
            round(q["head"], 3), round(q["r"], 2), round(rm, 1), SRC[src]]


def elements(r, v):
    mu = D.MU_EARTH
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1 / (2 / rn - vn**2 / mu)
    h = np.cross(r, v)
    e = np.linalg.norm(np.cross(v, h) / mu - r / rn)
    return a, e, np.degrees(np.arccos(h[2] / np.linalg.norm(h)))


def main(out_path):
    rows, checks = [], []

    # 0. On the pad: the stack rides the rotating Earth for a minute before liftoff.
    L = A.ASCENT[0]
    for t in np.arange(-60.0, L[1], 2.0):
        r, v = F.state_from_table(t, L[6], L[7], L[2] * F.NMI, L[5] * F.FT, 0.0, 90.0)
        rows.append(row(t, r, v, "documented"))

    # 1. Ascent: interpolate the documented table.
    tab = np.array([a[1:] for a in A.ASCENT])
    t_tab = tab[:, 0]
    t_asc = np.unique(np.r_[np.arange(L[1], t_tab[-1], 2.0), t_tab])
    cols = {k: pchip(t_tab, tab[:, j], t_asc)
            for k, j in dict(alt=1, v=4, lat=5, lon=6, fpa=7, head=8).items()}
    for i, t in enumerate(t_asc):
        r, v = F.state_from_table(t, cols["lat"][i], cols["lon"][i], cols["alt"][i] * F.NMI,
                                  cols["v"][i] * F.FT, cols["fpa"][i], cols["head"][i])
        rows.append(row(t, r, v, "documented" if np.any(np.isclose(t, t_tab)) else "interpolated"))

    # 2. Parking orbit.
    E = A.ASCENT[-1]
    t_eoi = E[1]
    r0, v0 = F.state_from_table(t_eoi, E[6], E[7], E[2] * F.NMI, E[5] * F.FT, E[8], E[9])
    a, e, inc = elements(r0, v0)
    po = A.PARKING_ORBIT
    checks += [
        ["Parking orbit apogee (n mi, spherical Earth)", po["apogee_nmi"], (a * (1 + e) - F.RE_EQ) / F.NMI],
        ["Parking orbit perigee (n mi, spherical Earth)", po["perigee_nmi"], (a * (1 - e) - F.RE_EQ) / F.NMI],
        ["Parking orbit period (min)", po["period_min"], 2 * np.pi * np.sqrt(a**3 / D.MU_EARTH) / 60],
        ["Parking orbit inclination (deg)", po["incl_deg"], inc],
    ]
    t_ign = A.TLI_IGNITION_GET
    y0 = np.r_[r0, v0]

    def coast_to_ign(av):
        th = (lambda t, r, v: av * v / np.linalg.norm(v)) if av else None
        return D.make_rhs(th)

    y_free = D.integrate(coast_to_ign(0.0), t_eoi, y0, [t_ign], h_max=60)[0]
    lo, hi = 0.0, 3e-6
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        y = D.integrate(coast_to_ign(mid), t_eoi, y0, [t_ign], h_max=60)[0]
        lo, hi = (mid, hi) if np.linalg.norm(y[3:]) / F.FT > A.TLI_IGNITION_SFV_FTS else (lo, mid)
    a_vent = 0.5 * (lo + hi)
    checks += [
        ["Speed at TLI ignition, gravity only (ft/s)", A.TLI_IGNITION_SFV_FTS,
         np.linalg.norm(y_free[3:]) / F.FT],
        ["Along-track acceleration needed to match it (mm/s²)", None, a_vent * 1e6],
    ]
    t_po = np.unique(np.r_[np.arange(t_eoi, t_ign, 20.0), t_ign])
    Y = D.integrate(coast_to_ign(a_vent), t_eoi, y0, t_po, h_max=60)
    for t, y in zip(t_po[1:], Y[1:]):          # t_eoi is already the last ascent row
        rows.append(row(t, y[:3], y[3:], "propagated"))
    y_ign = Y[-1]

    # 3. TLI burn, fitted to the documented TLI state.
    T = A.TLI
    r_tli, v_tli = F.state_from_table(T["get"], T["lat"], T["lon"], 0, T["v_kms"], T["fpa"],
                                      T["head"], radius_km=T["radius_km"])
    p, res = B.fit(t_ign, y_ign, A.TLI_CUTOFF_GET, T["get"], np.r_[r_tli, v_tli])
    t_burn = np.unique(np.r_[np.arange(t_ign, A.TLI_CUTOFF_GET, 3.0)[1:], A.TLI_CUTOFF_GET])
    Yb, y_end = B.fly(p, t_ign, y_ign, A.TLI_CUTOFF_GET, T["get"], t_out=t_burn[:-1])
    miss = np.r_[r_tli, v_tli] - y_end
    t_post = np.unique(np.r_[np.arange(A.TLI_CUTOFF_GET, T["get"], 2.0), T["get"]])[1:]
    Yp = D.integrate(D.make_rhs(), A.TLI_CUTOFF_GET, Yb[-1], t_post)
    span = T["get"] - t_ign
    for t, y in zip(np.r_[t_burn, t_post], np.vstack([Yb, Yp])):
        y = y + miss * (t - t_ign) / span
        rows.append(row(t, y[:3], y[3:], "documented" if t == T["get"] else "fitted"))
    a0, pb, pk, yb_, yk = p
    checks += [
        ["TLI miss before correction, position (km)", None, float(np.linalg.norm(res[:3]))],
        ["TLI miss before correction, velocity (m/s)", None, float(np.linalg.norm(res[3:]) * 1e3)],
        ["Fitted initial acceleration (m/s²)", None, a0 * 1e3],
        ["Fitted final acceleration (g)", None, a0 / (1 - a0 * A.TLI_BURN_S / B.VE) / 9.80665e-3],
        ["Fitted pitch above velocity vector at ignition (deg)", None, np.degrees(pb)],
        ["Ideal (rocket-equation) delta-v of burn (m/s)", None, -B.VE * np.log(1 - a0 * A.TLI_BURN_S / B.VE) * 1e3],
        ["Speed gained across burn (m/s)", None, (T["v_kms"] - np.linalg.norm(y_ign[3:])) * 1e3],
    ]
    a, e, inc = elements(r_tli, v_tli)
    checks += [
        ["TLI orbit eccentricity", None, e],
        ["TLI orbit semi-major axis (km)", None, a],
        ["TLI orbit inclination (deg)", T["incl"], inc],
        ["TLI characteristic energy C3 (km²/s²)", None, -D.MU_EARTH / a],
    ]

    # 4. Translunar coast to six hours, from the documented TLI state, with the
    #    evasive burn at 4:40 (PAD vector, ABTN magnitude; see Part 2).
    import build_part2
    c = build_part2.coast_to_mcc()
    y_tli = np.r_[r_tli, v_tli]
    t_c = np.arange(T["get"] + 30.0, c["te"], 30.0)
    Yc = D.integrate(D.make_rhs(), T["get"], y_tli, t_c)
    for t, y in zip(t_c, Yc):
        rows.append(row(t, y[:3], y[3:], "propagated"))
    t_c2 = np.r_[c["te"], np.arange(np.ceil(c["te"] / 30) * 30, 6 * 3600 + 1, 30.0)]
    Yc2 = D.integrate(D.make_rhs(), c["te"], c["y_ev"], t_c2)
    for t, y in zip(t_c2, Yc2):
        rows.append(row(t, y[:3], y[3:], "propagated"))

    # The long view: no midcourse corrections. Documented TLI state and evasive burn,
    # re-anchored to the ABTN state at the time MCC-2 was made, then left alone.
    t_a = np.arange(T["get"], c["te"], 600.0)
    Ya = D.integrate(D.make_rhs(), T["get"], y_tli, t_a)
    t_b = np.arange(np.ceil(c["te"] / 600) * 600, c["t_mcc_ign"], 600.0)
    Yb_ = D.integrate(D.make_rhs(), c["te"], c["y_ev"], t_b)
    t_c3 = np.arange(c["t_mcc_ign"], 160 * 3600, 600.0)
    Yc3 = D.integrate(D.make_rhs(), c["t_mcc_ign"], c["y_mcc_anchor"], t_c3)
    t_l = np.r_[t_a, t_b, t_c3]
    Yl = np.vstack([Ya, Yb_, Yc3])
    dm = np.array([np.linalg.norm(y[:3] - D.moon_pos(t)) for t, y in zip(t_l, Yl)])
    i_min = int(dm.argmin())
    t_fine = np.arange(t_l[i_min] - 1200, t_l[i_min] + 1200, 10.0)
    Yf = D.integrate(D.make_rhs(), c["t_mcc_ign"], c["y_mcc_anchor"], t_fine)
    dmf = np.array([np.linalg.norm(y[:3] - D.moon_pos(t)) for t, y in zip(t_fine, Yf)])
    j = int(dmf.argmin())
    re = np.linalg.norm(Yl[:, :3], axis=1)
    r_ei = F.RE_EQ + 400000 * F.FT          # entry interface, 400,000 ft
    k = i_min + int(np.argmax(re[i_min:] < r_ei))
    t_ref = np.arange(t_l[k - 1], t_l[k] + 1, 1.0)
    Yr = D.integrate(D.make_rhs(), c["t_mcc_ign"], c["y_mcc_anchor"], t_ref)
    kk = int(np.argmax(np.linalg.norm(Yr[:, :3], axis=1) < r_ei))
    q_ei = F.table_from_state(t_ref[kk], Yr[kk, :3], Yr[kk, 3:])
    Yl = Yl[: k + 1]
    t_l = t_l[: k + 1]
    checks += [
        ["Arrival at the Moon, GET (h); 1969: lunar orbit insertion", round(A.LOI1_IGNITION_GET / 3600, 3), t_fine[j] / 3600],
        ["Height above the Moon at arrival (n mi); 1969: at insertion", A.LOI1_IGNITION_ALT_NMI,
         (dmf[j] - D.R_MOON) / F.NMI],
        ["Free return reaches 400,000 ft, GET (h)", None, t_ref[kk] / 3600],
        ["Free return entry flight-path angle (deg)", None, q_ei["fpa"]],
        ["Free return entry speed (ft/s)", None, q_ei["v"] / F.FT],
    ]

    moon = [[round(t, 0), *np.round(D.moon_pos(t), 0)] for t in np.arange(-3600, 160 * 3600 + 1, 1800.0)]
    sun = [[round(t, 0), *np.round(D.sun_pos(t) / np.linalg.norm(D.sun_pos(t)), 6)]
           for t in np.arange(-3600, 8 * 3600 + 1, 1800.0)]
    events = [[a_[0], round(a_[1], 2)] for a_ in A.ASCENT] + [
        ["S-IVB second burn ignition (TLI)", A.TLI_IGNITION_GET],
        ["S-IVB second burn cutoff", A.TLI_CUTOFF_GET],
        ["Translunar injection", T["get"]],
        ["CSM separates from S-IVB", A.CSM_SEPARATION_GET],
        ["CSM docks with LM", A.DOCKING_GET],
        ["Evasive burn", A.EVASIVE["tig"]],
    ]
    out = dict(
        meta=dict(
            title="Apollo 11 Part 1: launch to translunar injection",
            frame="Geocentric, mean equator and equinox of 16 July 1969, km",
            columns=["get", "x", "y", "z", "lat", "lon", "alt_km", "v_kms", "vrel_kms", "fpa",
                     "head", "r_km", "rmoon_km", "src"],
            src_codes=SRC,
            range_zero_utc="1969-07-16T13:32:00Z",
            gmst0_rad=float(F.gmst_rad(0.0)),
            omega_earth=F.OMEGA_EARTH,
        ),
        track=rows,
        long_track=[[round(t, 0), *np.round(y[:3], 0)] for t, y in zip(t_l, Yl)],
        moon=moon,
        sun=sun,
        events=events,
        checks=[[c[0], c[1], round(float(c[2]), 6)] for c in checks],
        burn_params=dict(a0_kms2=a0, pitch0_rad=pb, pitch_rate=pk, yaw0_rad=yb_, yaw_rate=yk,
                         ve_kms=B.VE, vent_accel_kms2=a_vent),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for c in checks:
        print(f"{c[0]:<62} doc={c[1]!s:<10} model={c[2]:.4f}")
    print("rows", len(rows), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "part1.json")
