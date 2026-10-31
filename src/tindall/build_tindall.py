"""Data for the post 'Vent, Bent, Descent, Lament' (space/tindall-vent-bent-descent-lament).

1. A crude piston model of the docking-tunnel vent: LM speed change against tunnel pressure and volume.
2. Backward pushes of 1 ft/s added to the Part 3 reconstruction at several times, propagated to PDI
   with the series' force model (src/apollo11), and compared with the Clohessy-Wiltshire formula.
3. Steady backward thrust (water boiler) checks.

    python3 build_tindall.py ../../space/tindall-vent-bent-descent-lament/data/tindall.json \
        ../../space/apollo-11-part-3-descent-and-landing/data/part3.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apollo11"))
import dynamics as D      # noqa: E402
import anchors as A       # noqa: E402

FT = 0.3048e-3            # km per ft
LM_KG, CSM_KG = 15100.0, 28800.0

# --- 1. Tunnel vent ---------------------------------------------------------------
R_AIR, T_AIR, GAMMA = 287.0, 290.0, 1.4
D_TUNNEL = 0.8128         # 32 in


def vent_dv(psi, vol, cd=0.6, gap_max=0.5, dt=1e-4):
    """LM speed change (ft/s): tunnel pressure pushes the two craft apart while air escapes
    (choked) through the ring-shaped gap of the tunnel's circumference."""
    if psi <= 0:
        return 0.0
    area, circ = np.pi * (D_TUNNEL / 2) ** 2, np.pi * D_TUNNEL
    mu = LM_KG * CSM_KG / (LM_KG + CSM_KG)
    cs = np.sqrt(GAMMA * R_AIR * T_AIR)
    choke = (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
    p = psi * 6894.76
    m0 = m = p * vol / (R_AIR * T_AIR)
    x = v = impulse = t = 0.0
    while m > 1e-6 * m0 and x < gap_max and t < 60:
        p = m * R_AIR * T_AIR / vol
        force = p * area
        m -= cd * (m / vol) * cs * choke * circ * x * dt
        v += force / mu * dt
        x += v * dt
        impulse += force * dt
        t += dt
    return impulse / LM_KG / 0.3048


# --- 2. Pushes in the full model -----------------------------------------------------
def lvlh(rm, vm):
    z = -rm / np.linalg.norm(rm)
    y = np.cross(vm, rm); y /= np.linalg.norm(y)
    return np.cross(y, z), y, z


def moonrel(t, y):
    return y[:3] - D.moon_pos(t), y[3:] - D.moon_vel(t)


def main(out_path, part3_path):
    psis = [round(float(p), 2) for p in np.arange(0, 5.01, 0.25)]
    vols = [0.5, 1.0, 2.0]
    vent = {str(v): [round(vent_dv(p, v), 4) for p in psis] for v in vols}

    d = json.load(open(part3_path))
    tr = np.array([r[:4] for r in d["track"]])
    t_u = A.LM_UNDOCK["get"]
    t_doi = A.DOI["tig"] + A.DOI["dur"] / 2
    t_pdi = A.PDI["tig"]
    i = np.searchsorted(tr[:, 0], t_u)
    sl = slice(i - 3, i + 4)
    P = [np.polyfit(tr[sl, 0] - t_u, tr[sl, k], 4) for k in (1, 2, 3)]
    r0 = np.array([np.polyval(p, 0) for p in P]); v0 = np.array([np.polyval(np.polyder(p), 0) for p in P])
    y0 = np.r_[r0 + D.moon_pos(t_u), v0 + D.moon_vel(t_u)]

    def doi(t, y):
        x, _, _ = lvlh(*moonrel(t, y)); return np.r_[y[:3], y[3:] - x * A.DOI["dv_fps"] * FT]

    def push(dv):
        def f(t, y):
            x, yy, z = lvlh(*moonrel(t, y))
            return np.r_[y[:3], y[3:] + (dv[0] * x + dv[1] * yy + dv[2] * z) * FT]
        return f

    def run(events, thrust=None):
        rhs = D.make_rhs(thrust)
        t, y = t_u, y0.copy()
        for te, fn in sorted(events + [(t_doi, doi)], key=lambda e: e[0]):
            if te > t:
                y = D.integrate(rhs, t, y, [te])[-1]; t = te
            y = fn(t, y)
        return D.integrate(rhs, t, y, [t_pdi])[-1]

    nom = run([])
    X, _, Z = lvlh(*moonrel(t_pdi, nom))
    ahead = lambda p: float((p[:3] - nom[:3]) @ X / FT)
    n = float(np.sqrt(D.MU_MOON / 1849.0 ** 3))
    cw = lambda dv, tau: dv * (4 / n * np.sin(n * tau) - 3 * tau)

    full = []
    for tp in [t_u, t_u + 1800, t_u + 3600, t_doi - 60, t_doi + 60, t_doi + 1800, t_pdi - 1200]:
        a = ahead(run([(tp, push([-1, 0, 0]))]))
        full.append(dict(get=tp, min_before=round((t_pdi - tp) / 60, 1), full_ft=round(a), cw_ft=round(float(cw(-1, t_pdi - tp)))))
        print(full[-1])

    thrust_full = []
    for lbf, start, t1 in [(0.16, "undocking", t_u), (0.16, "DOI", t_doi), (0.25, "undocking", t_u)]:
        acc = lbf * 4.44822 / LM_KG / 1000.0
        def th(t, r, v, acc=acc, t1=t1):
            if t < t1:
                return np.zeros(3)
            x, _, _ = lvlh(r - D.moon_pos(t), v - D.moon_vel(t)); return -acc * x
        thrust_full.append(dict(lbf=lbf, start=start, full_ft=round(ahead(run([], th)))))
        print(thrust_full[-1])

    out = dict(vent=dict(psi=psis, vols=vols, dv_fps=vent, lm_kg=LM_KG, csm_kg=CSM_KG, tunnel_diam_m=D_TUNNEL),
               full=full, thrust_full=thrust_full,
               events=dict(undock=t_u, sep=A.CSM_SEP["tig"], doi=t_doi, pdi=t_pdi, touchdown=A.TOUCHDOWN_GET),
               n=n, r_ft=1849.0e3 / 0.3048, v_pdi=5560)
    json.dump(out, open(out_path, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
