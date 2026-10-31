"""Build the data for the sailing post: a dense polar of the J/109 with every force,
the ORC certificate polars it is checked against, and the tables behind the plots.

    python3 build_sailing.py ORC_ALL2024.json out.json

ORC_ALL2024.json: the 2024 file of github.com/jieter/orc-data (ORC certificate data).
"""
import ast
import json
import multiprocessing as mp
import sys
import time

import numpy as np

import vpp as V

GM, LSM = None, None
TWS = list(range(4, 25, 2))
TWA = list(range(20, 181, 2))


def certificates(path):
    D = ast.literal_eval(open(path).read())
    rows = [d for d in D if (d["boat"].get("type") or "").upper().replace(" ", "").replace("/", "").replace("-", "") == "J109"]
    for d in rows:
        d["vpp"] = {str(k): v for k, v in d["vpp"].items()}
    return rows


def _point(args):
    gm, lsm, tws, twa = args
    r = V.solve_point(V.j109(gm=gm, lsm=lsm), tws, twa)
    return {k: (round(float(v), 4) if not isinstance(v, bool) else v) for k, v in r.items()}


def fit(rows, pool):
    from scipy.optimize import least_squares
    ang, sp = rows[0]["vpp"]["angles"], rows[0]["vpp"]["speeds"]
    med = np.median([[d["vpp"][str(a)] for a in ang] for d in rows], 0)

    bv = np.median([d["vpp"]["beat_vmg"] for d in rows], 0)
    up_angles = list(range(30, 52, 2))

    def res(p):
        jobs = [(p[0], p[1], s, a) for a in ang for s in sp] + [(p[0], p[1], s, a) for s in sp for a in up_angles]
        out = pool.map(_point, jobs)
        n = len(ang) * len(sp)
        g = np.array([o["v"] for o in out[:n]]).reshape(len(ang), len(sp)) - med
        u = np.array([o["v"] * np.cos(np.radians(o["twa"])) for o in out[n:]]).reshape(len(sp), len(up_angles))
        r_vmg = u.max(1) - bv
        print("  fit", np.round(p, 4), "grid rms %.3f, beat VMG rms %.3f" % (np.sqrt((g**2).mean()), np.sqrt((r_vmg**2).mean())), flush=True)
        return np.r_[g.ravel(), r_vmg]
    r = least_squares(res, [0.95, 10.1], diff_step=[0.02, 0.005], bounds=([0.4, 9.3], [2.5, 11.0]), max_nfev=20)
    r.fun = r.fun[:len(ang) * len(sp)]
    return r.x, r.fun.reshape(len(ang), len(sp)), med


def main(cert_path, out_path):
    t0 = time.time()
    rows = certificates(cert_path)
    ang, sp = rows[0]["vpp"]["angles"], rows[0]["vpp"]["speeds"]
    with mp.Pool(3) as pool:
        (gm, lsm), resid, med = fit(rows, pool)
        print("fit GM %.3f m, sailing length %.3f m, rms %.3f kn" % (gm, lsm, np.sqrt((resid**2).mean())), flush=True)
        grid = pool.map(_point, [(gm, lsm, s, a) for s in TWS for a in TWA])
    boat = V.j109(gm=gm, lsm=lsm)

    keys = ["v", "heel", "leeway", "flat", "reef", "kite", "awa", "aws", "lift", "drag", "drive", "side", "hm", "zce",
            "rr", "rv", "rapp", "ri", "rtot", "rm_hull", "rm_crew", "rm_keel", "cl", "cd", "area", "fn"]
    table = [[g["tws"], g["twa"]] + [int(g[k]) if k == "kite" else g[k] for k in keys] for g in grid]

    # Optimum VMG up and down, from the dense polar.
    vmg = []
    for s in TWS:
        pts = [g for g in grid if g["tws"] == s]
        up = max(pts, key=lambda g: g["v"] * np.cos(np.radians(g["twa"])))
        dn = max(pts, key=lambda g: -g["v"] * np.cos(np.radians(g["twa"])))
        vmg.append([s, up["twa"], round(up["v"] * np.cos(np.radians(up["twa"])), 3),
                    dn["twa"], round(-dn["v"] * np.cos(np.radians(dn["twa"])), 3)])

    P = np.array([[d["vpp"][str(a)] for a in ang] for d in rows])
    cert = dict(n=len(rows), angles=ang, speeds=sp, median=np.round(med, 3).tolist(),
                p10=np.round(np.percentile(P, 10, 0), 3).tolist(), p90=np.round(np.percentile(P, 90, 0), 3).tolist(),
                lo=np.round(P.min(0), 3).tolist(), hi=np.round(P.max(0), 3).tolist(),
                beat_angle=np.round(np.median([d["vpp"]["beat_angle"] for d in rows], 0), 2).tolist(),
                beat_vmg=np.round(np.median([d["vpp"]["beat_vmg"] for d in rows], 0), 3).tolist(),
                run_angle=np.round(np.median([d["vpp"]["run_angle"] for d in rows], 0), 2).tolist(),
                run_vmg=np.round(np.median([d["vpp"]["run_vmg"] for d in rows], 0), 3).tolist(),
                countries=sorted(set(d["country"] for d in rows)),
                sizes={k: round(float(np.median([d["boat"]["sizes"][k] for d in rows])), 3)
                       for k in ("loa", "beam", "draft", "displacement", "main", "genoa", "spinnaker_asym", "crew", "wetted_surface")})

    # Sail coefficients (published points and the curves used).
    sails = {}
    for s in (V.MAIN, V.JIB, V.KITE):
        sails[s.name] = dict(awa=s.awa.tolist(), cl=s.cl_pts.tolist(), cd=s.cd_pts.tolist(),
                             curve=[[float(a), round(float(l), 4), round(float(d), 4)]
                                    for a, l, d in zip(V.Coeff.GRID[::4], s.cl[::4], s.cd[::4])])

    # Upright resistance against speed, and its parts.
    res = []
    for kn in np.arange(0.5, 12.01, 0.25):
        h = V.hydro(boat, kn * V.KN, 0.0, 0.0)
        res.append([round(kn, 2), round(h["fn"], 4), round(h["rr"], 1), round(h["rv"], 1), round(h["app"], 1)])
    # Righting moment against heel (no side force).
    rm = []
    for phi in range(0, 61, 2):
        rh, rc, _ = V.righting(boat, phi, 0.0)
        rm.append([phi, round(rh, 1), round(rc, 1)])

    out = dict(
        meta=dict(boat=boat.name, gm=gm, lsm=lsm, lwl=boat.lwl, bwl=boat.bwl, tc=boat.tc, beam=boat.beam,
                  tmax=boat.tmax, mass=boat.mass, crew=boat.crew, crew_arm=boat.crew_arm, wss=boat.wss,
                  I=boat.I, J=boat.J, P=boat.P, E=boat.E, hbi=boat.hbi, bad=boat.bad,
                  main=boat.main_area, jib=boat.jib_area, kite=boat.kite_area,
                  keel=[boat.keel.root, boat.keel.tip, boat.keel.span], rudder=[boat.rudder.root, boat.rudder.tip, boat.rudder.span],
                  tws=TWS, twa=TWA, columns=["tws", "twa"] + keys,
                  rms=float(np.sqrt((resid ** 2).mean()))),
        grid=table, vmg=vmg, cert=cert, resid=np.round(resid, 3).tolist(),
        sails=sails, resistance=res, righting=rm)
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print("points", len(grid), "->", out_path, "%.0f s" % (time.time() - t0))
    for v in vmg:
        print("vmg", v)
    print("cert beat", cert["beat_angle"], cert["beat_vmg"])
    print("cert run ", cert["run_angle"], cert["run_vmg"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
