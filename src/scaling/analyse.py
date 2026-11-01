"""Fit the scaling study: iso-FLOP parabolas, frontier exponent, Chinchilla-form surface, hold-out test."""
import json, glob, sys
import numpy as np
from scipy.optimize import minimize

MINSTEP = 1000
runs = []
for f in sorted(glob.glob("runs/C*.json")):
    r = json.load(open(f)); name = f.split("_", 1)[1][:-5]; C = float(f.split("/")[1].split("_")[0][1:])
    per = 6 * r["N"] + 6 * r["L"] * r["ctx"] * r["d"]
    runs.append(dict(name=name, C=C, d=r["d"], L=r["L"], N=r["N"], tokens=r["tokens"], per_token=per, flops=per * r["tokens"],
                     steps=r["steps"], used=r["steps"] >= MINSTEP, final_val=r["final_val"], seconds=r["seconds"], log=[dict(tokens=p["tokens"], val=p["val"]) for p in r["log"]]))
print(len(runs), "runs")
allruns = runs; runs = [r for r in allruns if r["used"]]; print(len(runs), "used")

# iso-FLOP parabolas
iso = []
for C in sorted(set(r["C"] for r in runs)):
    rs = [r for r in runs if r["C"] == C]
    x = np.log10([r["N"] for r in rs]); y = np.array([r["final_val"] for r in rs])
    p = np.polyfit(x, y, 2)
    xo = -p[1] / (2 * p[0]) if p[0] > 0 else x[np.argmin(y)]
    xo = float(np.clip(xo, x.min() - 0.2, x.max() + 0.2))
    Nopt = 10 ** xo; Lmin = float(np.polyval(p, xo))
    per = np.interp(xo, x, [r["per_token"] for r in rs])
    iso.append(dict(C=C, p=p.tolist(), Nopt=Nopt, Lmin=Lmin, Nmin=float(10 ** x.min()), Nmax=float(10 ** x.max()), ratio=float(C / per / Nopt)))
    print("iso", C, "Nopt %.3g" % Nopt, "Lmin %.4f" % Lmin, "ratio %.1f" % (C / per / Nopt))
lc = np.log10([c["C"] for c in iso]); ln = np.log10([c["Nopt"] for c in iso])
a_iso = float(np.polyfit(lc, ln, 1)[0]); print("a_iso", a_iso)

def fit_surface(rs):
    N = np.array([r["N"] for r in rs], float); D = np.array([r["tokens"] for r in rs], float); L = np.array([r["final_val"] for r in rs])
    def obj(t):
        e, a, b, al, be = t
        pred = np.logaddexp.reduce([np.full_like(N, e), a - al * np.log(N), b - be * np.log(D)], axis=0)
        r = pred - np.log(L); d = 1e-3
        return np.sum(np.where(np.abs(r) < d, 0.5 * r ** 2, d * (np.abs(r) - 0.5 * d)))
    best = None
    for e0 in (-1, -0.5, 0, 0.5):
        for a0 in (0, 5, 10):
            for b0 in (0, 5, 10):
                for al0 in (0.2, 0.5, 1.0):
                    for be0 in (0.2, 0.5, 1.0):
                        res = minimize(obj, [e0, a0, b0, al0, be0], method="L-BFGS-B")
                        if best is None or res.fun < best.fun: best = res
    e, a, b, al, be = best.x
    f = dict(E=float(np.exp(e)), A=float(np.exp(a)), B=float(np.exp(b)), alpha=float(al), beta=float(be))
    pred = f["E"] + f["A"] / N ** f["alpha"] + f["B"] / D ** f["beta"]
    f["rmse"] = float(np.sqrt(np.mean((pred - L) ** 2)))
    return f

ours = fit_surface(runs); print("fit", ours)
Cmax = max(r["C"] for r in runs)
small = fit_surface([r for r in runs if r["C"] < Cmax]); print("holdout fit", small)
rows = []
for r in runs:
    if r["C"] == Cmax:
        pr = small["E"] + small["A"] / r["N"] ** small["alpha"] + small["B"] / r["tokens"] ** small["beta"]
        rows.append(dict(name=r["name"], N=r["N"], D=r["tokens"], pred=float(pr), act=r["final_val"]))
mae = float(np.mean([abs(x["pred"] - x["act"]) for x in rows])); mae_rel = float(np.mean([abs(x["pred"] - x["act"]) / x["act"] for x in rows]))
print("holdout mae", mae, mae_rel)

out = dict(runs=allruns, min_steps=MINSTEP, n_used=len(runs), isoflop=iso, frontier=dict(a_iso=a_iso), fits=dict(ours=ours,
           chinchilla=dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
           epoch=dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)),
           holdout=dict(fit=small, rows=rows, mae=mae, mae_rel=mae_rel),
           best_final=min(r["final_val"] for r in runs if r["C"] == Cmax),
           total_cpu_hours=sum(r["seconds"] for r in allruns) / 3600)
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
