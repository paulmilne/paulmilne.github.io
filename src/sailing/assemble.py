"""Fill the numbers in the sailing post from the model's data file."""
import json, re, sys
import numpy as np

src, data_path, out = sys.argv[1:4]
d = json.load(open(data_path))
C, M = d["meta"]["columns"], d["meta"]
G = [dict(zip(C, g)) for g in d["grid"]]
at = lambda tws, twa: next(g for g in G if g["tws"] == tws and g["twa"] == twa)
vmg = {v[0]: v for v in d["vmg"]}
f = lambda x, n=1: f"{x:,.{n}f}"

ex = at(12, vmg[12][1])
rh = M["mass"] * 9.81 * M["gm"] * np.sin(np.radians(25))
rc = M["crew"] * 9.81 * M["crew_arm"] * np.cos(np.radians(25))
up = {s: at(s, vmg[s][1]) for s in M["tws"]}
dep = min(s for s in M["tws"] if up[s]["flat"] < 0.97)
reef = min(s for s in M["tws"] if up[s]["reef"] < 0.99)
cert = d["cert"]
spread = np.median(np.array(cert["p90"]) - np.array(cert["p10"]))
# upwind and downwind VMG comparison sentence
diffs_u = [vmg[s][2] - cert["beat_vmg"][j] for j, s in enumerate(cert["speeds"])]
diffs_d = [vmg[s][4] - cert["run_vmg"][j] for j, s in enumerate(cert["speeds"])]
ang_u = [vmg[s][1] - cert["beat_angle"][j] for j, s in enumerate(cert["speeds"])]
worst_u = max(range(len(diffs_u)), key=lambda j: abs(diffs_u[j]))
light = [j for j, s_ in enumerate(cert["speeds"]) if s_ <= 10]
heavy = [j for j, s_ in enumerate(cert["speeds"]) if s_ >= 12]
sp16 = cert["speeds"].index(16); sp20 = cert["speeds"].index(20)
vmgtext = (f"Upwind, the model's best angles are within {f(max(abs(x) for x in ang_u), 0)}° of the certificates' (on the model's 2° grid). "
           f"Its VMG is within {f(max(abs(diffs_u[j]) for j in heavy), 2)} knots from 12 knots of wind upwards, but up to "
           f"{f(max(abs(diffs_u[j]) for j in light), 2)} knots low in lighter air, where the real boat's sails, trimmed fuller and "
           "with less twist than the model's two-parameter trim allows, make more of a little wind. "
           f"Downwind, without being fitted, the VMG is within {f(max(abs(x) for x in diffs_d), 2)} knots at every wind speed. The best downwind angles differ more, "
           "because near the optimum VMG hardly changes with angle, so small details decide where the optimum sits. In 16 and 20 knots the certificates' optimum "
           f"is almost dead downwind ({f(cert['run_angle'][sp16], 0)}° and {f(cert['run_angle'][sp20], 0)}°), while the model's is {f(vmg[16][3], 0)}° and {f(vmg[20][3], 0)}°. "
           "Why the best way downwind is usually not straight downwind is the subject of the next post.")
vals = dict(
    W1=f(12, 0), A1=f(vmg[12][1], 0), V1=f(ex["v"], 1), AWS1=f(ex["aws"], 1), AWA1=f(ex["awa"], 0),
    LIFT1=f(ex["lift"], 0), DRAG1=f(ex["drag"], 0), DRIVE1=f(ex["drive"], 0), SIDE1=f(ex["side"], 0),
    RATIO1=f(ex["side"] / ex["drive"], 1), LEE1=f(ex["leeway"], 1), ZCE1=f(ex["zce"], 1),
    HULLSPEED=f(1.34 * np.sqrt(M["lsm"] / 0.3048), 1), VMAX=f(max(g["v"] for g in G), 1),
    LSM=f(M["lsm"], 2), GM=f(M["gm"], 2), MASS=f(M["mass"] / 1000, 1), CREW=f(M["crew"], 0),
    RM25=f((rh + rc) / 1000, 0), CREWPCT=f(100 * rc / (rh + rc), 0), DEPOWER_TWS=f(dep, 0), REEF_TWS=f(reef, 0),
    AWAMIN=f(min(up[s]["awa"] for s in M["tws"] if s >= 8), 0), RMS=f(M["rms"], 2), SPREAD=f(spread, 2), VMGTEXT=vmgtext)
s = open(src).read()
for k, v in vals.items():
    s = s.replace(f"%%{k}%%", v)
left = re.findall(r"%%\w+%%", s)
assert not left, left
open(out, "w").write(s)
for k, v in vals.items():
    if k != "VMGTEXT":
        print(k, v)
print(vmgtext)
