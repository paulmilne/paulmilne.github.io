"""Raster and order parameter from a fireflies_log.py CSV, like the simulated figure in the post.

    python3 tools/fireflies_plot.py flashes_20261112_101500.csv [-o raster.png]
"""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("-o", "--out"); a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
t = np.array([float(r["t_s"]) for r in rows]); nodes = [r["node"] for r in rows]
ids = sorted(set(nodes), key=lambda s: int(s)); row = {n: i for i, n in enumerate(ids)}
y = np.array([row[n] for n in nodes])

# phase of each node from its own last two flashes; order parameter on a 10 ms grid
grid = np.arange(t.min(), t.max(), 0.01); R = np.full(len(grid), np.nan)
ft = {i: t[y == i] for i in range(len(ids))}
for k, tg in enumerate(grid):
    ph = []
    for i, ts in ft.items():
        j = np.searchsorted(ts, tg) - 1
        if j < 1 or j + 1 >= len(ts): continue
        ph.append((tg - ts[j]) / (ts[j + 1] - ts[j]))
    if len(ph) == len(ids): R[k] = abs(np.exp(2j * np.pi * np.array(ph)).mean())

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
ax[0].plot(t, y, "|", color="#b08d3c", ms=7, mew=1.6); ax[0].set_yticks(range(len(ids))); ax[0].set_yticklabels(ids, fontsize=7)
ax[0].set_ylabel("node")
ax[1].plot(grid, R, color="#1f5fa8"); ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("order parameter r"); ax[1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(a.out or a.csv.replace(".csv", ".png"), dpi=150)

# precision of the locked group: spread of each group flash once all nodes take part
groups, cur = [], [0]
for k in range(1, len(t)):
    if t[k] - t[cur[-1]] > 0.15: groups.append(cur); cur = []
    cur.append(k)
groups.append(cur)
full = [t[g].max() - t[g].min() for g in groups if len({nodes[k] for k in g}) == len(ids)]
if full: print(f"{len(full)} group flashes with all {len(ids)} nodes; median spread {np.median(full)*1000:.1f} ms")
print("saved", a.out or a.csv.replace(".csv", ".png"))
