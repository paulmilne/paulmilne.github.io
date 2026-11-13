"""Figures for the topology and open sets explainer."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.size':10,'figure.dpi':150,'legend.frameon':False})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/topology-and-open-sets/'

# ---- open and closed sets in the plane ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a in ax: a.set_aspect('equal'); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.5, 1.5); a.axis('off')
a = ax[0]; a.add_patch(Circle((0, 0), 1, fc='#1f5fa822', ec=C1, ls='--', lw=1.6))
p = np.array([0.55, 0.45]); a.plot(*p, 'k.', ms=6); a.add_patch(Circle(p, 0.27, fc='#d9622b33', ec=C2, lw=1.2))
a.text(0, -1.38, 'a smaller ball fits round every point', ha='center', fontsize=8.5, color=C2)
a.set_title('open disc x² + y² < 1\n(edge not included: dashed)', fontsize=10)
a = ax[1]; a.add_patch(Circle((0, 0), 1, fc='#1f5fa822', ec=C1, lw=1.8))
q = np.array([np.cos(0.7), np.sin(0.7)]); a.plot(*q, 'k.', ms=7); a.add_patch(Circle(q, 0.25, fc='none', ec=C2, lw=1.2))
a.text(q[0] + 0.05, q[1] + 0.33, 'every ball round this edge\npoint pokes outside', fontsize=8.5, color=C2, ha='center')
a.set_title('closed disc x² + y² ≤ 1\n(edge included: not open)', fontsize=10)
a = ax[2]; th = np.linspace(0, 2*np.pi, 300)
a.fill_between(np.linspace(-1.3, 1.3, 10), -0.9, 0.9, color='#1f5fa822'); a.plot([-1.3, 1.3, 1.3, -1.3, -1.3], [-0.9, -0.9, 0.9, 0.9, -0.9], color=C1, lw=1.8)
a.plot([-1.3, 1.3], [0.9, 0.9], color='white', lw=2.2); a.plot([-1.3, 1.3], [0.9, 0.9], color=C1, lw=1.6, ls='--')
a.set_title('half-open rectangle: neither open\nnor closed (top edge left out)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'fig_open_closed.png'); plt.close(fig)

# ---- gluing (quotient spaces) ----
def square(a, title, top, bottom, left, right):
    a.set_aspect('equal'); a.axis('off'); a.set_xlim(-0.3, 1.3); a.set_ylim(-0.35, 1.3)
    a.add_patch(Rectangle((0, 0), 1, 1, fc='#dde6f2', ec='none'))
    def edge(p0, p1, spec):
        if spec is None: a.plot(*zip(p0, p1), color=GR, lw=1.2); return
        col, n, rev = spec; q0, q1 = (p1, p0) if rev else (p0, p1)
        a.plot(*zip(p0, p1), color=col, lw=2.4)
        for k in range(n):
            m = np.array(q0) + (np.array(q1) - np.array(q0))*(0.45 + 0.1*k)
            a.add_patch(FancyArrowPatch(m - (np.array(q1) - np.array(q0))*0.05, m + (np.array(q1) - np.array(q0))*0.05, arrowstyle='-|>', mutation_scale=14, color=col))
    edge((0, 1), (1, 1), top); edge((0, 0), (1, 0), bottom); edge((0, 0), (0, 1), left); edge((1, 0), (1, 1), right)
    a.set_title(title, fontsize=10)
fig, ax = plt.subplots(1, 5, figsize=(12, 2.9))
square(ax[0], 'cylinder', None, None, (C1, 1, False), (C1, 1, False))
square(ax[1], 'Möbius strip', None, None, (C1, 1, False), (C1, 1, True))
square(ax[2], 'torus', (C2, 2, False), (C2, 2, False), (C1, 1, False), (C1, 1, False))
square(ax[3], 'Klein bottle', (C2, 2, False), (C2, 2, False), (C1, 1, False), (C1, 1, True))
square(ax[4], 'sphere', (C2, 2, False), (C1, 1, True), (C2, 2, True), (C1, 1, False))
fig.text(0.5, 0.02, 'Glue edges of the same colour so that their arrows match. Gluing creates new spaces from a flat square: quotient topology.', ha='center', fontsize=8.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.05); fig.savefig(OUT+'fig_gluing.png'); plt.close(fig)

# ---- line with two origins ----
fig, ax = plt.subplots(figsize=(7.5, 2.4)); ax.axis('off'); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1, 1)
ax.plot([-3, -0.06], [0, 0], color='k', lw=2); ax.plot([0.06, 3], [0, 0], color='k', lw=2)
ax.plot(0, 0.22, 'o', color=C2, ms=8); ax.plot(0, -0.22, 'o', color=C1, ms=8)
ax.text(0.15, 0.32, 'origin A', color=C2); ax.text(0.15, -0.42, 'origin B', color=C1)
for s, c in [(1, C2), (-1, C1)]:
    ax.plot([-0.7, 0.7], [0.08*s + 0.0, 0.08*s], color=c, lw=6, alpha=.35, solid_capstyle='round')
ax.text(-3, 0.55, 'Every neighbourhood of A meets every neighbourhood of B (the shaded intervals overlap\naway from 0), so a sequence like 1/n converges to both. Not Hausdorff, so not a manifold.', fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT+'fig_two_origins.png'); plt.close(fig)

# ---- homeomorphism: open interval to the line, and sphere minus a point to the plane ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.linspace(-1, 1, 400)[1:-1]
ax[0].plot(x, np.tan(np.pi*x/2), color=C1, lw=2); ax[0].set_ylim(-6, 6); ax[0].axhline(0, color=GR, lw=.6); ax[0].axvline(0, color=GR, lw=.6)
ax[0].axvline(-1, color=GR, ls=':'); ax[0].axvline(1, color=GR, ls=':')
ax[0].set_xlabel('x in the open interval (−1, 1)'); ax[0].set_ylabel('tan(πx/2) in ℝ')
ax[0].set_title('(−1, 1) and the whole line are homeomorphic\nlength is not a topological property', fontsize=10)
for s in ['top', 'right']: ax[0].spines[s].set_visible(False)
a = ax[1]; a.set_aspect('equal'); a.axis('off')
th = np.linspace(0, 2*np.pi, 200); a.plot(np.cos(th), np.sin(th) + 1, color=C1, lw=2); a.axhline(0, color='k', lw=1.2)
N = np.array([0, 2]); a.plot(*N, 'o', color=C2); a.text(0.1, 2.1, 'N (removed)', color=C2, fontsize=9)
for X0 in [-3.3, -1.6, -0.6, 0.5, 1.5, 3.0]:
    X = np.array([X0, 0.0]); d = X - N; c = N - np.array([0, 1])
    sP = -2*np.dot(c, d)/np.dot(d, d)                 # second intersection of the line N->X with the circle
    P = N + sP*d
    a.plot([N[0], X[0]], [N[1], X[1]], color=GR, lw=.8); a.plot(*P, 'k.', ms=6); a.plot(X[0], 0, '.', color=PU, ms=9)
a.set_xlim(-4, 4); a.set_ylim(-0.4, 2.4)
a.set_title('a circle minus one point is homeomorphic to the line\n(stereographic projection, as in Part 1)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'fig_homeomorphisms.png'); plt.close(fig)
print('ok')
