"""Figures for the metrics explainer."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams.update({'font.size':10,'figure.dpi':150,'legend.frameon':False})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/metrics/'

# ---- unit balls of different norms ----
fig, ax = plt.subplots(figsize=(4.6, 4.4))
t = np.linspace(0, 2*np.pi, 400)
for p, c, lab in [(1, C2, 'p = 1 (taxicab)'), (2, C1, 'p = 2 (Euclidean)'), (4, C3, 'p = 4'), (np.inf, PU, 'p = ∞ (maximum)')]:
    if np.isinf(p): xs, ys = np.array([1, -1, -1, 1, 1]), np.array([1, 1, -1, -1, 1])
    else:
        xs = np.sign(np.cos(t))*np.abs(np.cos(t))**(2/p); ys = np.sign(np.sin(t))*np.abs(np.sin(t))**(2/p)
    ax.plot(xs, ys, color=c, lw=2, label=lab)
ax.set_aspect('equal'); ax.axhline(0, color=GR, lw=.6); ax.axvline(0, color=GR, lw=.6); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.6)
ax.legend(fontsize=8, loc='upper center', ncol=2); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.set_title('"Circles" of radius 1 for the p-norms\n‖v‖ = (|v₁|ᵖ + |v₂|ᵖ)^(1/p)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'fig_unit_balls.png'); plt.close(fig)

# ---- Tissot indicatrices: the same small circles on the sphere, drawn in three charts ----
eps = 0.09
def tissot(ax, g, pts, color):
    for (q1, q2) in pts:
        G = np.array(g(q1, q2), float); w, V = np.linalg.eigh(G)
        ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        ax.add_patch(Ellipse((q1, q2), 2*eps/np.sqrt(w[0]), 2*eps/np.sqrt(w[1]), angle=ang, fc=color+'55', ec=color, lw=.8))
fig, ax = plt.subplots(1, 3, figsize=(12, 4.8))
lat = np.radians(np.arange(-75, 76, 15)); lon = np.radians(np.arange(-150, 151, 30))
a0 = ax[0]; tissot(a0, lambda lo, la: [[np.cos(la)**2, 0], [0, 1]], [(lo, la) for lo in lon for la in lat], C1)
a0.set_xlim(-np.pi, np.pi); a0.set_ylim(-np.pi/2, np.pi/2); a0.set_aspect('equal'); a0.set_title('latitude–longitude chart\ng = diag(cos²φ, 1): circles stretched east–west', fontsize=10)
a0.set_xlabel('longitude λ'); a0.set_ylabel('latitude φ')
a1 = ax[1]; ys = np.arcsinh(np.tan(lat)); tissot(a1, lambda X, Y: [[1/np.cosh(Y)**2, 0], [0, 1/np.cosh(Y)**2]], [(lo, yy) for lo in lon for yy in ys], C2)
a1.set_xlim(-np.pi, np.pi); a1.set_ylim(-3.2, 3.2); a1.set_aspect('equal'); a1.set_title('Mercator chart\ng = sech²y · (dx² + dy²): still circles, but bigger', fontsize=10)
a1.set_xlabel('x = λ'); a1.set_ylabel('y = arsinh(tan φ)')
a2 = ax[2]; pts = []
for la in np.radians(np.arange(-60, 76, 15)):
    for lo in lon:
        rr = np.cos(la)/(1 - np.sin(la)) if la < np.radians(80) else 0   # stereographic from the south pole... use north pole projection of colatitude
        pts.append((rr*np.cos(lo), rr*np.sin(lo)))
tissot(a2, lambda U, V: [[4/(1 + U*U + V*V)**2, 0], [0, 4/(1 + U*U + V*V)**2]], pts, C3)
a2.set_xlim(-4.2, 4.2); a2.set_ylim(-4.2, 4.2); a2.set_aspect('equal'); a2.set_title('stereographic chart\ng = 4(du² + dv²)/(1 + u² + v²)²', fontsize=10)
a2.set_xlabel('u'); a2.set_ylabel('v')
for a_ in ax:
    for s in ['top', 'right']: a_.spines[s].set_visible(False)
fig.text(0.5, 0.01, 'Each shape is the same small circle on the sphere, radius 0.09a, drawn at a grid of points in each chart. The metric decides the shape.', ha='center', fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(OUT+'fig_tissot.png'); plt.close(fig)

# ---- Poincare disc: equal hyperbolic circles and geodesics ----
fig, ax = plt.subplots(figsize=(5, 5)); ax.set_aspect('equal'); ax.axis('off')
ax.add_patch(plt.Circle((0, 0), 1, fc='#dde6f2', ec='k', lw=1.2))
R = 0.35                                            # hyperbolic radius of each circle
for d in [0, 0.8, 1.6, 2.4, 3.2]:                   # hyperbolic distance of centre from origin
    for ang in np.radians([0, 120, 240]) if d > 0 else [0]:
        # a hyperbolic circle is a Euclidean circle; its Euclidean extent along the radius is between tanh((d-R)/2), tanh((d+R)/2)
        r1, r2 = np.tanh((d - R)/2), np.tanh((d + R)/2)
        c = (r1 + r2)/2; rad = (r2 - r1)/2
        ax.add_patch(plt.Circle((c*np.cos(ang), c*np.sin(ang)), rad, fc=C2+'66', ec=C2, lw=.8))
for ang in np.radians([30, 90, 150, 210, 270, 330]): ax.plot([-np.cos(ang), np.cos(ang)], [-np.sin(ang), np.sin(ang)], color=GR, lw=.6)
for th0, w in [(np.radians(60), np.radians(40)), (np.radians(200), np.radians(55))]:   # geodesic arcs orthogonal to the boundary
    cx, cy = np.cos(th0)/np.cos(w), np.sin(th0)/np.cos(w); rr = np.tan(w)
    tt = np.linspace(0, 2*np.pi, 600); X, Y = cx + rr*np.cos(tt), cy + rr*np.sin(tt); m = X**2 + Y**2 < 1
    ax.plot(np.where(m, X, np.nan), np.where(m, Y, np.nan), color=C1, lw=1.8)
ax.set_title('The Poincaré disc, g = 4(dx² + dy²)/(1 − x² − y²)².\nAll orange circles have the same radius; blue arcs are geodesics.', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'fig_poincare.png'); plt.close(fig)
print('ok')
