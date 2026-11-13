"""Figures for parts 3 and 4."""
import json, numpy as np, sympy as sp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.integrate import solve_ivp
from gr import Metric
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'legend.frameon':False})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/'

# ---- Part 3: light cone ----
fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(projection='3d')
h = np.linspace(-1, 1, 30); p = np.linspace(0, 2*np.pi, 60); Hh, P = np.meshgrid(h, p)
ax.plot_surface(Hh*np.cos(P), Hh*np.sin(P), Hh, color=C2, alpha=.18, linewidth=0)
ax.plot([0, 0.25], [0, 0.1], [0, 0.95], color=C1, lw=2.5); ax.text(0.27, 0.1, 0.98, 'timelike\n(a massive body)', color=C1, fontsize=9)
ax.plot([0, 0.7], [0, -0.7], [0, 0.99], color=C2, lw=2.5); ax.text(0.72, -0.72, 1.02, 'null (light)', color=C2, fontsize=9)
ax.plot([0, 1.0], [0, 0.6], [0, 0.25], color=C3, lw=2.5); ax.text(1.02, 0.62, 0.25, 'spacelike', color=C3, fontsize=9)
ax.text(0, 0, 1.15, 'future', ha='center'); ax.text(0, 0, -1.25, 'past', ha='center')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('ct'); ax.view_init(14, -50); ax.set_box_aspect((1, 1, 1))
ax.set_title('The light cone at an event: ds² < 0 inside, = 0 on it, > 0 outside', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-3-metric-and-spacetime/fig_lightcone.png'); plt.close(fig)

# ---- Part 3: twin paradox ----
v = 0.8; T = 10.0                              # Earth time for the round trip (years), c = 1
gam = 1/np.sqrt(1 - v*v)
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.plot([0, 0], [0, T], color=C1, lw=2.5, label='stay-at-home: 10.0 years')
ax.plot([0, v*T/2, 0], [0, T/2, T], color=C2, lw=2.5, label=f'traveller at 0.8c: {T/gam:.1f} years')
for k in range(11): ax.plot(0, k, 'o', color=C1, ms=4)
for k in range(int(T/gam) + 1):
    tt = k*gam; x = v*tt if tt <= T/2 else v*(T - tt)
    ax.plot(x, tt, 's', color=C2, ms=5)
ax.plot([0, 6], [0, 6], color=GR, ls=':', lw=1); ax.plot([0, 6], [10, 4], color=GR, ls=':', lw=1)
ax.set_xlabel('distance from Earth (light-years)'); ax.set_ylabel('time on Earth (years)'); ax.set_xlim(-0.5, 6); ax.set_ylim(0, 10.5)
ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8.5); ax.grid(alpha=.25)
ax.set_title('Proper time is the length of a world line.\nMarkers are one year on each clock.', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-3-metric-and-spacetime/fig_twins.png'); plt.close(fig)

# ---- Part 4: parallel transport round a geodesic triangle (north pole -> equator -> 90 deg along -> pole) ----
def sph(t, p): return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
fig = plt.figure(figsize=(6.5, 5.6)); ax = fig.add_subplot(projection='3d')
u, w = np.mgrid[0:np.pi:40j, 0:2*np.pi:60j]
ax.plot_surface(np.sin(u)*np.cos(w), np.sin(u)*np.sin(w), np.cos(u), color='#dde6f2', alpha=.3, linewidth=0)
legs = [(np.linspace(0.001, np.pi/2, 60), np.zeros(60)), (np.full(60, np.pi/2), np.linspace(0, np.pi/2, 60)), (np.linspace(np.pi/2, 0.001, 60), np.full(60, np.pi/2))]
for th, ph in legs: ax.plot(*sph(th, ph), color='k', lw=1.6)
# the transported vector: along each geodesic leg it keeps a fixed angle to the leg
def arrow(Pt, d, c): ax.quiver(*Pt, *(d*0.28), color=c, lw=2, arrow_length_ratio=.25)
ex = lambda t, p: np.array([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)])     # e_theta
ey = lambda t, p: np.array([-np.sin(p), np.cos(p), 0.0])                              # e_phi (unit)
cols = [C2]*4 + [C3]*4 + [C1]*4
pts = []
for t in np.linspace(0.15, np.pi/2, 4): pts.append((sph(t, 0), ex(t, 0)))                 # leg 1: vector along the meridian (tangent)
for p in np.linspace(0, np.pi/2, 4): pts.append((sph(np.pi/2, p), ex(np.pi/2, p)))       # leg 2: still pointing 'south', perpendicular to the equator
for t in np.linspace(np.pi/2, 0.15, 4): pts.append((sph(t, np.pi/2), ex(t, np.pi/2)))     # leg 3: along the meridian, pointing away from the pole
for (Pt, d), c in zip(pts, cols): arrow(Pt, d, c)
ax.text(0, 0, 1.12, 'start and finish', ha='center', fontsize=9)
ax.set_box_aspect((1, 1, 1)); ax.axis('off'); ax.view_init(25, 35)
ax.set_title('Carry a vector round a triangle with three right angles:\nit comes back turned through 90°, the area enclosed (in units of a²)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-4-connections-and-geodesics/fig_triangle.png'); plt.close(fig)

# ---- Part 4: geodesic equation on the sphere, integrated numerically, against a great circle ----
th, ph = sp.symbols('theta phi'); m = Metric(sp.diag(1, sp.sin(th)**2), [th, ph]); G = m.christoffel()
Gf = sp.lambdify((th, ph), [[[G[a][b][c] for c in range(2)] for b in range(2)] for a in range(2)], 'numpy')
def rhs(s, y):
    x, d = y[:2], y[2:]; g = np.array(Gf(*x), dtype=float)
    return [*d, *(-np.einsum('abc,b,c->a', g, d, d))]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
errs = []
for k, (t0, dth, dph, c) in enumerate([(1.0, 0.3, 1.0, C1), (0.6, -0.5, 1.4, C2), (1.3, 0.8, 0.6, C3)]):
    nrm = np.sqrt(dth**2 + np.sin(t0)**2*dph**2); y0 = [t0, 0.0, dth/nrm, dph/nrm]
    s = np.linspace(0, 2*np.pi, 400)
    sol = solve_ivp(rhs, (0, 2*np.pi), y0, t_eval=s, rtol=1e-11, atol=1e-12)
    P = sph(sol.y[0], sol.y[1])
    # great circle through the start point with the same initial direction
    P0 = sph(t0, 0.0); D0 = (y0[2]*ex(t0, 0.0) + y0[3]*np.sin(t0)*ey(t0, 0.0))
    GCc = np.outer(P0, np.cos(s)) + np.outer(D0, np.sin(s))
    errs.append(float(np.abs(P - GCc).max()))
    ax[0].plot(np.degrees(sol.y[1]), 90 - np.degrees(sol.y[0]), color=c, lw=2)
    ax[1].semilogy(s, np.linalg.norm(P - GCc, axis=0) + 1e-17, color=c, lw=1.5)
ax[0].set_xlabel('longitude φ (degrees)'); ax[0].set_ylabel('latitude (degrees)'); ax[0].set_title('Solutions of the geodesic equation, in latitude and longitude', fontsize=10); ax[0].grid(alpha=.25)
ax[1].set_xlabel('distance along the path (radians)'); ax[1].set_ylabel('distance from the great circle'); ax[1].set_title('All three are great circles, to integration accuracy', fontsize=10); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT+'gr-4-connections-and-geodesics/fig_geodesics.png'); plt.close(fig)
print('max deviation from great circles:', errs)
