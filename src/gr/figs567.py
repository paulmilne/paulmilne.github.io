"""Figures for parts 5-7."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'legend.frameon':False})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/'

# ---- Part 5: geodesic deviation on surfaces of constant curvature K ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
s = np.linspace(0, 3.2, 300)
for K, c, lab in [(1, C2, 'sphere, K = +1/a²: geodesics converge'), (0, GR, 'plane, K = 0: stay parallel'), (-1, C1, 'hyperbolic plane, K = −1/a²: diverge')]:
    xi = np.cos(np.sqrt(K)*s) if K > 0 else (np.ones_like(s) if K == 0 else np.cosh(s))
    ax[0].plot(s, xi, color=c, lw=2, label=lab)
# numerical check on the sphere: two meridians that start parallel at the equator
th1 = np.pi/2 - s; d = 0.01; sep = 2*np.arcsin(np.sin(d/2)*np.sin(np.pi/2 - th1 + np.pi/2) * 1)  # chord on unit sphere, small
sep_num = np.arccos(np.clip(np.cos(th1)**2 + np.sin(th1)**2*np.cos(d), -1, 1))/d
m = s <= np.pi/2
ax[0].plot(s[m][::12], sep_num[m][::12], 'o', color=C2, ms=4, label='measured on a sphere: two meridians,\nwhich meet at the pole, s = πa/2')
ax[0].axhline(0, color='k', lw=.5); ax[0].set_ylim(-1.1, 4); ax[0].set_xlim(0, 3.2); ax[0].set_xlabel('distance along the geodesics, s/a'); ax[0].set_ylabel('separation ξ / ξ₀')
ax[0].set_title('Geodesic deviation: d²ξ/ds² = −K ξ', fontsize=10); ax[0].legend(fontsize=8)
# tidal ring falling radially into a black hole (small ring, fixed r: stretch along r, squeeze across)
a1 = ax[1]; a1.set_aspect('equal'); a1.axis('off')
ang = np.linspace(0, 2*np.pi, 60); tau = [0, 0.5, 0.9, 1.2]
for k, T in enumerate(tau):
    xr = np.cosh(np.sqrt(2)*T)*np.cos(ang); xp = np.cos(T)*np.sin(ang)
    off = k*3.4; a1.plot(xp + off, xr, color=[C1, C3, C2, PU][k], lw=2)
    a1.text(off, -4.6, f'τ = {T}', ha='center', fontsize=9)
a1.annotate('', (-1.6, -3.5), (-1.6, 3.5), arrowprops=dict(arrowstyle='<->', color=GR)); a1.text(-2.1, 0, 'radial direction\n(towards the mass)', rotation=90, va='center', ha='center', fontsize=8.5, color='#555')
a1.set_title('A ring of free particles near a mass: stretched radially,\nsqueezed sideways (τ in units of (r³/M)^½)', fontsize=10)
a1.set_xlim(-2.6, 12); a1.set_ylim(-5, 5)
fig.tight_layout(); fig.savefig(OUT+'gr-5-curvature/fig_deviation.png'); plt.close(fig)

# ---- Part 6: Friedmann solutions ----
fig, ax = plt.subplots(figsize=(6.6, 4.2))
t = np.linspace(1e-3, 3, 400)
ax.plot(t, t**0.5, color=C3, lw=2, label='radiation: a ∝ t^(1/2)')
ax.plot(t, t**(2/3), color=C1, lw=2, label='matter (dust): a ∝ t^(2/3)')
ax.plot(t, np.exp(t - 1), color=C2, lw=2, label='cosmological constant: a ∝ e^(Ht)  (de Sitter)')
Om, OL = 0.315, 0.685
f = lambda tt, a: [np.sqrt(Om/a[0] + OL*a[0]**2)]
sol = solve_ivp(f, (0, 3), [1e-6], dense_output=True, rtol=1e-9, max_step=0.01); tt = np.linspace(1e-3, 3, 400)
t0 = tt[np.argmin(abs(sol.sol(tt)[0] - 1))]
ax.plot(tt/t0, sol.sol(tt)[0], color='k', lw=2, ls='--', label='our universe (matter + Λ), time in units of its age')
ax.set_ylim(0, 4); ax.set_xlim(0, 3); ax.set_xlabel('time'); ax.set_ylabel('scale factor a'); ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.set_title('Solutions of the Friedmann equation', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-6-einstein-equations/fig_friedmann.png'); plt.close(fig)

# ---- Part 7: the charts of 2D de Sitter on its hyperboloid ----
def base(ax):
    tg = np.linspace(-1.5, 1.5, 40); pg = np.linspace(0, 2*np.pi, 80); Tg, Pg = np.meshgrid(tg, pg)
    ax.plot_surface(np.cosh(Tg)*np.cos(Pg), np.cosh(Tg)*np.sin(Pg), np.sinh(Tg), color='#dde6f2', alpha=.18, linewidth=0)
    ax.set_box_aspect((1, 1, 1)); ax.set_xlabel('X₂'); ax.set_ylabel('X₁'); ax.set_zlabel('T'); ax.view_init(16, -35)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_zlim(-2.1, 2.1)
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_subplot(1, 3, 1, projection='3d'); base(ax)
for t0 in np.linspace(-1.4, 1.4, 8):          # global time slices (circles)
    p = np.linspace(0, 2*np.pi, 100); ax.plot(np.cosh(t0)*np.cos(p), np.cosh(t0)*np.sin(p), np.sinh(t0)+0*p, color=C1, lw=1)
ax.set_title('global coordinates: cover everything', fontsize=10)
ax = fig.add_subplot(1, 3, 2, projection='3d'); base(ax)
for t0 in np.linspace(-1.8, 1.2, 8):          # flat slicing: T = sinh t + x^2 e^t/2, X1 = e^t x, X2 = cosh t - x^2 e^t/2
    xx = np.linspace(-4, 4, 200); T = np.sinh(t0) + xx**2*np.exp(t0)/2; X1 = np.exp(t0)*xx; X2 = np.cosh(t0) - xx**2*np.exp(t0)/2
    m = (np.abs(T) < 2.1) & (np.abs(X1) < 2.4) & (np.abs(X2) < 2.4); ax.plot(np.where(m, X2, np.nan), np.where(m, X1, np.nan), np.where(m, T, np.nan), color=C2, lw=1)
for s_ in (1, -1):                             # boundary T + X2 = 0: two null lines X1 = +-ell
    tt = np.linspace(-2.1, 2.1, 50); ax.plot(-tt, s_*np.ones_like(tt), tt, color='k', lw=1.6)
ax.set_title('flat slicing: half (T + X₂ > 0)\nexpands forever, e^(t/ℓ)', fontsize=10)
ax = fig.add_subplot(1, 3, 3, projection='3d'); base(ax)
for r0 in np.linspace(-0.9, 0.9, 7):          # static patch: X1 = r, T = sqrt(1-r^2) sinh t, X2 = sqrt(1-r^2) cosh t
    tt = np.linspace(-2.2, 2.2, 100); w = np.sqrt(1 - r0**2); T = w*np.sinh(tt); X2 = w*np.cosh(tt)
    m = np.abs(T) < 2.1; ax.plot(np.where(m, X2, np.nan), r0 + 0*tt, np.where(m, T, np.nan), color=C3, lw=1)
for s_ in (1, -1):
    tt = np.linspace(0, 2.1, 50); ax.plot(tt, s_*np.ones_like(tt), tt, color='k', lw=1.6); ax.plot(tt, s_*np.ones_like(tt), -tt, color='k', lw=1.6)
ax.set_title('static patch: a quarter (X₂ > |T|)\nobservers at fixed r, bounded by horizons', fontsize=10)
fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.03, wspace=0.0)
fig.savefig(OUT+'gr-7-de-sitter-solved/fig_ds_charts.png'); plt.close(fig)
print('ok')
