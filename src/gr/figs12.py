"""Figures for parts 1 and 2 of the general relativity series."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':False,'legend.frameon':False})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/'

# ---- Part 1: two stereographic charts of the sphere ----
fig = plt.figure(figsize=(11, 4.2))
ax = fig.add_subplot(1, 3, 2, projection='3d')
u, v = np.mgrid[0:np.pi:40j, 0:2*np.pi:60j]
ax.plot_surface(np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u), color='#dde6f2', alpha=.35, linewidth=0)
for lat in np.radians([-60, -30, 0, 30, 60]):
    p = np.linspace(0, 2*np.pi, 100); ax.plot(np.cos(lat)*np.cos(p), np.cos(lat)*np.sin(p), np.sin(lat)+0*p, color=GR, lw=.6)
for lon in np.radians(range(0, 180, 30)):
    q = np.linspace(0, 2*np.pi, 100); ax.plot(np.cos(q)*np.cos(lon), np.cos(q)*np.sin(lon), np.sin(q), color=GR, lw=.6)
# a curve on the sphere, coloured by which chart it is in
s = np.linspace(0, 1, 400); th = 0.25 + 2.6*s; phc = 0.6 + 4*s
X, Y, Z = np.sin(th)*np.cos(phc), np.sin(th)*np.sin(phc), np.cos(th)
ax.plot(X, Y, Z, color=PU, lw=2.4)
ax.scatter([0], [0], [1], color=C2, s=30); ax.scatter([0], [0], [-1], color=C1, s=30)
ax.text(0, 0, 1.18, 'N', color=C2, ha='center'); ax.text(0, 0, -1.32, 'S', color=C1, ha='center')
ax.set_box_aspect((1, 1, 1)); ax.axis('off'); ax.view_init(18, -55); ax.set_title('the sphere $S^2$', fontsize=10)
def plane(axp, pole, col, title):
    # stereographic projection from 'pole' (+1 north, -1 south) onto the equatorial plane
    for lat in np.radians([-60, -30, 0, 30, 60]):
        p = np.linspace(0, 2*np.pi, 200); x, y, zz = np.cos(lat)*np.cos(p), np.cos(lat)*np.sin(p), np.sin(lat)+0*p
        axp.plot(x/(1-pole*zz), y/(1-pole*zz), color=GR, lw=.6)
    for lon in np.radians(range(0, 360, 30)):
        q = np.linspace(-np.pi/2+0.02, np.pi/2-0.02, 200); x, y, zz = np.cos(q)*np.cos(lon), np.cos(q)*np.sin(lon), np.sin(q)
        m = (1 - pole*zz) > 0.08; axp.plot((x/(1-pole*zz))[m], (y/(1-pole*zz))[m], color=GR, lw=.6)
    U, V = X/(1-pole*Z), Y/(1-pole*Z); ok = np.hypot(U, V) < 6
    axp.plot(np.where(ok, U, np.nan), np.where(ok, V, np.nan), color=PU, lw=2.2)
    axp.plot(0, 0, 'o', color=C1 if pole > 0 else C2, ms=5)
    axp.set_xlim(-4, 4); axp.set_ylim(-4, 4); axp.set_aspect('equal'); axp.set_xticks([]); axp.set_yticks([])
    for sp_ in axp.spines.values(): sp_.set_visible(True); sp_.set_color(col); sp_.set_linewidth(1.6)
    axp.set_title(title, fontsize=10, color=col)
plane(fig.add_subplot(1, 3, 1), 1, C2, 'chart 1: project from N\n(covers all but N)')
plane(fig.add_subplot(1, 3, 3), -1, C1, 'chart 2: project from S\n(covers all but S)')
fig.text(0.5, 0.03, 'The purple curve, drawn in each chart. Where the charts overlap, the transition map between them is (u, v) ↦ (u, v)/(u² + v²), which is smooth.', ha='center', fontsize=8.5)
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.1, wspace=0.05)
fig.savefig(OUT+'gr-1-manifolds/fig_charts.png'); plt.close(fig)

# ---- Part 1: tangent space at a point ----
fig = plt.figure(figsize=(6.5, 4.6)); ax = fig.add_subplot(projection='3d')
ax.plot_surface(np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u), color='#dde6f2', alpha=.4, linewidth=0)
t0, p0 = np.radians(50), np.radians(20); P = np.array([np.sin(t0)*np.cos(p0), np.sin(t0)*np.sin(p0), np.cos(t0)])
et = np.array([np.cos(t0)*np.cos(p0), np.cos(t0)*np.sin(p0), -np.sin(t0)]); ep = np.array([-np.sin(p0), np.cos(p0), 0])
g1, g2 = np.mgrid[-0.7:0.7:2j, -0.7:0.7:2j]
TP = P[:, None, None] + et[:, None, None]*g1 + ep[:, None, None]*g2
ax.plot_surface(*TP, color=C3, alpha=.25)
for d, c, lab in [(et*0.55, C2, r'$\partial_\theta$'), (ep*0.55*np.sin(t0)/np.sin(t0), C1, r'$\partial_\phi$')]:
    ax.quiver(*P, *d, color=c, lw=2, arrow_length_ratio=.15); ax.text(*(P + d*1.25), lab, color=c, fontsize=12)
cs = np.linspace(-0.8, 0.8, 100); curve = np.array([np.sin(t0 - 0.6*cs)*np.cos(p0 + cs), np.sin(t0 - 0.6*cs)*np.sin(p0 + cs), np.cos(t0 - 0.6*cs)])
ax.plot(*curve, color=PU, lw=2); vel = 0.6*et*(-0.6) + ep*np.sin(t0); vel = vel/np.linalg.norm(vel)*0.6
ax.quiver(*P, *vel, color=PU, lw=2, arrow_length_ratio=.15); ax.text(*(P + vel*1.3 + np.array([0, 0, .08])), r"$\gamma'(0)$", color=PU, fontsize=12)
ax.scatter(*P, color='k', s=20); ax.text(*(P - 0.12*et), 'p', fontsize=11)
ax.set_box_aspect((1, 1, 1)); ax.axis('off'); ax.view_init(22, 10)
ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.4, 1.0); ax.set_zlim(0.0, 1.4)
ax.set_title('The tangent space $T_pS^2$: the plane of all velocities at p', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-1-manifolds/fig_tangent.png'); plt.close(fig)

# ---- Part 2: vectors and covectors in polar coordinates ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
for a_ in ax:
    a_.set_aspect('equal'); a_.set_xlim(-3.2, 3.2); a_.set_ylim(-3.2, 3.2); a_.set_xticks([]); a_.set_yticks([])
    for sp_ in a_.spines.values(): sp_.set_visible(False)
for rr in [1, 2, 3]:
    p = np.linspace(0, 2*np.pi, 200)
    for a_ in ax: a_.plot(rr*np.cos(p), rr*np.sin(p), color=GR, lw=.5)
for an in np.radians(range(0, 360, 30)):
    for a_ in ax: a_.plot([0, 3.2*np.cos(an)], [0, 3.2*np.sin(an)], color=GR, lw=.5)
for rr in [1, 2, 3]:
    for an in np.radians([20, 140, 260]):
        x0, y0 = rr*np.cos(an), rr*np.sin(an)
        ax[0].annotate('', (x0 + 0.5*np.cos(an), y0 + 0.5*np.sin(an)), (x0, y0), arrowprops=dict(arrowstyle='->', color=C2, lw=1.6))
        ax[0].annotate('', (x0 - 0.5*rr*np.sin(an)*0.5, y0 + 0.5*rr*np.cos(an)*0.5), (x0, y0), arrowprops=dict(arrowstyle='->', color=C1, lw=1.6))
ax[0].set_title(r'basis vectors: $\partial_r$ (orange), $\partial_\phi$ (blue)' '\n' r'$\partial_\phi$ grows with r: $|\partial_\phi| = r$', fontsize=10)
# covector d(phi): level sets phi = const are rays; density ~ 1/r. Show df for f = x^2 + y^2 contours too
xx, yy = np.mgrid[-3.2:3.2:400j, -3.2:3.2:400j]
ph_ = np.arctan2(yy, xx)
ax[1].contour(xx, yy, ph_, levels=np.linspace(-np.pi, np.pi, 25)[1:-1], colors=C1, linewidths=.9, linestyles='solid')
x0, y0 = 2*np.cos(np.radians(40)), 2*np.sin(np.radians(40))
ax[1].annotate('', (x0 - 0.9*np.sin(np.radians(40)), y0 + 0.9*np.cos(np.radians(40))), (x0, y0), arrowprops=dict(arrowstyle='->', color=PU, lw=2))
ax[1].text(x0 - 1.6, y0 + 0.9, r'vector $V$', color=PU)
ax[1].set_title(r'the covector $d\phi$, drawn as its level sets:' '\n' r'$d\phi(V)$ = number of lines $V$ crosses', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'gr-2-tensors/fig_polar_basis.png'); plt.close(fig)
print('ok')
