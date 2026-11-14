"""Figures for the hairy ball section of the topology explainer."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams.update({'font.size':10,'figure.dpi':150})
C1,C2,C3,GR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#7b4fa3'
OUT='../../explainers/topology-and-open-sets/'

# ---- zeros of vector fields in the plane and their indices ----
def winding(F, n=2000):
    t = np.linspace(0, 2*np.pi, n); vx, vy = F(np.cos(t)*0.5, np.sin(t)*0.5)
    ang = np.unwrap(np.arctan2(vy, vx)); return round((ang[-1] - ang[0])/(2*np.pi))
fields = [('source', lambda x, y: (x, y)), ('sink', lambda x, y: (-x, -y)), ('vortex', lambda x, y: (-y, x)),
          ('saddle', lambda x, y: (x, -y)), ('dipole', lambda x, y: (x*x - y*y, 2*x*y))]
fig, ax = plt.subplots(1, 5, figsize=(12, 2.9))
g = np.linspace(-1, 1, 11); X, Y = np.meshgrid(g, g)
for a, (name, F) in zip(ax, fields):
    U, V = F(X, Y); N = np.hypot(U, V); N[N == 0] = 1
    a.quiver(X, Y, U/N, V/N, color=C1, scale=14, width=0.012); a.plot(0, 0, 'o', color=C2, ms=6)
    t = np.linspace(0, 2*np.pi, 100); a.plot(0.5*np.cos(t), 0.5*np.sin(t), color=GR, lw=.8, ls='--')
    a.set_aspect('equal'); a.axis('off'); a.set_title(f'{name}: index {winding(F):+d}', fontsize=10)
fig.text(0.5, 0.02, 'The index of a zero is the number of times the arrows turn, anticlockwise, as you go once round a small loop (dashed) about it.', ha='center', fontsize=8.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.08); fig.savefig(OUT+'fig_index.png'); plt.close(fig)
print('indices:', {n: winding(F) for n, F in fields})

# ---- sphere (two fields, each with zeros summing to 2) and a combed torus ----
fig = plt.figure(figsize=(12, 4.3))
def sphere_ax(k, title):
    a = fig.add_subplot(1, 3, k, projection='3d'); u, v = np.mgrid[0:np.pi:30j, 0:2*np.pi:40j]
    a.plot_surface(np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u), color='#dde6f2', alpha=.35, linewidth=0)
    a.set_box_aspect((1, 1, 1)); a.axis('off'); a.view_init(18, -50); a.set_xlim(-0.75, 0.75); a.set_ylim(-0.75, 0.75); a.set_zlim(-0.75, 0.75); a.set_title(title, fontsize=10); return a
pts = [(th, ph) for th in np.radians(np.arange(15, 166, 22.5)) for ph in np.radians(np.arange(0, 360, 30))]
a = sphere_ax(1, 'sphere: rotation about the axis\nvortices at both poles, 1 + 1 = 2')
for th, ph in pts:
    P = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]); Vv = np.array([-np.sin(ph), np.cos(ph), 0])*np.sin(th)*0.22
    a.quiver(*P, *Vv, color=C1, lw=1.2, arrow_length_ratio=.35)
a.scatter([0, 0], [0, 0], [1, -1], color=C2, s=30)
a = sphere_ax(2, 'sphere: flow from south to north pole\nsource and sink, 1 + 1 = 2')
for th, ph in pts:
    P = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]); Vv = -np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])*np.sin(th)*0.22
    a.quiver(*P, *Vv, color=C1, lw=1.2, arrow_length_ratio=.35)
a.scatter([0, 0], [0, 0], [1, -1], color=C2, s=30)
a = fig.add_subplot(1, 3, 3, projection='3d'); R, r = 1.0, 0.38
u, v = np.mgrid[0:2*np.pi:50j, 0:2*np.pi:30j]
a.plot_surface((R + r*np.cos(v))*np.cos(u), (R + r*np.cos(v))*np.sin(u), r*np.sin(v), color='#dde6f2', alpha=.35, linewidth=0)
for uu in np.radians(np.arange(0, 360, 20)):
    for vv in np.radians(np.arange(0, 360, 45)):
        P = np.array([(R + r*np.cos(vv))*np.cos(uu), (R + r*np.cos(vv))*np.sin(uu), r*np.sin(vv)])
        Vv = np.array([-np.sin(uu), np.cos(uu), 0])*0.2
        a.quiver(*P, *Vv, color=C3, lw=1.2, arrow_length_ratio=.35)
a.set_box_aspect((1, 1, 0.45)); a.set_xlim(-1.05, 1.05); a.set_ylim(-1.05, 1.05); a.set_zlim(-0.47, 0.47); a.axis('off'); a.view_init(35, -50); a.set_title('torus: combed flat round the ring\nno zeros at all: χ = 0', fontsize=10)
fig.subplots_adjust(left=0, right=1, top=0.86, bottom=0.02, wspace=0); fig.savefig(OUT+'fig_hairy.png'); plt.close(fig)
