"""Figures for the Interstellar bulk post and the branes / de Sitter explainers."""
import json, numpy as np, mpmath as mp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
OUT = '../../'
import miller as M  # noqa (prints its table)

# ---------- Miller's planet ----------
mp.mp.dps = 50
le = np.linspace(-20, -1, 200); u = [float(M.ut(M.risco(1-mp.power(10, x)), 1-mp.power(10, x))) for x in le]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].loglog(10**le, u, color=C1, lw=2, label='innermost stable orbit')
ax[0].axhline(7*8766, color=C2, ls='--', lw=1.3); ax[0].text(1e-19, 7*8766*1.4, '1 hour = 7 years (61,362×)', color=C2, fontsize=9)
ax[0].plot(1e-14, float(M.ut(M.risco(1-mp.mpf('1e-14')), 1-mp.mpf('1e-14'))), 'o', color=C2)
ax[0].annotate("Thorne's Gargantua\n1 − a = 10⁻¹⁴", (1e-14, 6.75e4), xytext=(2e-11, 5e5), fontsize=8.5, arrowprops=dict(arrowstyle='->', color=GR))
ax[0].plot(0.002, float(M.ut(M.risco(mp.mpf('0.998')), mp.mpf('0.998'))), 's', color=C3)
ax[0].annotate("Thorne limit for real\nblack holes, a = 0.998", (0.002, 6.0), xytext=(1e-7, 2.5), fontsize=8.5, arrowprops=dict(arrowstyle='->', color=GR))
ax[0].invert_xaxis(); ax[0].set_xlabel('how far short of maximum spin, 1 − a/M'); ax[0].set_ylabel('time dilation dt/dτ (far clock / planet clock)')
ax[0].set_title('Slowest clock in a stable orbit', fontsize=10)
a = 1 - mp.mpf('1e-14'); ri = M.risco(a); rh = M.rhor(a)
xs = np.logspace(-5.6, 0.5, 300); us = [float(M.ut(ri + mp.mpf(x)*(1), a)) for x in xs]
ax[1].loglog(xs, us, color=C1, lw=2)
r7 = M.ut  # placeholder
rm = json.load(open('miller.json'))['book']['r_miller']
ax[1].axhline(7*8766, color=C2, ls='--', lw=1.3); ax[1].axvline(rm - float(ri), color=C2, lw=0.8, ls=':')
ax[1].text((rm-float(ri))*1.3, 2e2, "Miller's planet", color=C2, fontsize=9, rotation=90)
ax[1].set_xlabel('orbit radius above the innermost stable orbit, (r − r_isco)/M'); ax[1].set_ylabel('dt/dτ')
ax[1].set_title('Spin 1 − 10⁻¹⁴: dilation against orbit radius', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'sci-fi/interstellar-bulk/fig_miller.png'); plt.close(fig)

# ---------- branes explainer: gravity with a compact extra dimension ----------
def force_compact(r, n, L=1.0, K=200):
    """Force from a point mass in 3 large dimensions plus n circles of circumference L,
    by the method of images. Normalised so that F*r^2 -> 1 far away."""
    if n == 1:
        k = np.arange(-K, K+1)[:, None]; rho2 = r[None, :]**2 + (k*L)**2
        F = (2*r[None, :]/rho2**2).sum(0)                 # -d/dr of 1/rho^2
        F += 2 * 2*r / (3 * (K*L)**3) * 0                 # tail negligible at K=200 for r<=50
        Ffar = 1/(L) * 2 / r**2 * (np.pi/2)               # asymptote: sum -> integral
        return F, Ffar
    j, k = np.meshgrid(np.arange(-K, K+1), np.arange(-K, K+1)); s2 = ((j**2 + k**2).ravel()*L*L)[:, None]
    rho2 = r[None, :]**2 + s2; F = (3*r[None, :]/rho2**2.5).sum(0)
    Ffar = 2*np.pi/(L*L) * 1/r**2                         # integral of 3r/(r^2+s^2)^(5/2) over the plane
    return F, Ffar
r = np.logspace(-2, 0.9, 160)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for n, c in [(1, C1), (2, C2)]:
    F, Ff = force_compact(r, n, K=400 if n == 1 else 60)
    ax[0].loglog(r, F/Ff, color=c, lw=2, label=f'{n} extra dimension{"s" if n > 1 else ""}')
    sl = -np.gradient(np.log(F), np.log(r)); ax[1].semilogx(r, sl, color=c, lw=2, label=f'{n} extra dimension{"s" if n > 1 else ""}')
ax[0].axhline(1, color=GR, ls=':'); ax[0].set_xlabel('distance / size of the extra dimension'); ax[0].set_ylabel('force ÷ Newton’s inverse square')
ax[0].set_title('Gravity is stronger at short range', fontsize=10); ax[0].legend()
ax[1].axhline(2, color=GR, ls=':'); ax[1].set_ylim(1.8, 4.2); ax[1].set_xlabel('distance / size of the extra dimension'); ax[1].set_ylabel('power law exponent (F ∝ 1/rᵖ)')
ax[1].set_title('Inverse square far away, 1/r^(2+n) close up', fontsize=10); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT+'explainers/branes-and-the-bulk/fig_force_law.png'); plt.close(fig)

# ---------- branes explainer: Randall-Sundrum warp factor and potential ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
y = np.linspace(-3, 3, 400)
ax[0].plot(y, np.exp(-2*np.abs(y)), color=C1, lw=2, label='warp factor e^(−2k|y|): size of a ruler')
z = (np.exp(np.abs(y))-1)                         # conformal coordinate k|z| = e^{k|y|} - 1
ax[0].plot(y, (1+z)**-1.5/1, color=C2, lw=2, ls='--', label='graviton bound to the brane (strength)')
ax[0].axvline(0, color='k', lw=3, alpha=.6); ax[0].text(0.12, 0.45, 'our\nbrane', fontsize=9)
ax[0].set_xlabel('distance into the bulk, k·y'); ax[0].set_title('Randall–Sundrum: a warped bulk (anti-de Sitter)', fontsize=10); ax[0].legend(fontsize=7.8, loc='upper left')
rr = np.logspace(-6, -2.5, 300)                    # metres
for ell, c in [(1e-4, C1), (3e-5, C2), (1e-5, C3)]:
    ax[1].loglog(rr*1e6, 1 + 2*ell**2/(3*rr**2), color=c, lw=2, label=f'ℓ = {ell*1e6:.0f} µm')
ax[1].axvspan(52, rr[-1]*1e6, color=GR, alpha=.15); ax[1].text(70, 1.6e3, 'measured\n(Eöt-Wash,\n2020)', fontsize=8.5, color='#555')
ax[1].set_xlabel('separation (µm)'); ax[1].set_ylabel('gravity ÷ Newton'); ax[1].set_title('Short-range correction, V = GM/r · (1 + 2ℓ²/3r²)', fontsize=10); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT+'explainers/branes-and-the-bulk/fig_randall_sundrum.png'); plt.close(fig)

# ---------- de Sitter explainer: the two hyperboloids ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(10, 4.6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
tau = np.linspace(-1.6, 1.6, 50); th = np.linspace(0, 2*np.pi, 70); T, TH = np.meshgrid(tau, th)
X0 = np.sinh(T); X1 = np.cosh(T)*np.cos(TH); X2 = np.cosh(T)*np.sin(TH)
ax.plot_surface(X1, X2, X0, color=C1, alpha=.25, linewidth=0)
for t0 in [-1.2, -0.6, 0, 0.6, 1.2]: ax.plot(np.cosh(t0)*np.cos(th), np.cosh(t0)*np.sin(th), np.sinh(t0)+0*th, color=C1, lw=1)
for th0 in [0, np.pi/2, np.pi, 3*np.pi/2]: ax.plot(np.cosh(tau)*np.cos(th0), np.cosh(tau)*np.sin(th0), np.sinh(tau), color=C2, lw=1.3)
ax.set_title('de Sitter (Λ > 0)\n−T² + X² + Y² = ℓ²', fontsize=10); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('T (time)')
ax.view_init(14, -60)
ax = fig.add_subplot(1, 2, 2, projection='3d')
rho = np.linspace(-1.4, 1.4, 50); T, R = np.meshgrid(th, rho)
U = np.cosh(R)*np.cos(T); V = np.cosh(R)*np.sin(T); X = np.sinh(R)
ax.plot_surface(U, V, X, color=C2, alpha=.20, linewidth=0)
for t0 in np.arange(0, 2*np.pi, np.pi/3): ax.plot(np.cosh(rho)*np.cos(t0), np.cosh(rho)*np.sin(t0), np.sinh(rho), color=C1, lw=1)
for r0 in [-1, 1]: ax.plot(np.cosh(r0)*np.cos(th), np.cosh(r0)*np.sin(th), np.sinh(r0)+0*th, color=C2, lw=1.2)
ax.plot(np.cos(th), np.sin(th), 0*th, color=C2, lw=2.6)
ax.set_title('anti-de Sitter (Λ < 0)\n−U² − V² + X² = −ℓ²', fontsize=10); ax.set_xlabel('U'); ax.set_ylabel('V'); ax.set_zlabel('X (space)')
ax.view_init(16, -60)
fig.text(0.27, 0.02, 'Blue circles: space at one moment, growing after T = 0.\nOrange lines: observers, flying apart.', ha='center', fontsize=8.5)
fig.text(0.74, 0.02, 'Orange circles run round in time and close up: an observer at rest\n(thick) returns to the same point. Blue lines: space at one moment.', ha='center', fontsize=8.5)
fig.subplots_adjust(bottom=0.12, top=0.88, left=0.0, right=1.0, wspace=0.0); fig.savefig(OUT+'explainers/de-sitter-and-anti-de-sitter/fig_hyperboloids.png'); plt.close(fig)

# ---------- de Sitter explainer: our universe tends to de Sitter ----------
H0 = 67.4; Om = 0.315; OL = 1 - Om - 9e-5; Or = 9e-5            # Planck 2018
Hs = H0/977.8                                                     # in 1/Gyr
from scipy.integrate import solve_ivp
f = lambda t, a: [Hs*a[0]*np.sqrt(Or/a[0]**4 + Om/a[0]**3 + OL)]
s = solve_ivp(f, (0, 80), [1e-6], rtol=1e-10, atol=1e-14, dense_output=True, max_step=0.05)
t = np.linspace(1e-4, 80, 4000); aa = s.sol(t)[0]; t0 = t[np.argmin(abs(aa-1))]
Hinf = Hs*np.sqrt(OL)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].semilogy(t, aa, color=C1, lw=2, label='our universe (ΛCDM, Planck 2018)')
ax[0].semilogy(t, np.exp(Hinf*(t-t0)), color=C2, ls='--', lw=1.5, label='pure de Sitter, e^(H∞ t)')
ax[0].axvline(t0, color=GR, ls=':'); ax[0].text(t0+0.8, 1e-3, f'today\n{t0:.1f} bn yr', fontsize=8.5)
ax[0].set_ylim(1e-4, 1e3); ax[0].set_xlabel('time since the Big Bang (billion years)'); ax[0].set_ylabel('scale factor (today = 1)'); ax[0].legend(fontsize=8)
ax[0].set_title('The expansion tends to de Sitter', fontsize=10)
Ht = np.sqrt(Or/aa**4 + Om/aa**3 + OL)*H0
ax[1].plot(t, Ht, color=C1, lw=2); ax[1].axhline(H0*np.sqrt(OL), color=C2, ls='--'); ax[1].set_ylim(0, 200)
ax[1].text(40, H0*np.sqrt(OL)+6, f'H∞ = H₀√Ω_Λ = {H0*np.sqrt(OL):.1f} km/s/Mpc', color=C2, fontsize=9)
ax[1].axvline(t0, color=GR, ls=':'); ax[1].set_xlabel('time since the Big Bang (billion years)'); ax[1].set_ylabel('Hubble rate (km/s/Mpc)')
ax[1].set_title('The Hubble rate levels off', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'explainers/de-sitter-and-anti-de-sitter/fig_lcdm_to_ds.png'); plt.close(fig)
print('t0', t0, 'Hinf', H0*np.sqrt(OL), 'c/Hinf Gly', 299792.458/(H0*np.sqrt(OL))*3.2616e-3)

# ---------- de Sitter explainer: causal diagrams ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
a0 = ax[0]; a0.set_aspect('equal'); a0.axis('off')
a0.plot([0, np.pi, np.pi, 0, 0], [0, 0, np.pi, np.pi, 0], color='k', lw=1)
a0.plot([0, np.pi], [0, np.pi], color=C2, lw=1.5); a0.plot([0, np.pi], [np.pi, 0], color=C2, lw=1.5, ls='--')
a0.plot([0, 0], [0, np.pi], color=C1, lw=3); a0.text(-0.12, np.pi/2, 'you', rotation=90, va='center', ha='right', color=C1)
a0.fill([0, 0, np.pi], [0, np.pi, 0], color=C1, alpha=.08)
a0.text(0.75, 1.05, 'events you\ncan ever see', fontsize=8.5, color=C1, ha='center')
a0.text(2.45, 0.85, 'your event horizon', fontsize=8.5, color=C2, rotation=-45, ha='center')
a0.text(2.05, 2.25, 'limit of where your\nsignals can reach', fontsize=8.5, color=C2, rotation=45, ha='center')
a0.text(np.pi/2, np.pi+0.12, 'infinite future', ha='center', fontsize=9); a0.text(np.pi/2, -0.25, 'infinite past', ha='center', fontsize=9)
a0.set_title('de Sitter: horizons', fontsize=10)
a1 = ax[1]; a1.set_xlim(-0.2, np.pi/2+1.0); a1.set_ylim(-0.3, 2*np.pi+0.3); a1.set_aspect('equal'); a1.axis('off')
a1.plot([0, 0], [0, 2*np.pi], color='k', lw=1); a1.plot([np.pi/2]*2, [0, 2*np.pi], color='k', lw=2.5)
a1.text(-0.08, np.pi, 'centre', rotation=90, va='center', ha='right', fontsize=9)
a1.text(np.pi/2, 2*np.pi+0.12, 'boundary (infinitely far away)', fontsize=8.5, ha='center')
tt = np.linspace(0, 2*np.pi, 400)
a1.plot([0, np.pi/2, 0, np.pi/2, 0], [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], color=C2, lw=1.6)
a1.text(np.pi/2+0.1, np.pi/2, 'light reaches the\nboundary and\ncomes back', fontsize=8.5, color=C2, va='center')
for amp, c in [(0.35, C1), (0.8, C3), (1.2, PU)]:
    x = np.arcsin(np.sin(amp)*np.abs(np.sin(tt)))     # timelike geodesics thrown out from the centre
    a1.plot(x, tt, color=c, lw=1.4)
a1.text(np.pi/2+0.1, 3.9, 'thrown bodies all\nfall back after\nthe same time, πℓ/c', fontsize=8.5, color=C3, va='center')
a1.text(0.8, -0.3, 'time runs upwards', fontsize=8.5, ha='center', color='#555')
a1.set_title('anti-de Sitter: a box', fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'explainers/de-sitter-and-anti-de-sitter/fig_causal.png'); plt.close(fig)
