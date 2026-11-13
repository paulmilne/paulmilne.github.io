"""Checks of gr.py against textbook results (run: python3 test_gr.py)."""
import sympy as sp
from gr import Metric, is_zero
t, x, y, z, r, th, ph, eps = sp.symbols('t x y z r theta phi epsilon', real=True)
a, M = sp.symbols('a M', positive=True)
Phi = sp.Function('Phi')(x, y, z)

s2 = Metric(sp.diag(a**2, a**2*sp.sin(th)**2), [th, ph])
G = s2.christoffel()
assert is_zero(G[0][1][1] + sp.sin(th)*sp.cos(th)), "Gamma^theta_phiphi"
assert is_zero(G[1][0][1] - sp.cos(th)/sp.sin(th)), "Gamma^phi_thetaphi"
assert is_zero(s2.riemann_lower()[0][1][0][1] - a**2*sp.sin(th)**2), "R_thetaphithetaphi = a^2 sin^2"
pol = Metric(sp.diag(1, r**2), [r, ph]); G = pol.christoffel()
assert is_zero(G[0][1][1] + r) and is_zero(G[1][0][1] - 1/r), "polar Christoffels"
assert all(is_zero(v) for v in pol.ricci()), "flat plane is flat in polar coordinates"
f = 1 - 2*M/r
sch = Metric(sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2), [t, r, th, ph])
assert all(is_zero(v) for v in sch.ricci()), "Schwarzschild is Ricci-flat"
assert not sch.is_max_symmetric(sp.Integer(0)), "but not flat"
# Newtonian limit: g = diag(-(1+2 eps Phi), (1-2 eps Phi) delta) to first order in eps
weak = Metric(sp.diag(-(1 + 2*eps*Phi), *(3*[1 - 2*eps*Phi])), [t, x, y, z])
R00 = sp.series(weak.ricci()[0, 0], eps, 0, 2).removeO()
lap = sum(sp.diff(Phi, v, 2) for v in (x, y, z))
assert is_zero(sp.expand(R00 - eps*lap)), R00
print("all checks passed")
