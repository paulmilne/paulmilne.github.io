"""Checks for parts 3-4."""
import sympy as sp
from gr import Metric, is_zero
r, ph, t, x, y, z, eps = sp.symbols('r phi t x y z epsilon', real=True)
pol = Metric(sp.diag(1, r**2), [r, ph]); G = pol.christoffel()
V = [sp.cos(ph), -sp.sin(ph)/r]                            # d/dx in polar components
cov = [[sp.simplify(sp.diff(V[n], [r, ph][m]) + sum(G[n][m][l]*V[l] for l in range(2))) for n in range(2)] for m in range(2)]
print("nabla_m (d/dx)^n in polar coordinates:", cov)
Phi = sp.Function('Phi')(x, y, z)
weak = Metric(sp.diag(-(1 + 2*eps*Phi), *(3*[1 - 2*eps*Phi])), [t, x, y, z]); Gw = weak.christoffel()
print("Gamma^i_00 to first order:", [sp.series(Gw[i][0][0], eps, 0, 2).removeO() for i in (1, 2, 3)])
# local flatness counting in 4D
n = 4; metric = n*(n+1)//2; d1 = metric*n; d2 = metric*n*(n+1)//2
c1 = n*n; c2 = n*n*(n+1)//2; c3 = n*(n*(n+1)*(n+2)//6)
print(f"metric comps {metric} (fixable {c1} from Lorentz-free part: {c1}-6={c1-6}); first derivs {d1} vs {c2}; second derivs {d2} vs {c3} -> {d2-c3} left")
