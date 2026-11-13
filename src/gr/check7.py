"""Checks for part 7: de Sitter static patch and anti-de Sitter global and Poincare
coordinates from the embeddings; the Poincare -> Randall-Sundrum change of variable;
comoving observers in the flat slicing are geodesics."""
import sympy as sp
from gr import Metric, is_zero
t, r, th, ph, rho, x1, x2, z, y = sp.symbols('t r theta phi rho x1 x2 z y', real=True)
l = sp.symbols('ell', positive=True)
def induced(E, coords, sig):
    J = sp.Matrix(E).jacobian(coords); return (J.T*sp.diag(*sig)*J).applyfunc(lambda e: sp.simplify(sp.trigsimp(e)))
# dS static patch: T = sqrt(l^2-r^2) sinh(t/l), X4 = sqrt(l^2-r^2) cosh(t/l), X1..3 = r * (unit vector)
w = sp.sqrt(l**2 - r**2)
E = [w*sp.sinh(t/l), r*sp.sin(th)*sp.cos(ph), r*sp.sin(th)*sp.sin(ph), r*sp.cos(th), w*sp.cosh(t/l)]
print("dS static, on hyperboloid:", sp.simplify(-E[0]**2 + sum(e**2 for e in E[1:])))
print("dS static induced metric:", induced(E, [t, r, th, ph], [-1, 1, 1, 1, 1]))
# AdS global: U = l cosh(rho) cos t, V = l cosh(rho) sin t, X = l sinh(rho) * unit vector, signature (-,-,+,+,+)
E = [l*sp.cosh(rho)*sp.cos(t), l*sp.cosh(rho)*sp.sin(t), l*sp.sinh(rho)*sp.sin(th)*sp.cos(ph), l*sp.sinh(rho)*sp.sin(th)*sp.sin(ph), l*sp.sinh(rho)*sp.cos(th)]
print("AdS global, on hyperboloid:", sp.simplify(-E[0]**2 - E[1]**2 + sum(e**2 for e in E[2:])))
print("AdS global induced metric:", induced(E, [t, rho, th, ph], [-1, -1, 1, 1, 1]))
# AdS Poincare: U = (z/2)(1 + (l^2 + x^2 - t^2)/z^2), X3 = (z/2)(1 - (l^2 - x^2 + t^2)/z^2), V = l t/z, X_i = l x_i / z
s2 = x1**2 + x2**2 - t**2
E = [z/2*(1 + (l**2 + s2)/z**2), l*t/z, l*x1/z, l*x2/z, z/2*(1 - (l**2 - s2)/z**2)]
print("AdS Poincare, on hyperboloid:", sp.simplify(-E[0]**2 - E[1]**2 + E[2]**2 + E[3]**2 + E[4]**2))
print("AdS Poincare induced metric:", induced(E, [t, x1, x2, z], [-1, -1, 1, 1, 1]))
# Poincare -> RS: z = l e^{y/l}
P = sp.diag(*(l**2/z**2*sp.Matrix([-1, 1, 1, 1])))
J = sp.diag(1, 1, 1, sp.diff(l*sp.exp(y/l), y))
print("RS form:", sp.simplify((J.T*P*J).subs(z, l*sp.exp(y/l))))
# comoving observers x = const in the flat slicing: geodesic equation for x(tau)=const, t = tau
m = Metric(sp.diag(-1, *(3*[sp.exp(2*t/l)])), [t, x1, x2, z]); G = m.christoffel()
u = [1, 0, 0, 0]
print("acceleration of comoving observers:", [sp.simplify(sum(G[a][b][c]*u[b]*u[c] for b in range(4) for c in range(4))) for a in range(4)])
