"""Checks for parts 5-7: Riemann symmetries and component count, contracted Bianchi
identity for FRW, geodesic deviation in Schwarzschild, embedding of de Sitter."""
import sympy as sp, itertools as it
from gr import Metric, is_zero
t, x, y, z, r, th, ph, chi = sp.symbols('t x y z r theta phi chi', real=True)
M, l = sp.symbols('M ell', positive=True)
A = sp.Function('a')(t)
# 1. Riemann symmetries for Schwarzschild
f = 1 - 2*M/r
sch = Metric(sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2), [t, r, th, ph]); Rl = sch.riemann_lower()
ok = all(is_zero(Rl[a][b][c][d] + Rl[b][a][c][d]) and is_zero(Rl[a][b][c][d] - Rl[c][d][a][b]) and
         is_zero(Rl[a][b][c][d] + Rl[a][c][d][b] + Rl[a][d][b][c]) for a, b, c, d in it.product(range(4), repeat=4))
print("Riemann symmetries hold for Schwarzschild:", ok)
n = 4; print("independent components n^2(n^2-1)/12 =", n*n*(n*n - 1)//12)
# tidal tensor for a static observer: E^r_r = R^r_{t r t} / f  (orthonormal) -> -2M/r^3 stretch
R = sch.riemann()
print("orthonormal R^r_{trt}:", sp.simplify(R[1][0][1][0]/f), "  R^th_{t th t}:", sp.simplify(R[2][0][2][0]/f))
# 2. contracted Bianchi identity: nabla_mu G^mu_nu = 0 for flat FRW with arbitrary a(t)
frw = Metric(sp.diag(-1, A**2, A**2, A**2), [t, x, y, z]); G = frw.christoffel(); Gt = frw.einstein()
Gu = sp.simplify(frw.ginv * Gt)        # G^mu_nu
X = frw.x
div = [sp.simplify(sum(sp.diff(Gu[m, nn], X[m]) for m in range(4)) + sum(G[m][m][lam]*Gu[lam, nn] for m in range(4) for lam in range(4))
       - sum(G[lam][m][nn]*Gu[m, lam] for m in range(4) for lam in range(4))) for nn in range(4)]
print("nabla_mu G^mu_nu for FRW:", div)
# 3. de Sitter: global coordinates on the hyperboloid -T^2 + X1^2+..+X4^2 = ell^2
T = l*sp.sinh(t/l); X1 = l*sp.cosh(t/l)*sp.cos(chi); X2 = l*sp.cosh(t/l)*sp.sin(chi)*sp.cos(th)
X3 = l*sp.cosh(t/l)*sp.sin(chi)*sp.sin(th)*sp.cos(ph); X4 = l*sp.cosh(t/l)*sp.sin(chi)*sp.sin(th)*sp.sin(ph)
print("on hyperboloid:", sp.simplify(-T**2 + X1**2 + X2**2 + X3**2 + X4**2))
E = sp.Matrix([T, X1, X2, X3, X4]); J = E.jacobian([t, chi, th, ph]); eta5 = sp.diag(-1, 1, 1, 1, 1)
print("induced metric (global):", sp.simplify(J.T*eta5*J))
# flat slicing: T = ell sinh(t/l) + x^2/(2l) e^{t/l}, X4 = ell cosh(t/l) - x^2/(2l) e^{t/l}, Xi = e^{t/l} x_i
xs = sp.symbols('x1 x2 x3', real=True); s2 = sum(v**2 for v in xs)
T = l*sp.sinh(t/l) + s2/(2*l)*sp.exp(t/l); X4 = l*sp.cosh(t/l) - s2/(2*l)*sp.exp(t/l)
E = sp.Matrix([T, *[sp.exp(t/l)*v for v in xs], X4]); J = E.jacobian([t, *xs])
print("flat slicing on hyperboloid:", sp.simplify((-T**2 + sum((sp.exp(t/l)*v)**2 for v in xs) + X4**2).rewrite(sp.exp)))
print("induced metric (flat slicing):", sp.simplify((J.T*eta5*J).applyfunc(lambda e: sp.simplify(e.rewrite(sp.exp)))))
print("T + X4 =", sp.simplify((T + X4).rewrite(sp.exp)))
