"""Checks for the metrics explainer: the sphere in three charts (latitude-longitude, Mercator,
stereographic), the Poincare disc, Gaussian curvature of each, and the parallelogram law."""
import sympy as sp
from gr import Metric, is_zero
th, ph, x, y, u, v, lam, phi_ = sp.symbols('theta phi x y u v lambda varphi', real=True)
a = sp.symbols('a', positive=True)
def K2(m):   # Gaussian curvature of a 2D metric: R_{1212} / det g
    return sp.simplify(m.riemann_lower()[0][1][0][1] / m.g.det())
# 1. latitude-longitude
ll = Metric(sp.diag(a**2, a**2*sp.cos(phi_)**2), [phi_, lam]); print("lat-long K =", K2(ll))
# 2. Mercator: x = lambda, y = asinh(tan phi) -> phi = atan(sinh y); metric a^2 sech^2 y (dx^2 + dy^2)
phi_y = sp.atan(sp.sinh(y)); J = sp.Matrix([[0, sp.diff(phi_y, y)], [1, 0]])   # (phi, lam) wrt (x, y)
gm = sp.simplify(J.T*sp.diag(a**2, a**2*sp.cos(phi_y)**2)*J); print("Mercator metric:", gm)
merc = Metric(sp.diag(a**2/sp.cosh(y)**2, a**2/sp.cosh(y)**2), [x, y]); print("Mercator K =", K2(merc))
# 3. stereographic: 4a^2/(1+u^2+v^2)^2 (du^2+dv^2)
st = Metric(sp.diag(*(2*[4*a**2/(1 + u**2 + v**2)**2])), [u, v]); print("stereographic K =", K2(st))
# 4. Poincare disc
pd = Metric(sp.diag(*(2*[4/(1 - u**2 - v**2)**2])), [u, v]); print("Poincare disc K =", K2(pd))
# distance from the centre to radius r in the Poincare disc
r = sp.symbols('r', positive=True); s = sp.symbols('s', positive=True)
print("Poincare distance to r:", sp.simplify(sp.integrate(2/(1 - s**2), (s, 0, r))), "= 2 artanh r")
# 5. parallelogram law ||x+y||^2 + ||x-y||^2 = 2||x||^2 + 2||y||^2 for p-norms with x=(1,0), y=(0,1)
for p in [1, 2, 3, sp.oo]:
    n = (lambda w: (abs(w[0])**p + abs(w[1])**p)**(sp.Rational(1)/p)) if p != sp.oo else (lambda w: max(abs(w[0]), abs(w[1])))
    X, Y = (1, 0), (0, 1)
    lhs = n((1, 1))**2 + n((1, -1))**2; rhs = 2*n(X)**2 + 2*n(Y)**2
    print(f"p={p}: lhs={sp.nsimplify(lhs)} rhs={rhs}  holds={sp.simplify(lhs - rhs) == 0}")
