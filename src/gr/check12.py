"""Checks for parts 1-2: the stereographic transition map, and a vector and a covector
transformed from Cartesian to polar coordinates."""
import sympy as sp
x, y, z, u, v, r, ph = sp.symbols('x y z u v r phi', real=True)
# stereographic charts on the unit sphere
uN, vN = x/(1 - z), y/(1 - z); uS, vS = x/(1 + z), y/(1 + z)
sub = {z: sp.sqrt(1 - x**2 - y**2)}
lhs = sp.simplify((uS - uN/(uN**2 + vN**2)).subs(sub)); print("u_S - u_N/(u_N^2+v_N^2) =", lhs)
print("u_N^2+v_N^2 =", sp.simplify((uN**2 + vN**2).subs(sub)), " vs (1+z)/(1-z) =", sp.simplify(((1 + z)/(1 - z)).subs(sub)))
# Jacobian from (x, y) to (r, phi)
X = sp.Matrix([r*sp.cos(ph), r*sp.sin(ph)]); J = X.jacobian([r, ph])       # dx^i/dx'^j
Jinv = sp.simplify(J.inv())                                                    # dx'^i/dx^j
print("components of d/dx in polar:", list(sp.simplify(Jinv * sp.Matrix([1, 0]))))
print("components of dx in polar:", list(sp.simplify(J.T * sp.Matrix([1, 0])).T))
