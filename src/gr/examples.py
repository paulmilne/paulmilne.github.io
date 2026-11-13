"""Every curvature result quoted in the series, computed with gr.py and saved as LaTeX."""
import json, sympy as sp
from gr import Metric
t, x, y, z, r, th, ph, chi, tau, rho, w, u = sp.symbols('t x y z r theta phi chi tau rho w u', real=True)
a, l, H, M, k, Lam = sp.symbols('a ell H M k Lambda', positive=True)
A = sp.Function('a')(t)
out = {}
def record(key, m, K=None, extra=None):
    d = dict(name=m.name, coords=[sp.latex(c) for c in m.x], metric=sp.latex(m.g),
             christoffel={f"{a_},{b},{c}": sp.latex(v) for (a_, b, c), v in m.nonzero_christoffel().items()},
             ricci=sp.latex(m.ricci()), R=sp.latex(m.ricci_scalar()), einstein=sp.latex(m.einstein()))
    if K is not None: d["max_symmetric_K"] = sp.latex(K); d["is_max_symmetric"] = m.is_max_symmetric(K)
    if extra: d.update(extra)
    out[key] = d; print(key, "R =", m.ricci_scalar(), ("maxsym " + str(d.get("is_max_symmetric"))) if K is not None else "")
    return m

record("plane_polar", Metric(sp.diag(1, r**2), [r, ph], "flat plane, polar coordinates"))
s2 = record("sphere", Metric(sp.diag(a**2, a**2*sp.sin(th)**2), [th, ph], "sphere of radius a"), K=1/a**2)
record("hyperbolic_plane", Metric(sp.diag(1, a**2*sp.sinh(r/a)**2), [r, ph], "hyperbolic plane"), K=-1/a**2)
record("minkowski", Metric(sp.diag(-1, 1, 1, 1), [t, x, y, z], "Minkowski"), K=0)
f = 1 - 2*M/r
sch = record("schwarzschild", Metric(sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2), [t, r, th, ph], "Schwarzschild"))
out["schwarzschild"]["kretschmann"] = sp.latex(sch.kretschmann())
# FRW, flat: Einstein tensor gives the Friedmann equations
frw = record("frw_flat", Metric(sp.diag(-1, A**2, A**2, A**2), [t, x, y, z], "flat FRW"))
# de Sitter in three charts (4D, radius ell)
dsflat = record("ds_flat", Metric(sp.diag(-1, *(3*[sp.exp(2*t/l)])), [t, x, y, z], "de Sitter, flat slicing"), K=1/l**2)
dsstat = record("ds_static", Metric(sp.diag(-(1 - r**2/l**2), 1/(1 - r**2/l**2), r**2, r**2*sp.sin(th)**2), [t, r, th, ph], "de Sitter, static patch"), K=1/l**2)
dsglob = record("ds_global", Metric(sp.diag(-1, l**2*sp.cosh(t/l)**2, l**2*sp.cosh(t/l)**2*sp.sin(chi)**2, l**2*sp.cosh(t/l)**2*sp.sin(chi)**2*sp.sin(th)**2), [t, chi, th, ph], "de Sitter, global"), K=1/l**2)
# anti-de Sitter
adsglob = record("ads_global", Metric(sp.diag(-sp.cosh(rho)**2*l**2, l**2, l**2*sp.sinh(rho)**2, l**2*sp.sinh(rho)**2*sp.sin(th)**2), [t, rho, th, ph], "anti-de Sitter, global"), K=-1/l**2)
adspoinc = record("ads_poincare", Metric(sp.diag(*(l**2/z**2*sp.Matrix([-1, 1, 1, 1]))), [t, x, y, z], "anti-de Sitter, Poincare"), K=-1/l**2)
adsrs = record("ads_rs", Metric(sp.diag(-sp.exp(-2*y/l), sp.exp(-2*y/l), sp.exp(-2*y/l), 1), [t, x, z, y], "anti-de Sitter, Randall-Sundrum slicing (y>0)"), K=-1/l**2)
# Einstein equations with Lambda: check G + Lambda g = 0 for dS with Lambda = 3/ell^2
for key, m, L in [("ds_flat", dsflat, 3/l**2), ("ds_static", dsstat, 3/l**2), ("ds_global", dsglob, 3/l**2), ("ads_global", adsglob, -3/l**2), ("ads_poincare", adspoinc, -3/l**2)]:
    from gr import is_zero
    out[key]["vacuum_with_Lambda"] = all(is_zero(v) for v in (m.einstein() + L*m.g))
    print(key, "G + Lambda g = 0 with Lambda =", L, out[key]["vacuum_with_Lambda"])
json.dump(out, open("results.json", "w"), indent=1)
print("FRW G_tt:", sp.simplify(frw.einstein()[0, 0]), "  G_xx:", sp.simplify(frw.einstein()[1, 1]))
print("Kretschmann Schw:", out["schwarzschild"]["kretschmann"])
