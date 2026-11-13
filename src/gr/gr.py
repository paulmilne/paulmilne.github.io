"""A small symbolic general-relativity toolkit (SymPy), used to check every formula in the
blog's general relativity series. Conventions (Misner, Thorne & Wheeler; Carroll):
  signature (-,+,+,+); Gamma^a_{bc} = 1/2 g^{ad}(d_b g_{dc} + d_c g_{db} - d_d g_{bc});
  R^a_{bcd} = d_c Gamma^a_{db} - d_d Gamma^a_{cb} + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb};
  R_{bd} = R^a_{bad};  G_{ab} = R_{ab} - R g_{ab}/2."""
import sympy as sp
from itertools import product

def simp(e):
    """Simplify, trying harder for hyperbolic and trigonometric expressions."""
    e = sp.simplify(e)
    if e == 0 or not e.has(sp.sinh, sp.cosh, sp.tanh, sp.exp, sp.sin, sp.cos):
        return e
    e2 = sp.simplify(sp.factor(sp.expand(e.rewrite(sp.exp))))
    e3 = sp.simplify(sp.trigsimp(e))
    best = min([e, e2, e3], key=sp.count_ops)
    if best.has(sp.exp) and not e.has(sp.exp):          # prefer hyperbolic form when equal
        back = sp.simplify(best.rewrite(sp.cosh))
        if sp.count_ops(back) <= sp.count_ops(e): best = back
        else: best = e if sp.count_ops(e) <= sp.count_ops(e3) else e3
    return best

def is_zero(e):
    e = sp.simplify(e)
    return e == 0 or sp.simplify(sp.expand(e.rewrite(sp.exp))) == 0

class Metric:
    def __init__(self, g, coords, name=""):
        self.g = sp.Matrix(g); self.x = list(coords); self.n = len(coords); self.name = name
        self.ginv = sp.simplify(self.g.inv())
        self._G = self._R = self._Ric = None

    def christoffel(self):
        if self._G is None:
            n, g, gi, x = self.n, self.g, self.ginv, self.x
            self._G = [[[simp(sum(gi[a, d] * (sp.diff(g[d, c], x[b]) + sp.diff(g[d, b], x[c]) - sp.diff(g[b, c], x[d]))
                                         for d in range(n)) / 2) for c in range(n)] for b in range(n)] for a in range(n)]
        return self._G

    def riemann(self):            # R[a][b][c][d] = R^a_{bcd}
        if self._R is None:
            G, n, x = self.christoffel(), self.n, self.x
            self._R = [[[[simp(sp.diff(G[a][d][b], x[c]) - sp.diff(G[a][c][b], x[d])
                          + sum(G[a][c][e] * G[e][d][b] - G[a][d][e] * G[e][c][b] for e in range(n)))
                          for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
        return self._R

    def ricci(self):
        if self._Ric is None:
            R, n = self.riemann(), self.n
            self._Ric = sp.Matrix(n, n, lambda b, d: simp(sum(R[a][b][a][d] for a in range(n))))
        return self._Ric

    def ricci_scalar(self):
        Ric, gi, n = self.ricci(), self.ginv, self.n
        return simp(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)))

    def einstein(self):
        return (self.ricci() - self.ricci_scalar() * self.g / 2).applyfunc(simp)

    def riemann_lower(self):      # R_{abcd}
        R, g, n = self.riemann(), self.g, self.n
        return [[[[simp(sum(g[a, e] * R[e][b][c][d] for e in range(n))) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]

    def kretschmann(self):
        Rl, gi, n = self.riemann_lower(), self.ginv, self.n
        Ru = [[[[sum(gi[a, p] * gi[b, q] * gi[c, r] * gi[d, s] * Rl[p][q][r][s]
                     for p in range(n) for q in range(n) for r in range(n) for s in range(n) if Rl[p][q][r][s] != 0)
                 for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
        return simp(sum(Rl[a][b][c][d] * Ru[a][b][c][d] for a, b, c, d in product(range(n), repeat=4)))

    def is_max_symmetric(self, K):
        """Check R_{abcd} = K (g_ac g_bd - g_ad g_bc): constant curvature K."""
        Rl, g, n = self.riemann_lower(), self.g, self.n
        return all(is_zero(Rl[a][b][c][d] - K * (g[a, c] * g[b, d] - g[a, d] * g[b, c]))
                   for a, b, c, d in product(range(n), repeat=4))

    def nonzero_christoffel(self):
        G, x = self.christoffel(), self.x
        return {(a, b, c): G[a][b][c] for a, b, c in product(range(self.n), repeat=3) if G[a][b][c] != 0 and b <= c}

    def geodesic_equations(self, lam=sp.Symbol('lambda')):
        G, x = self.christoffel(), self.x
        X = [sp.Function(str(xi))(lam) for xi in x]; sub = dict(zip(x, X))
        return [sp.Eq(sp.diff(X[a], lam, 2), -sum(G[a][b][c].subs(sub) * sp.diff(X[b], lam) * sp.diff(X[c], lam)
                       for b in range(self.n) for c in range(self.n))) for a in range(self.n)]
