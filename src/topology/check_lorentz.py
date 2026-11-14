"""Check: from a positive-definite metric h and a nowhere-zero vector V, the metric
g = h - 2 (hV)(hV)^T / (V^T h V) is Lorentzian (one negative eigenvalue), V is timelike for g,
and vectors h-orthogonal to V keep their h-length. Random tests in 4 dimensions."""
import numpy as np
rng = np.random.default_rng(1); ok = True
for trial in range(10000):
    A = rng.normal(size=(4, 4)); h = A@A.T + 0.1*np.eye(4); V = rng.normal(size=4)
    hV = h@V; g = h - 2*np.outer(hV, hV)/(V@hV)
    neg = (np.linalg.eigvalsh(g) < 0).sum()           # Sylvester: inertia does not depend on the basis
    W = rng.normal(size=4); W -= (W@hV)/(V@hV)*V       # h-orthogonal to V
    ok &= neg == 1 and V@g@V < 0 and np.isclose(W@g@W, W@h@W) and np.isclose(V@g@V, -(V@h@V))
print("10,000 random cases: g Lorentzian, V timelike with g(V,V) = -h(V,V), h-orthogonal vectors unchanged:", ok)
