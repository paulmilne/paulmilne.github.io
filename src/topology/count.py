"""Count topologies on a set of n points by brute force (n <= 4), and check Euler
characteristics of some polyhedra used in the explainer."""
from itertools import combinations
def topologies(n):
    X = (1 << n) - 1; subsets = [s for s in range(1, X)]       # proper non-empty subsets
    count = 0; hausdorff = 0
    for mask in range(1 << len(subsets)):
        T = {0, X} | {subsets[i] for i in range(len(subsets)) if mask >> i & 1}
        if all((a | b) in T and (a & b) in T for a in T for b in T):
            count += 1
    return count
for n in range(1, 5): print(n, "points:", topologies(n), "topologies")
for name, V, E, F in [("tetrahedron", 4, 6, 4), ("cube", 8, 12, 6), ("octahedron", 6, 12, 8), ("icosahedron", 12, 30, 20), ("square torus (3x3 grid)", 9, 18, 9)]:
    print(name, "V-E+F =", V - E + F)
