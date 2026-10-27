"""Counting LEGO buildings, after Søren Eilers.

A *building* is a set of n identical 2x4 bricks, each one attached to the rest by
studs (so the whole is one piece), counted as the same building when it can be
turned about the vertical axis or moved into the other. Bricks sit on the stud
grid in layers; a brick is

    (x, y, z, o)    o = 0: four studs along x, two along y;  o = 1: turned 90 degrees

Two bricks in neighbouring layers are joined when their footprints overlap by at
least one stud; two bricks in the same layer may not overlap.

The enumeration grows buildings one brick at a time. Every connected building of
n bricks contains a brick whose removal leaves it connected (a leaf of a spanning
tree), so adding one brick, in every possible place, to every building of n - 1
bricks reaches every building of n. Each result is reduced to a canonical form,
the smallest of its four rotations, so duplicates collapse in a set.
"""
from __future__ import annotations

from collections import Counter

DIMS = {0: (4, 2), 1: (2, 4)}


def overlap(a, b) -> bool:
    ax, ay, _, ao = a
    bx, by, _, bo = b
    aw, ah = DIMS[ao]
    bw, bh = DIMS[bo]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def rot90(b):
    """Turn a brick a quarter turn about the vertical axis: stud (i, j) -> (-j, i)."""
    x, y, z, o = b
    w, h = DIMS[o]
    return (-(y + h - 1), x, z, 1 - o)


def canon(bricks) -> tuple:
    """Canonical form: the least of the four rotations, each moved to the origin."""
    best = None
    cur = list(bricks)
    for _ in range(4):
        mx = min(b[0] for b in cur)
        my = min(b[1] for b in cur)
        mz = min(b[2] for b in cur)
        t = tuple(sorted((b[0] - mx, b[1] - my, b[2] - mz, b[3]) for b in cur))
        if best is None or t < best:
            best = t
        cur = [rot90(b) for b in cur]
    return best


def attachments(building):
    """Every way to add one brick to a building, above or below any brick."""
    for b in building:
        bx, by, bz, bo = b
        bw, bh = DIMS[bo]
        for z in (bz - 1, bz + 1):
            same = [c for c in building if c[2] == z]
            for o in (0, 1):
                w, h = DIMS[o]
                for x in range(bx - w + 1, bx + bw):
                    for y in range(by - h + 1, by + bh):
                        nb = (x, y, z, o)
                        if not any(overlap(nb, c) for c in same):
                            yield nb


def grow(buildings) -> set:
    out = set()
    for s in buildings:
        for nb in attachments(s):
            out.add(canon(s + (nb,)))
    return out


def enumerate_buildings(n_max: int) -> dict[int, set]:
    """All buildings of 1..n_max bricks (n_max = 4 takes seconds; 5 is too many for Python)."""
    levels = {1: {canon([(0, 0, 0, 0)])}}
    for n in range(2, n_max + 1):
        levels[n] = grow(levels[n - 1])
    return levels


def height(building) -> int:
    return max(b[2] for b in building) + 1


def by_height(buildings) -> dict[int, int]:
    return dict(sorted(Counter(height(b) for b in buildings).items()))


def towers(n: int) -> int:
    """Buildings of n bricks, one on each level (height n), in closed form.

    Each brick sits on the one below in one of 46 ways (21 parallel, 25 crossed), so
    there are 46**(n-1) towers before symmetry. A half turn pairs them up, except the
    towers it leaves unchanged: those with every brick centred on the one below,
    parallel or crossed, 2**(n-1) of them. Burnside's lemma: (46**(n-1) + 2**(n-1)) / 2.
    """
    return (46 ** (n - 1) + 2 ** (n - 1)) // 2


# Contiguous buildings of n 2x4 bricks, OEIS A112389 (Eilers, Abrahamsen and others)
PUBLISHED = {1: 1, 2: 24, 3: 1560, 4: 119580, 5: 10166403, 6: 915103765, 7: 85747377755,
             8: 8274075616387, 9: 816630819554486, 10: 82052796578652749, 11: 8364925094730088313}
# by height, OEIS A112390
PUBLISHED_BY_HEIGHT = {2: {2: 24}, 3: {2: 500, 3: 1060}, 4: {2: 11707, 3: 59201, 4: 48672},
                       5: {2: 248688, 3: 3203175, 4: 4425804, 5: 2238736},
                       6: {2: 7946227, 3: 162216127, 4: 359949655, 5: 282010252, 6: 102981504}}
