"""Gear trains from Technic gears: approximating the ratios an orrery needs.

A train of meshing pairs multiplies speed by the product of (driver teeth /
driven teeth) over its stages. Technic gears mesh within a family (spur with
spur, double-bevel with double-bevel), so each stage is a pair from one family.
Every tooth count is built from the primes 2, 3, 5 and 7, so every ratio a
train can make is a "7-smooth" rational number; that is the arithmetic limit on
how well a LEGO orrery can keep time.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

from .geometry import BEVEL, SPUR

# periods in mean solar days
SIDEREAL_YEAR = 365.256363     # Earth round the Sun, relative to the stars
TROPICAL_YEAR = 365.242190     # season to season
SYNODIC_MONTH = 29.530589      # new Moon to new Moon
SIDEREAL_MONTH = 27.321662     # Moon round the Earth, relative to the stars

SIDEREAL_DAY = 1 / (1 + 1 / SIDEREAL_YEAR)        # one turn of the Earth relative to the stars (0.99727 d)

# The orrery's two reductions (see the post): the crank turns the central shaft once per
# sidereal day; the year arm must turn once per SIDEREAL_DAYS_PER_YEAR turns of the crank,
# and the Moon train riding on the arm must reduce by the synodic month in solar days.
SIDEREAL_DAYS_PER_YEAR = SIDEREAL_YEAR + 1         # 366.2564
MOON_REDUCTION = SYNODIC_MONTH                     # 29.5306

WORMS = (16, 24, 40)          # gears a single-start worm can drive (one tooth per turn)


def continued_fraction(x: float, n: int = 10) -> list[int]:
    out = []
    for _ in range(n):
        a = math.floor(x)
        out.append(a)
        if abs(x - a) < 1e-12:
            break
        x = 1 / (x - a)
    return out


def convergents(cf: list[int]) -> list[Fraction]:
    h0, h1, k0, k1 = 0, 1, 1, 0
    out = []
    for a in cf:
        h0, h1 = h1, a * h1 + h0
        k0, k1 = k1, a * k1 + k0
        out.append(Fraction(h1, k1))
    return out


def factor(n: int) -> dict[int, int]:
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def largest_prime(fr: Fraction) -> int:
    return max([1] + list(factor(fr.numerator)) + list(factor(fr.denominator)))


def stage_options(families=(SPUR, BEVEL), worms: bool = True):
    """Every reducing stage as (driver teeth, driven teeth): pairs within a family with the
    driven gear at least as big, and a worm (1 'tooth') driving a spur gear."""
    out = []
    for fam in families:
        for a in fam:
            for b in fam:
                if b > a:
                    out.append((a, b))
    if worms:
        out += [(1, n) for n in WORMS]
    return out


def ratio(stages) -> Fraction:
    """Reduction of a train: product of driven / driver."""
    r = Fraction(1)
    for a, b in stages:
        r *= Fraction(b, a)
    return r


def best_by_stages(target: float, max_stages: int, max_worms: int = 1, worms: bool = True):
    """The best reduction with exactly k stages, k = 1..max_stages: (ratio, stages, rel. error).

    At most max_worms worm stages (a worm turns slowly and wastes height; one is plenty)."""
    opts = stage_options(worms=worms)
    out = {}
    for k in range(1, max_stages + 1):
        champion = None
        for combo in itertools.combinations_with_replacement(range(len(opts)), k):
            stages = [opts[i] for i in combo]
            if sum(a == 1 for a, _ in stages) > max_worms:
                continue
            r = ratio(stages)
            e = float(r) / target - 1
            if champion is None or abs(e) < abs(champion[2]) or (abs(e) == abs(champion[2]) and
                                                                  max(r.numerator, r.denominator) <
                                                                  max(champion[0].numerator, champion[0].denominator)):
                champion = (r, stages, e)
        out[k] = champion
    return out


def reachable(max_stages: int, max_worms: int = 1, worms: bool = True):
    """Every reduction a train of up to max_stages can make, with one way to make it.

    Built up a stage at a time: level k holds every ratio reachable with k stages. The
    sets stay small because every ratio is 2^a 3^b 5^c 7^d with small exponents.
    Returns {k: {ratio: stages}} (ratios first reached at fewer stages are kept there).
    """
    opts = stage_options(worms=worms)
    levels = {0: {Fraction(1): ()}}
    seen = {Fraction(1)}
    for k in range(1, max_stages + 1):
        nxt = {}
        for r, st in levels[k - 1].items():
            for a, b in opts:
                if a == 1 and sum(x == 1 for x, _ in st) >= max_worms:
                    continue
                q = r * Fraction(b, a)
                if q not in seen and q not in nxt:
                    nxt[q] = st + ((a, b),)
        levels[k] = nxt
        seen |= set(nxt)
    return levels


def best_reachable(target: float, levels) -> dict[int, tuple]:
    """Best ratio using at most k stages, for each k: (ratio, stages, relative error)."""
    out, champ = {}, None
    for k in sorted(levels):
        for r, st in levels[k].items():
            e = float(r) / target - 1
            if champ is None or abs(e) < abs(champ[2]):
                champ = (r, st, e)
        out[k] = champ
    return out


# ------------------------------------------------------------------ the planets
# Mean longitudes at J2000 (deg) and their rates (deg per Julian century), JPL "Approximate
# Positions of the Planets", Table 1 (1800-2050). "Earth" is the Earth-Moon barycentre.
PLANETS = {
    "Mercury": (252.25032350, 149472.67411175),
    "Venus":   (181.97909950, 58517.81538729),
    "Earth":   (100.46457166, 35999.37244981),
    "Mars":    (-4.55343205, 19140.30268499),
    "Jupiter": (34.39644051, 3034.74612775),
    "Saturn":  (49.95424423, 1222.49362201),
    "Uranus":  (313.23810451, 428.48202785),
    "Neptune": (-55.12002969, 218.45945325),
}


def planet_ratio(name: str) -> float:
    """Turns of the crank (Earth years) per turn of the planet: its period in Earth years."""
    return PLANETS["Earth"][1] / PLANETS[name][1]


def mean_longitude(name: str, jd: float) -> float:
    """Mean heliocentric longitude (deg) at Julian date jd."""
    L0, rate = PLANETS[name]
    return (L0 + rate * (jd - 2451545.0) / 36525.0) % 360.0


def reachable_mixed(max_stages: int):
    """Every ratio a train of up to max_stages can make when stages may step up or down
    (no worms: a worm cannot be driven backwards). {k: {ratio: stages}}, ratio = product of
    driven/driver, so values below 1 are speed-ups."""
    opts = []
    for fam in (SPUR, BEVEL):
        for a in fam:
            for b in fam:
                if a != b:
                    opts.append((a, b))
    levels = {0: {Fraction(1): ()}}
    seen = {Fraction(1)}
    for k in range(1, max_stages + 1):
        nxt = {}
        for r, st in levels[k - 1].items():
            for a, b in opts:
                q = r * Fraction(b, a)
                if q not in seen and q not in nxt:
                    nxt[q] = st + ((a, b),)
        levels[k] = nxt
        seen |= set(nxt)
    return levels
