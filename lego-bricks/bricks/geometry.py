"""The LEGO lattice: dimensions, heights, sideways building, Technic diagonals and gears.

All lengths in millimetres unless stated. The system's unit is the stud pitch,
8 mm; a plate is 3.2 mm tall and a brick three plates, 9.6 mm, so

    plate : pitch = 2 : 5        brick : pitch = 6 : 5

(Meccano, for comparison, is a 1/2-inch = 12.7 mm grid of holes in every direction.)
"""
from __future__ import annotations

import math
from fractions import Fraction

PITCH = 8.0           # stud to stud
PLATE = 3.2           # plate height
BRICK = 9.6           # brick height, three plates
STUD_D = 4.8          # stud diameter
STUD_H = 1.7          # stud height
PLAY = 0.1            # each brick is 0.1 mm short of the grid on every side (a 2x4 is 31.8 x 15.8)
PLATE_IN_STUDS = Fraction(2, 5)

# Technic gears: module 1 mm, so a gear of N teeth has a pitch diameter of N mm
SPUR = (8, 16, 24, 40)
BEVEL = (12, 20, 28, 36)          # double-bevel gears
GEARS = tuple(sorted(SPUR + BEVEL))


def stack_in_studs(bricks: int, plates: int) -> Fraction:
    """Height of a stack of bricks and plates, in stud pitches."""
    return (3 * bricks + plates) * PLATE_IN_STUDS


def stacks_on_grid(max_plates: int = 30):
    """Stack heights (in plates) that are a whole number of studs: multiples of 5."""
    return [p for p in range(1, max_plates + 1) if (p * PLATE_IN_STUDS).denominator == 1]


def diagonals(max_studs: int = 12, vertical_plates: bool = False, half: bool = False):
    """Hole-to-hole offsets that a straight Technic beam fits exactly.

    Horizontal offsets are in studs (or half studs if half=True). With vertical_plates,
    the second offset is a rise in plates (0.4 stud each), as when beams are pinned to
    Technic bricks stacked with plates between them.
    Returns (a, b, length in studs) with a, b in their own units.
    """
    out = []
    step = 0.5 if half else 1.0
    rng = range(0, int(max_studs / step) + 1)
    rise = range(0, int(max_studs / 0.4) + 1) if vertical_plates else rng
    for i in rng:
        for j in rise:
            if i == 0 and j == 0:
                continue
            dx = i * step
            dz = j * 0.4 if vertical_plates else j * step
            c = math.hypot(dx, dz)
            if c <= max_studs and abs(c - round(c)) < 1e-9 and i > 0 and j > 0:
                out.append((i * step, j, c) if vertical_plates else (dx, dz, c))
    return out


def gear_pairs(teeth=GEARS):
    """Centre distance of each pair (in studs) and how the pair can be mounted.

    in line on whole studs, in line on half studs, or on a diagonal with whole-stud
    (a, b) offsets, or across a stack of bricks and plates (x studs over, p plates up).
    """
    rows = []
    for i, n1 in enumerate(teeth):
        for n2 in teeth[i:]:
            d = (n1 + n2) / 16                      # (N1 + N2)/2 mm, in 8 mm studs
            ways = []
            if d == int(d):
                ways.append("in line")
            elif 2 * d == int(2 * d):
                ways.append("in line, half stud")
            for a in range(1, int(d) + 1):
                for b in range(a, int(d) + 1):
                    if abs(math.hypot(a, b) - d) < 1e-9:
                        ways.append(f"diagonal {a}x{b}")
            for x2 in range(1, int(2 * d) + 1):      # half-stud steps across, plates up
                x = x2 / 2
                p = math.sqrt(max(d * d - x * x, 0)) / 0.4
                if p >= 1 and abs(p - round(p)) < 1e-9:
                    ways.append(f"{x:g} across, {round(p)} plates up")
            rows.append((n1, n2, d, ways))
    return rows


MECCANO = {"pitch (mm)": 12.7, "vertical unit (mm)": 12.7, "gear pitch": "38 teeth per inch of diameter",
           "gear sizes": "19, 57, 95, 133 (multiples of 19)"}
LEGO = {"pitch (mm)": 8.0, "vertical unit (mm)": "3.2 (plate), 9.6 (brick)", "gear pitch": "module 1 mm",
        "gear sizes": "8, 16, 24, 40 spur; 12, 20, 28, 36 bevel"}
