# LEGO bricks: the lattice and the count

Code behind the post *915,103,765 Ways: The Numbers Inside a LEGO Brick*
(`mechanics/lego-numbers`).

- `bricks/geometry.py`: LEGO's dimensions and the lattice they make. Covers
  stack heights (one brick and two plates make two studs), building sideways,
  exact Technic diagonals, and which Technic gear pairs mesh on the grid.
- `bricks/count.py`: enumerates every *building* of n identical 2x4 bricks,
  counted up to rotation about the vertical, after Søren Eilers.
  - It grows buildings one brick at a time, and canonical forms remove the
    duplicates.
  - It reproduces the published counts 1, 24, 1560 and 119,580, split by
    height.
  - Also included: the closed form for towers, (46^(n-1) + 2^(n-1))/2, which
    gives LEGO's 1974 figure for six bricks, and the published sequence
    (OEIS A112389 and A112390).
- `bricks/render.py`: MuJoCo renders of buildings. Bricks are mocap bodies
  placed in one model, so a single renderer draws every picture.

```bash
uv sync
uv run python -c "from bricks.count import enumerate_buildings; print({n: len(s) for n, s in enumerate_buildings(4).items()})"
```

Four bricks takes a few seconds. Five would be about ten million buildings,
too many to hold in a Python set.

## The Technic orrery

Code behind *A Year in Gears: A LEGO Technic Orrery* (`mechanics/lego-orrery`).

- `bricks/gears.py`: continued fractions, and an exhaustive search over Technic
  gear trains. Every ratio is 7-smooth, of the form 2^a 3^b 5^c 7^d, so
  `reachable(k)` builds up the whole set stage by stage. The orrery's two
  targets are the sidereal days in a year (366.2564) and the synodic month
  (29.5306).
- `bricks/orrery.py`: the orrery in MuJoCo.
  - **Year train:** seven stages coiled round the central shaft, giving
    46875/128 = 366.2109.
  - **Spin chain:** a 1:1 chain on the arm (the solar day falls out of the
    epicyclic gearing).
  - **Moon train:** six stages on the arm, giving 945/32 = 29.53125.
  - Every mesh is a fixed tendon held at zero by an equality constraint.
  - The Sun lights the Earth and Moon from a spotlight that rides on the arm.
- `orrery_video.py`: films the orrery, with a day and lunation counter read
  from the mechanism.

```bash
uv run python orrery_video.py                       # a year, 24 s
uv run python orrery_video.py --camera earthcam --days 60 --seconds 20 --sunlight-only --name orrery_moon.mp4
```

## The whole solar system

Code behind *The Whole Solar System in LEGO Gears*
(`mechanics/lego-solar-system`).

- `bricks/gears.py`:
  - `PLANETS` holds JPL's mean longitudes and rates.
  - `reachable_mixed(k)` finds every ratio with stages in either direction (no
    worms), for Mercury and Venus, which must be geared up.
- `bricks/solar.py`: the grand orrery.
  - **Crank:** one turn is one year; the Earth's arm sits on the crank shaft.
  - **Trains:** for each other planet, the shortest train within 2° a century.
    Where none exists, the shortest that reaches the best accuracy eight stages
    allow.
  - **The tower:** each train is coiled on its own tier, with the outer planets
    at the bottom.
  - **Meshes:** 84 gears and 44 meshes, all fixed-tendon constraints.
  - **Start:** the arms are set to the planets' mean longitudes on
    27 October 2026.
- `solar_video.py`: films it, with a date counter.

```bash
uv run python solar_video.py                                       # 12 years, 30 s
uv run python solar_video.py --camera top --years 164.8 --seconds 24 --name solar_neptune_year.mp4
```
