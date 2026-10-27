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
