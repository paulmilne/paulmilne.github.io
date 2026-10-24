# Martian tripod

A parametric MuJoCo model of a Wells-style fighting-machine, and the first
gait experiment: can it really move like "a milking stool tilted and bowled
violently along the ground"?

## Setup (macOS)

```bash
cd ~/Dev/Quarto/math-blog/martian-tripod
uv sync                      # creates .venv with mujoco, numpy, matplotlib, imageio
```

## What's here

| File | Purpose |
|---|---|
| `tripod/params.py` | Single source of truth: dimensions, masses, servo gains, stance IK |
| `tripod/build.py` | Generates `models/tripod.xml` (MJCF) from the parameters |
| `stand_test.py` | Drops the machine into its stance; checks it holds and reports servo torques |
| `rimless_wheel.py` | The milking-stool gait as a rimless wheel: analysis, MuJoCo validation, power estimate |
| `view.py` | Interactive viewer (sliders for every leg servo, cameras `witness`, `martian`, `track`) |
| `render.py` | Offline clip or still to `out/` |

```bash
uv run stand_test.py
uv run rimless_wheel.py          # writes out/rimless_wheel_limit_cycle.png
uv run view.py
uv run render.py                 # out/tripod_demo_witness.mp4, eye-level view by the cottage
uv run render.py --camera track --still
```

Change anything in `params.py` and every script picks it up; `models/tripod.xml`
is regenerated on each run.

## The model

* **Hood**: 160 t brass ellipsoid, 9 m across, on a free joint, with a jointed Heat-Ray arm.
* **Legs**: three, each hip yaw + hip pitch + knee, on position servos; 16 m thigh, 28 m shin.
  Stance puts the hood 30 m up with the feet on a 22 m circle. Knees sit just above hip height.
* **Tentacles**: two chains of ten passive spring-damper links.
* **Scale reference**: a cottage, ~7 m to the ridge.
* Total mass about 248 t. Holding the stance takes about 7 MN·m at each hip and knee.

## First result: the milking stool can't roll by itself

Treat the stool as a planar rimless wheel with N spokes of length L and radius
of gyration kL. Conserving angular momentum about the new foot at each
footfall gives

  w+ / w- = (k² + cos 2α) / (1 + k²),  α = π/N.

For three spokes cos 2α = −½, so unless k² > ½ (most of the inertia out in the
legs) each footfall stops the roll dead. The tripod's own mass distribution
gives k² ≈ 0.11. MuJoCo agrees: a single footfall leaves the three-spoke wheel
stationary on both feet, while the six-spoke wheel's limit cycle matches the
analytic value to about 1%.

So a "bowled" tripod needs its legs to push it over every step. Supplying that
from the impact loss alone, a 248 t machine with ~35 m legs needs roughly
6 MW for 18 mph and 12 MW for 27 mph (Froude number ≈ 0.4, near where animals
switch from walking to running).

## Next steps

1. A 3-D version of the tumble (two feet planted, body vaulting over the edge
   between them, as in Virginia Tech's STriDER), driven by the leg servos.
2. Pre-emptive push-off from the trailing leg, which should cut the collision
   loss sharply.
3. Bake the joint trajectories onto a Blender armature built from the same `params.py`.
