# Martian tripod

A MuJoCo model of a Wells-style fighting-machine: a 248-tonne, 30-metre tripod
that walks with a three-dimensional tumbling gait and burns what it passes
with a Heat-Ray.

## Setup (macOS)

```bash
cd ~/Dev/Quarto/math-blog/martian-tripod
uv sync
```

## Run it

```bash
uv run render.py                    # out/tripod_attack_witness.mp4: the walk into the village, from the village
uv run render.py --camera track     # the same, following the machine
uv run render.py --still --at 100   # one frame, 100 s in
uv run mjpython view.py             # live, interactive (SPACE holds fire)
uv run walk_test.py                 # three gait cycles, with numbers
uv run stand_test.py                # balance controller holding the stance
uv run rimless_wheel.py             # why a tripod cannot simply roll
```

`render.py` plays at double speed by default (`--speed 1` for real time).

## Files

| File | Purpose |
|---|---|
| `tripod/params.py` | Single source of truth: dimensions, masses, motor limits, stance IK |
| `tripod/build.py` | Generates `models/tripod.xml` (MJCF), including the village |
| `tripod/kinematics.py` | Leg forward and inverse kinematics in the world frame |
| `tripod/control.py` | Whole-body balance controller (torque-controlled legs) |
| `tripod/gait.py` | The tumbling gait |
| `tripod/heatray.py` | Heat-Ray turret, ray casting, fire, and beam drawing |
| `tripod/martian.py` | Gait plus Heat-Ray: walks, picks targets, burns them |
| `render.py`, `view.py` | Offline video / stills, and the live viewer |
| `walk_test.py`, `stand_test.py` | Headless checks with pass/fail |
| `rimless_wheel.py` | The first experiment: the milking stool as a rimless wheel |

## The model

* **Hood**: 160 t brass ellipsoid, 9 m across, on a free joint.
* **Legs**: three, each with hip yaw, hip pitch and knee, driven by torque motors
  limited to 40 MN·m (about six times the standing load). 16 m thigh, 28 m shin;
  the stance puts the hood 30 m up with the feet on a 22 m circle.
* **Heat-Ray**: a projector on a two-axis arm under the front of the hood, with
  position servos; yaw is unlimited.
* **Tentacles**: two chains of passive spring-damper links.
* **Village**: a lodge, cottages, an inn, a church, haystacks and a stand of pines.
  Every geom whose name starts `burn_` can be set alight.

## Result 1: the milking stool cannot roll by itself

Treat Wells's "milking stool tilted and bowled violently along" as a planar
rimless wheel with N spokes of length L and radius of gyration kL. Conserving
angular momentum about the new foot at each footfall gives

  w+ / w- = (k² + cos 2α) / (1 + k²),  α = π/N.

With three legs cos 2α = −½, so unless k² > ½ every footfall stops the roll dead.
The tripod has k² ≈ 0.11. MuJoCo agrees: one footfall leaves the three-spoke
wheel planted on two feet. The six-spoke limit cycle matches the analytic value
to about 1%. So the legs must carry the body over every step.

## Result 2: a tumbling gait that works

Joint-angle servos turned out to be the wrong tool. A rigid topple about the
support leaves every joint angle unchanged, so position servos never see it,
and when their targets disagree about where the body is, a foot lifts. The legs
are therefore torque-controlled by a whole-body balance controller
(`control.py`): it computes the ground force and moment that bring the centre
of mass and hood attitude back on plan, shares them between the planted feet
with a least-squares solve that never pulls on the ground, maps them to joint
torques through each leg's Jacobian, and adds a posture term that holds the
hood's attitude relative to its legs.

The gait (`gait.py`) goes one leg at a time:

1. **Transfer**: on three feet, carry the centre of mass to 3 m inside the edge
   between the two feet that will stay down.
2. **Swing**: lift the third leg. On two feet the machine is a hinge about the
   line between them and topples toward the lifted side; nothing can resist
   that, so the controller leaves it alone. The swing leg lands ahead and
   catches it.
3. **Settle**: back on three feet, steady the hood.

The lean goes round the compass each cycle, which is the bowled-stool motion
in 3-D. Phases end on events (the COM has arrived and slowed; the foot is down),
not just the clock.

`walk_test.py`, three cycles: 45 m covered for 45 m planned, maximum hood tilt
17°, about 16% of the time on two legs, at 1.1 mph. It is slow because physics
says so: the centre of mass has to travel about 15 m between support edges each
step, and carrying 248 t that far any faster saturates the motors and shifts
the ground load enough to lift a foot. Gentle spin (about 6° per step) and any
heading work. At 12° per step the hip-yaw joints run out of range.

## Result 3: the Heat-Ray

The beam is a ray cast from the projector along its axis (`mj_ray`), so it is
stopped by the first thing in the village it meets; it passes through the
machine itself, whose geoms are in a group the ray ignores. Wells calls it an
invisible sword of heat, so the beam is drawn only as a faint shimmer; what you
see is the flash where it lands, then flames. Anything it rests on heats up,
catches after 1.5 s of beam, spreads fire to the rest of the same building or
tree, burns, and is left charred. The Martian (`martian.py`) burns the nearest
target within 80 m and holds fire while on two feet.

## Next steps

1. A faster gait: plan the centre of mass with a model-predictive controller
   instead of a fixed transfer-then-swing pattern, so the machine can let itself
   fall further and keep moving.
2. Pre-emptive push-off from the trailing leg, which should cut the collision loss.
3. Bake the joint trajectories onto a Blender armature built from the same
   `params.py`, for a proper render with smoke and fire.
