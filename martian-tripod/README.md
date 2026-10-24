# Martian tripod

A MuJoCo model of a Wells-style fighting-machine: a 257-tonne, 30-metre tripod
that walks with a three-dimensional tumbling gait at about 5.5 mph, planned by
model-predictive control, and burns what it passes with a Heat-Ray.

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
uv run walk_test.py                 # three gait cycles, with numbers (--gait transfer for the slow one)
uv run stand_test.py                # balance controller holding the stance
uv run rimless_wheel.py             # why a tripod cannot simply roll
```

`render.py` renders in real time; `--gait transfer --speed 2` shows the slow,
edge-to-edge gait instead.

## Files

| File | Purpose |
|---|---|
| `tripod/params.py` | Single source of truth: dimensions, masses, motor limits, stance IK |
| `tripod/build.py` | Generates `models/tripod.xml` (MJCF), including the village |
| `tripod/kinematics.py` | Leg forward and inverse kinematics in the world frame |
| `tripod/control.py` | Whole-body balance controller (torque-controlled legs) |
| `tripod/gait.py` | The two gaits: edge-to-edge transfer, and predictive |
| `tripod/planner.py` | Model-predictive centre-of-mass planner (inverted pendulum, OSQP) |
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
* **Carrier**: an open metal basket on the back of the hood, from Book II,
  chapter I, where a fighting-machine near Kew tosses the people it catches into
  "the great metallic carrier which projected behind him", hanging there "much as
  a workman's basket hangs over his shoulder". It weighs 9 t empty and sits about
  9 m behind the hood's centre, so the rear feet carry a little more of the load.
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

## Result 3: planning the fall makes it five times faster

The transfer gait wastes most of each step parking the centre of mass at an
edge. The predictive gait (`PredictiveGait`, `planner.py`) keeps it moving. It
treats the machine as a linear inverted pendulum, c'' = ω²(c − z) with
ω = √(g/h) and h ≈ 26.5 m, where z is the zero-moment point: the point the net
ground force acts through. z must lie inside the support polygon, which is the
triangle of the feet, or, with a leg in the air, the line between the other two.
That line constraint is the hinge again: across it the centre of mass can only
fall, until the next foot lands.

Given the footstep rhythm, a quadratic programme (OSQP, 0.1 s knots, 14 s
horizon) chooses the ZMP trajectory that keeps the centre of mass moving at
walking speed. It is re-solved from the measured state at every change of
support and every 0.4 s, which makes it model-predictive control. Footholds not
yet taken are re-planned from where the body actually is, so a body that falls
behind or drifts sideways is not left reaching for feet placed on the original
timeline. The balance controller tracks the plan, with the planned pendulum
acceleration as feed-forward.

`walk_test.py`, ten cycles (30 steps, 0.8 s on three feet and 1.2 s swinging
per step): 150 m in 61 s, **5.5 mph**, maximum hood tilt **2.6°**, on two feet
59% of the time. With the carrier added (9 t further back), five cycles give
5.6 mph and 2.7°: the controller absorbs it. That is five times the transfer gait's speed, with a sixth of
its tilt: planning the fall is both faster and steadier than waiting it out.
Faster rhythms (1.0 s swings, or longer strides) manage 6–7 mph for a dozen
steps and then fall: the swing legs saturate their 40 MN·m motors, and doubling
the motor torque does not fix it, so the next limit is in the planning, not the
hardware.

## Looking round

The fighting-machines in the book halt and rear up to stand scanning the
country. Every 22 s the Martian (`martian.py`) stops at the end of a step, with
all three feet down, and raises its centre of mass 7 m. The hood goes from 30 m
to about 39 m and the thighs, roughly level while walking, slope down to the
knees at about 30°. It slews the hood 25° to each side, settles back to walking
height, and a fresh gait takes over from wherever the feet are, starting with
the rearmost foot. It has to stop to do this: raised that high, the legs are at
about 90% of their reach and cannot take a full stride. Over 75 s (two walks and
two looks) the hood tilts at most 3.1°. `Martian(look_every=0)` turns it off.

## Result 4: the Heat-Ray

The beam is a ray cast from the projector along its axis (`mj_ray`), so it is
stopped by the first thing in the village it meets; it passes through the
machine itself, whose geoms are in a group the ray ignores. Wells calls it an
invisible sword of heat, so the beam is drawn only as a faint shimmer; what you
see is the flash where it lands, then flames. Anything it rests on heats up,
catches after 1.5 s of beam, spreads fire to the rest of the same building or
tree, burns, and is left charred. The Martian (`martian.py`) burns the nearest
target within 80 m and holds fire whenever the hood tilts more than 3°.

## Next steps

1. Choose footholds and step timing inside the planner too (they are fixed by a
   rhythm now), which is where the remaining speed limit seems to be.
2. Model the swing legs' momentum in the planner: at 24 t each they are not the
   massless sticks the inverted pendulum assumes.
3. Bake the joint trajectories onto a Blender armature built from the same
   `params.py`, for a proper render with smoke and fire.
