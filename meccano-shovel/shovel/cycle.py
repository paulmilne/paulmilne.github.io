"""Driving the shovel: settling into the photographed pose, and a dig cycle.

The cycle is a timetable of set-points for the four real controls (slew angle,
crowd, hoist-rope length, door), blended smoothly between key times, exactly
as a driver works the levers. Everything else, the stick's angle, the bucket's
swing, the gravel, is left to the physics.

The decorative shafts are turned to match: the hoist drum and the boom-head
sheave by the rope paid out, the crowd sprockets by the crowd travel, and the
engine's crank, flywheel and the motor by how hard the machine is working.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

DRUM_R = 0.0085        # hoist drum, radius to the rope
SHEAVE_R = 0.019
SPROCKET_R = {"crowd_drive": 0.0095 / (2 * math.sin(math.pi / 12)),
              "shipper_wheel": 0.0095 / (2 * math.sin(math.pi / 15))}


def ids(m):
    a = {m.actuator(i).name: i for i in range(m.nu)}
    j = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
    return a, j


def hoist_length(m, d):
    return d.ten_length[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "hoist")]


def settle(m, d, seconds=1.5):
    """Hold the photographed pose: hoist rope taut at its as-built length, crowd and slew at zero."""
    a, j = ids(m)
    mujoco.mj_forward(m, d)
    d.ctrl[a["hoist"]] = hoist_length(m, d) - 0.002
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# key frames: (time s, slew rad, crowd m, hoist rope m, door rad, phase)
DIG = [
    (0.0,  0.0,   0.00, 0.944, 0.0, "settle"),
    (1.5,  0.0,  -0.02, 0.950, 0.0, "lower"),          # rope slack: the bucket sits on the floor
    (3.0,  0.0,  -0.02, 0.950, 0.0, "lower"),
    (5.5,  0.0,   0.06, 0.950, 0.0, "dig"),            # crowd out along the floor, into the bank
    (7.5,  0.0,   0.06, 0.840, 0.0, "dig"),            # hoist up the face, filling the bucket
    (9.0,  0.0,   0.03, 0.640, 0.0, "hoist"),          # lift clear, draw the load in a little
    (9.6,  0.0,   0.03, 0.640, 0.0, "hoist"),
    (12.6, 1.57,  0.03, 0.640, 0.0, "swing"),          # slew a quarter turn to the left
    (13.2, 1.57,  0.03, 0.640, 0.0, "swing"),
    (13.6, 1.57,  0.03, 0.640, 1.4, "dump"),           # pull the trip cord: the door swings open
    (15.6, 1.57,  0.03, 0.640, 1.4, "dump"),
    (16.4, 1.57,  0.03, 0.640, 0.0, "close"),          # door falls shut
    (19.4, 0.0,  -0.02, 0.860, 0.0, "return"),         # swing back, lowering ready for the next pass
    (20.5, 0.0,  -0.02, 0.860, 0.0, "return"),
]


class DigCycle:
    def __init__(self, m, d, keys=DIG):
        self.m, self.d, self.keys = m, d, keys
        self.a, self.j = ids(m)
        mujoco.mj_forward(m, d)
        self.L0 = hoist_length(m, d)
        self.duration = keys[-1][0]
        self.phase = keys[0][5]
        self.engine = 0.0
        self.prev = None
        self.gravel = [m.body(i).id for i in range(m.nbody) if m.body(i).name.startswith("g")]
        self.deco = {n: self.j[f"{n}_spin"] for n in ("hoist_drum", "layshaft", "crank", "crowd_drive",
                                                      "shipper_wheel", "sheave", "motor_shaft")
                     if f"{n}_spin" in self.j}
        # the motor joint is called motor_spin
        if "motor_spin" in self.j:
            self.deco["motor_shaft"] = self.j["motor_spin"]
        qadr = list(m.jnt_qposadr)
        self.dof = {k: m.jnt_dofadr[qadr.index(v)] for k, v in self.deco.items()}
        self.hoist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "hoist")

    def targets(self, t):
        ks = self.keys
        if t >= ks[-1][0]:
            k = ks[-1]
            return np.array(k[1:5]), k[5]
        for k0, k1 in zip(ks, ks[1:]):
            if k0[0] <= t < k1[0]:
                u = smooth((t - k0[0]) / (k1[0] - k0[0]))
                return (1 - u) * np.array(k0[1:5]) + u * np.array(k1[1:5]), k1[5]
        return np.array(ks[0][1:5]), ks[0][5]

    def step(self):
        m, d, a = self.m, self.d, self.a
        x, self.phase = self.targets(d.time)
        slew, crowd, hoist, door = x
        if d.time < 0.3:
            hoist = self.L0 - 0.002
        d.ctrl[a["slew"]], d.ctrl[a["crowd"]], d.ctrl[a["hoist"]], d.ctrl[a["door"]] = slew, crowd, hoist, door
        mujoco.mj_step(m, d)
        self._turn_shafts()

    def _turn_shafts(self):
        m, d, j = self.m, self.d, self.j
        L = d.ten_length[self.hoist_id]
        crowd = d.qpos[j["crowd"]]
        slew = d.qpos[j["slew"]]
        now = np.array([L, crowd, slew])
        if self.prev is not None:
            rate = np.abs(now - self.prev) / m.opt.timestep
            work = rate[0] / 0.05 + rate[1] / 0.05 + rate[2] / 0.5
            self.engine += m.opt.timestep * (6.0 + 40.0 * min(work, 3.0))   # crank, rad
        self.prev = now
        set_ = lambda name, ang: self._set(name, ang)
        drum = -(L - self.L0) / DRUM_R
        set_("hoist_drum", drum)
        set_("layshaft", -drum * 19 / 57)
        set_("sheave", (L - self.L0) / SHEAVE_R)
        set_("crowd_drive", crowd / SPROCKET_R["crowd_drive"])
        set_("shipper_wheel", crowd / SPROCKET_R["shipper_wheel"])
        set_("crank", self.engine)
        set_("motor_shaft", 3 * self.engine)

    def _set(self, name, ang):
        if name in self.deco:
            adr = self.deco[name]
            self.d.qpos[adr] = ang
            self.d.qvel[self.dof[name]] = 0

    def report(self):
        """How much gravel was moved: balls now well away from the heap, to the left."""
        pos = self.d.xpos[self.gravel]
        moved = int(((pos[:, 1] > 0.25)).sum())
        return f"{moved} of {len(self.gravel)} stones dumped"
