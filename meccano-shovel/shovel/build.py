"""Assemble the Meccano steam shovel as MJCF.

Frames: x forward (toward the bucket), y to the machine's left, z up; metres.
Dimensions were taken from the two photographs, scaled so that the 1/2-inch
hole pitch matches, and then snapped to the hole grid wherever a real builder
would have had to use a whole number of holes.

Moving parts (all real joints, so MuJoCo works out the mechanics):
  slew      superstructure turning on the base             hinge z, position servo
  shipper   dipper stick rocking on the shipper shaft      hinge y, free
  crowd     stick sliding through its saddle               slide,   position servo
  hoist     rope from the drum, over the boom-head sheave,
            down to the pulley block on the bucket bail    tendon,  pull-only servo
  door      bucket door, held shut by the trip catch       hinge y, position servo
  block     pulley block swinging on the bail              hinge y, free
Decorative shafts (drum, gears, flywheel, sprockets, motor) carry hinge joints
that the animation turns in step with the machine; they carry no load.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from . import parts as pt

P = pt.P
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# ---------------------------------------------------------------- colours (from the photographs)
MATERIALS = {
    #           rgba                        specular shininess reflectance
    "green":   ((0.24, 0.43, 0.25, 1), 0.45, 0.55, 0.0),   # Meccano mid green enamel
    "red":     ((0.72, 0.16, 0.11, 1), 0.45, 0.55, 0.0),   # Meccano red enamel
    "brass":   ((0.80, 0.62, 0.30, 1), 0.85, 0.80, 0.05),
    "cream":   ((0.87, 0.80, 0.62, 1), 0.50, 0.50, 0.0),   # the boom-head pulley and motor end
    "nickel":  ((0.74, 0.74, 0.76, 1), 0.90, 0.85, 0.08),
    "steel":   ((0.62, 0.63, 0.66, 1), 0.80, 0.70, 0.05),
    "black":   ((0.07, 0.07, 0.07, 1), 0.55, 0.60, 0.0),
    "string":  ((0.32, 0.30, 0.26, 1), 0.10, 0.10, 0.0),
    "wire":    ((0.15, 0.13, 0.12, 1), 0.30, 0.30, 0.0),
    "floor":   ((0.97, 0.96, 0.94, 1), 0.05, 0.10, 0.0),
    "gravel":  ((0.62, 0.56, 0.47, 1), 0.10, 0.20, 0.0),
    "ghost":   ((0.2, 0.6, 1.0, 0.0), 0.0, 0.0, 0.0),
}
DENSITY = {"green": 7800, "red": 7800, "nickel": 7800, "steel": 7800, "black": 7800,
           "brass": 8500, "cream": 2500, "string": 1200, "wire": 1200}

# ---------------------------------------------------------------- key dimensions
BASE_HALF = 7 * P            # base frame corners
BASE_H = 0.052               # base height, floor to turntable
TURN_Z = 0.054               # slewing plane
DECK_HW = 4.5 * P            # deck side girders (corner line)
DECK_FRONT, DECK_REAR = 9 * P, -20 * P
DT = 0.0127 + pt.T / 2       # deck top, above the slewing plane
HOUSE_Y = 3 * P              # machinery side plates
BOOM_FOOT = np.array([8.3 * P, 0, 0.030])
BOOM_ANGLE = math.radians(53.8)
BOOM_U = np.array([math.cos(BOOM_ANGLE), 0, math.sin(BOOM_ANGLE)])
BOOM_HOLES = 31
BOOM_HW = 2 * P              # boom side girders (corner line), holes at +-1.5P
BOOM_TIP = BOOM_FOOT + (BOOM_HOLES - 1) * P * BOOM_U
SHIPPER = BOOM_FOOT + 14 * P * BOOM_U
APEX = np.array([5 * P, 0, 0.243])
STICK_ANGLE = math.radians(55)   # below horizontal, in the photographed pose
STICK_HW = 0.031
STICK_BACK, STICK_END = -11 * P, 12 * P
BUCKET_TILT = math.radians(-30)  # bucket axis relative to the stick
BKT = 6 * P                      # bucket inside: 6 x 6 x 6 holes
SHEAVE_R = 0.019


# ---------------------------------------------------------------- helpers
def f(*v) -> str:
    return " ".join(f"{x:.6g}" for x in (np.ravel(v) if len(v) == 1 else v))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def quat_from_axes(x, zhint):
    x = unit(x)
    z = np.asarray(zhint, float) - np.dot(zhint, x) * x
    z = unit(z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def quat_axis_angle(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), ang)
    return q


class Kit:
    """Holds the mesh library and the MJCF tree; places parts on bodies."""

    def __init__(self):
        self.meshes: dict[str, object] = {}
        self.n_parts = 0
        self.n_nuts = 0

    def mesh(self, name, maker):
        if name not in self.meshes:
            self.meshes[name] = maker()
        return name

    # -- generic placement
    def put(self, body, mesh, pos, x, zhint, mat, name=None):
        g = ET.SubElement(body, "geom", type="mesh", mesh=mesh, material=mat, pos=f(pos),
                          quat=f(quat_from_axes(x, zhint)), group="1", contype="0", conaffinity="0",
                          density=str(DENSITY.get(mat, 7800)))
        if name:
            g.set("name", name)
        self.n_parts += 1
        return g

    def nut(self, body, pos, normal, kind="nut"):
        """A nut (or round bolt head) sitting on a face whose outward normal is given."""
        normal = unit(normal)
        x = np.cross(normal, [0, 0, 1]) if abs(normal[2]) < 0.9 else np.cross(normal, [1, 0, 0])
        mesh = self.mesh("nut", pt.nut) if kind == "nut" else self.mesh("bolt", pt.bolt_head)
        self.put(body, mesh, pos, x, normal, "nickel")
        self.n_nuts += 1

    # -- Meccano members
    def strip(self, body, p0, p1, normal, mat="green", n=None, nuts=True):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = np.linalg.norm(p1 - p0)
        n = n or int(round(d / P)) + 1
        x = unit(p1 - p0)
        self.put(body, self.mesh(f"strip_{n}", lambda: pt.strip(n)), p0, x, normal, mat)
        if nuts:
            for p in (p0, p0 + (n - 1) * P * x):
                self.nut(body, p + unit(normal) * pt.T / 2, normal, "bolt")
        return n

    def girder(self, body, p0, p1, ya, mat="green", zb=None, n=None, nuts=True):
        """Angle girder from corner-line point p0 toward p1; round-hole flange toward ya.
        zb, if given, chooses which way the slotted flange points (by reversing the girder)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        x = unit(p1 - p0)
        ya = unit(np.asarray(ya, float) - np.dot(ya, x) * x)
        z = np.cross(x, ya)
        if zb is not None and np.dot(z, zb) < 0:
            p0, p1 = p1, p0
            x = -x
            z = -z
        d = np.linalg.norm(p1 - p0)
        n = n or int(round(d / P)) + 1
        self.put(body, self.mesh(f"girder_{n}", lambda: pt.angle_girder(n)), p0, x, z, mat)
        if nuts:
            for k in (0, n - 1):
                hole = p0 + k * P * x + ya * P / 2
                self.nut(body, hole + z * (pt.T + 0.0002), z, "bolt")
        return n

    def plate(self, body, origin, xa, ya, a, b, mat="red", flanged=False, nuts=True):
        """Flat (or flanged) plate a x b holes; hole (0,0) at origin, a-holes along xa, b along ya."""
        xa, ya = unit(xa), unit(ya)
        z = np.cross(xa, ya)
        name = f"{'fplate' if flanged else 'plate'}_{a}x{b}"
        maker = (lambda: pt.flanged_plate(a, b)) if flanged else (lambda: pt.flat_plate(a, b))
        self.put(body, self.mesh(name, maker), origin, xa, z, mat)
        if nuts:
            for i, j in ((0, 0), (a - 1, 0), (0, b - 1), (a - 1, b - 1)):
                self.nut(body, np.asarray(origin) + i * P * xa + j * P * ya + z * pt.T, z, "bolt")

    def flat_girder(self, body, p0, p1, up, normal, mat="green"):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = int(round(np.linalg.norm(p1 - p0) / P)) + 1
        x = unit(p1 - p0)
        up = unit(np.asarray(up, float))
        # local y of the mesh must be 'up': choose z = x cross up
        z = np.cross(x, up)
        if np.dot(z, normal) < 0:
            p0, p1, x = p1, p0, -x
            z = np.cross(x, up)
        self.put(body, self.mesh(f"fgirder_{n}", lambda: pt.flat_girder(n)), p0, x, z, mat)

    def rod(self, body, p0, p1, r=0.002, mat="steel", name=None):
        g = ET.SubElement(body, "geom", type="cylinder", fromto=f(*p0, *p1), size=f(r), material=mat,
                          group="1", contype="0", conaffinity="0", density=str(DENSITY.get(mat, 7800)))
        if name:
            g.set("name", name)
        return g

    def wheel(self, body, mesh, pos, axis, mat, spin_hint=(0, 0, 1)):
        """A round part (gear, pulley, sprocket) centred at pos with its axle along axis."""
        axis = unit(axis)
        x = np.cross(axis, spin_hint)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(axis, [1, 0, 0])
        return self.put(body, mesh, pos, x, axis, mat)

    def collar(self, body, pos, axis, mat="nickel"):
        axis = unit(axis)
        self.rod(body, np.asarray(pos) - axis * 0.0028, np.asarray(pos) + axis * 0.0028, r=0.0042, mat=mat)


def look(world, name, pos, target, fovy):
    """A fixed camera at pos looking at target, with z up."""
    fwd = unit(np.asarray(target, float) - pos)
    right = unit(np.cross(fwd, [0, 0, 1]))
    up = np.cross(right, fwd)
    ET.SubElement(world, "camera", name=name, pos=f(*pos), xyaxes=f(*right, *up), fovy=str(fovy))


def collider(body, name, size, pos, quat=None, kind="box", **kw):
    g = ET.SubElement(body, "geom", name=name, type=kind, size=f(*size), pos=f(pos), material="ghost",
                      group="3", contype="1", conaffinity="1", density="0", **kw)
    if quat is not None:
        g.set("quat", f(quat))
    return g


# ================================================================== the model
def build(gravel: int = 0, seed: int = 0) -> ET.Element:
    K = Kit()
    root = ET.Element("mujoco", model="meccano_steam_shovel")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(ASSETS), autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", cone="pyramidal", solver="CG",
                  iterations="60")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="2400", offheight="1600", fovy="30", azimuth="90", elevation="-12")
    ET.SubElement(vis, "quality", shadowsize="4096", offsamples="8", numslices="48", numstacks="24")
    ET.SubElement(vis, "headlight", ambient="0.40 0.40 0.39", diffuse="0.24 0.24 0.23", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "map", znear="0.005", zfar="20", shadowclip="1.2", shadowscale="0.5", haze="0")
    ET.SubElement(vis, "rgba", haze="0.95 0.95 0.94 1")
    ET.SubElement(root, "statistic", center="0.06 0 0.2", extent="0.8")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.985 0.985 0.98", rgb2="0.97 0.965 0.95", width="256", height="256")
    ET.SubElement(asset, "texture", name="paper", type="2d", builtin="flat", rgb1="1 1 1",
                  rgb2="1 1 1", width="16", height="16")
    for name, (rgba, spec, shin, refl) in MATERIALS.items():
        attrs = dict(name=name, rgba=f(*rgba), specular=str(spec), shininess=str(shin), reflectance=str(refl))
        if name == "floor":
            attrs["texture"] = "paper"
            attrs["texrepeat"] = "8 8"
        ET.SubElement(asset, "material", **attrs)

    world = ET.SubElement(root, "worldbody")
    # a soft key: four shadow-casting lights a few degrees apart, like a big studio softbox
    for k, (dx, dy) in enumerate(((0, 0), (0.12, 0), (0, 0.12), (0.12, 0.12))):
        ET.SubElement(world, "light", name=f"key{k}", directional="true", pos="1.5 1.6 2.2",
                      dir=f(*unit([-0.40 + dx, -0.50 + dy, -1])), diffuse="0.15 0.15 0.145",
                      specular="0.09 0.09 0.085", castshadow="true")
    ET.SubElement(world, "light", name="fill", directional="true", pos="-1 -1 1.5", dir=f(*unit([0.5, 0.6, -0.6])),
                  diffuse="0.22 0.22 0.23", specular="0.05 0.05 0.05", castshadow="false")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.05", material="floor",
                  contype="1", conaffinity="1", group="0", friction="0.9 0.01 0.001")
    # cameras that match the two photographs, plus a working view
    look(world, "photo_side", (0.42, 1.30, 0.42), (0.07, 0, 0.20), 23)
    look(world, "photo_front", (0.42, -0.86, 0.62), (0.03, 0, 0.17), 30)
    look(world, "work", (0.80, 0.82, 0.46), (0.20, 0.12, 0.13), 36)
    look(world, "dig", (0.60, 0.50, 0.22), (0.43, 0.0, 0.07), 42)

    base(K, world)
    sup = superstructure(K, world)
    if gravel:
        tray(K, world)
        gravel_heap(world, gravel, seed)

    # ---- tendons
    ten = ET.SubElement(root, "tendon")
    h = ET.SubElement(ten, "spatial", name="hoist", width="0.0007", material="string", group="1")
    for s in ("drum_rope", "tip_in", "block_rope", "tip_out"):
        ET.SubElement(h, "site", site=s)
    tc = ET.SubElement(ten, "spatial", name="trip_cord", width="0.0004", material="string", group="1")
    for s in ("trip_door", "trip_stick"):
        ET.SubElement(tc, "site", site=s)

    # ---- actuators
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "position", name="slew", joint="slew", kp="30", kv="6", ctrlrange="-3.2 3.2",
                  forcerange="-4 4")
    ET.SubElement(act, "position", name="crowd", joint="crowd", kp="3000", kv="60", ctrlrange="-0.07 0.06",
                  forcerange="-80 80")
    ET.SubElement(act, "position", name="hoist", tendon="hoist", kp="6000", kv="120", ctrlrange="0.3 1.2",
                  forcerange="-150 0")
    ET.SubElement(act, "position", name="door", joint="door", kp="8", kv="0.2", ctrlrange="0 1.6",
                  forcerange="-3 3")
    ET.SubElement(root, "contact")
    ET.SubElement(root, "keyframe")

    # meshes into the asset block
    ASSETS.mkdir(exist_ok=True)
    for name, m in K.meshes.items():
        path = ASSETS / f"{name}.stl"
        if not path.exists():
            m.export(path)
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl", inertia="exact")
    build.n_parts = K.n_parts
    return root


# ---------------------------------------------------------------- base
def base(K: Kit, world):
    b = ET.SubElement(world, "body", name="base", pos="0 0 0")
    a = BASE_HALF + P / 2
    lo, hi = 0.0, BASE_H - 0.0015
    corners = [(a, a), (-a, a), (-a, -a), (a, -a)]
    for i in range(4):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        inward = unit([-(x0 + x1), -(y0 + y1), 0])
        K.girder(b, (x0, y0, lo), (x1, y1, lo), inward, zb=(0, 0, 1))      # bottom ring
        K.girder(b, (x0, y0, hi), (x1, y1, hi), inward, zb=(0, 0, -1))     # top ring
        # side panel: a flat girder between the rings
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0])
        along = unit([x1 - x0, y1 - y0, 0])
        outward = -inward
        p0 = np.array([x0, y0, 0.0]) + along * P * 0.5 + outward * 0.0006 + [0, 0, P * 0.95]
        p1 = np.array([x1, y1, 0.0]) - along * P * 0.5 + outward * 0.0006 + [0, 0, P * 0.95]
        K.flat_girder(b, p0, p1, (0, 0, 1), outward)
    for (x, y) in corners:                                                   # corner posts
        K.girder(b, (x, y, lo), (x, y, hi), unit([-x, 0, 0]), zb=unit([0, -y, 0]), n=4)
    # top plate and the roller race the superstructure turns on
    K.plate(b, (BASE_HALF - P * 0.0, -(BASE_HALF - P), hi + 0.0008), (-1, 0, 0), (0, 1, 0), 15, 13, "red", nuts=False)
    K.put(b, K.mesh("race", lambda: pt.gear(133, face=0.003, boss=False, holes=8)), (0, 0, TURN_Z - 0.0016),
          (1, 0, 0), (0, 0, 1), "brass")
    for k in range(8):
        a_ = 2 * math.pi * (k + 0.5) / 8
        K.nut(b, (0.07 * math.cos(a_), 0.07 * math.sin(a_), hi + 0.0013), (0, 0, 1), "bolt")
    collider(b, "base_box", (BASE_HALF + P / 2, BASE_HALF + P / 2, BASE_H / 2), (0, 0, BASE_H / 2))


# ---------------------------------------------------------------- superstructure
def superstructure(K: Kit, world):
    s = ET.SubElement(world, "body", name="super", pos=f(0, 0, TURN_Z))
    ET.SubElement(s, "joint", name="slew", type="hinge", axis="0 0 1", damping="2", armature="0.05")
    deck(K, s)
    house(K, s)
    aframe(K, s)
    boom(K, s)
    boiler_and_motor(K, s)
    stick(K, s)
    return s


def deck(K, s):
    for side in (1, -1):
        K.girder(s, (DECK_REAR, side * DECK_HW, 0.0127), (DECK_FRONT, side * DECK_HW, 0.0127), (0, -side, 0),
                 zb=(0, 0, -1))
    for x in (DECK_FRONT + P / 2, DECK_REAR - P / 2, -6 * P, -13 * P):
        K.girder(s, (x, DECK_HW, 0.0127), (x, -DECK_HW, 0.0127), (-1 if x > 0 else 1, 0, 0), zb=(0, 0, -1),
                 nuts=False)
    # red plates: under the machinery, and at the back under boiler and motor
    K.plate(s, (DECK_FRONT, -4 * P, DT), (-1, 0, 0), (0, 1, 0), 11, 9, "red")
    K.plate(s, (-13 * P, -4 * P, DT), (-1, 0, 0), (0, 1, 0), 8, 9, "red")
    K.plate(s, (-1 * P, -4 * P, DT - 0.0003), (-1, 0, 0), (0, 1, 0), 6, 9, "green", nuts=False)


def house(K, s):
    """Machinery between two red side plates: hoist drum, gearing, crank and flywheel, crowd sprocket.

    The hoist drive is two 3:1 reductions on 2-hole centres: a 19-tooth pinion on the
    crankshaft drives a 57-tooth gear on the layshaft, whose 19-tooth pinion drives a
    57-tooth gear on the drum, so the drum turns once for nine turns of the crank."""
    zr = lambda k: DT + P / 2 + k * P
    for side in (1, -1):
        K.plate(s, (3 * P, side * HOUSE_Y, zr(0)), (-1, 0, 0), (0, 0, 1), 11, 5, "red")
        # angle brackets holding the plates to the deck
        K.girder(s, (3.5 * P, side * (HOUSE_Y + 0.0006), DT), (-7.5 * P, side * (HOUSE_Y + 0.0006), DT),
                 (0, side, 0), zb=(0, 0, 1), nuts=False)
    # top cross plate (engine bed)
    K.plate(s, (-3 * P, -2 * P, zr(4) + 0.008), (-1, 0, 0), (0, 1, 0), 5, 5, "red", flanged=True)
    # the upright red plate at the front of the machinery, carrying the engine's valve chest
    K.plate(s, (3.6 * P, -1.5 * P, zr(3)), (0, 1, 0), (0, 0, 1), 4, 5, "red")

    A1 = np.array([1 * P, 0, zr(1)])      # hoist drum
    A3 = np.array([-1 * P, 0, zr(1)])     # layshaft
    A4 = np.array([-1 * P, 0, zr(3)])     # crankshaft and flywheel
    A2 = np.array([2 * P, 0, zr(4)])      # crowd sprocket
    W = 0.064
    for name, A in (("hoist_drum", A1), ("layshaft", A3), ("crank", A4), ("crowd_drive", A2)):
        b = ET.SubElement(s, "body", name=name, pos=f(A))
        ET.SubElement(b, "joint", name=f"{name}_spin", type="hinge", axis="0 1 0", armature="0.001", damping="0.001")
        K.rod(b, (0, -W, 0), (0, W, 0))
        for yy in (HOUSE_Y + 0.004, -HOUSE_Y - 0.004):
            K.collar(b, (0, yy, 0), (0, 1, 0))
        if name == "hoist_drum":
            # a drum of wound cord between two bush wheels
            ET.SubElement(b, "geom", type="cylinder", fromto=f(0, -0.022, 0, 0, 0.022, 0), size="0.0075",
                          material="string", group="1", contype="0", conaffinity="0", density="1200")
            for yy in (-0.024, 0.024):
                K.wheel(b, K.mesh("bushwheel", lambda: pt.disc(0.035, 0.002, holes=8)), (0, yy, 0), (0, 1, 0), "brass")
            K.wheel(b, K.mesh("gear57", lambda: pt.gear(57, holes=8)), (0, HOUSE_Y + 0.009, 0), (0, 1, 0), "brass")
            ET.SubElement(b, "site", name="drum_rope", pos=f(0, 0.0, 0.0085), size="0.002", group="4")
        if name == "layshaft":
            K.wheel(b, K.mesh("pinion19", lambda: pt.gear(19)), (0, HOUSE_Y + 0.009, 0), (0, 1, 0), "brass")
            K.wheel(b, K.mesh("gear57", lambda: pt.gear(57, holes=8)), (0, -HOUSE_Y - 0.009, 0), (0, -1, 0), "brass")
        if name == "crank":
            K.wheel(b, K.mesh("pinion19", lambda: pt.gear(19)), (0, -HOUSE_Y - 0.009, 0), (0, -1, 0), "brass")
            K.wheel(b, K.mesh("flywheel", lambda: pt.pulley(0.0762, holes=8, face=0.006)), (0, -HOUSE_Y - 0.022, 0),
                    (0, -1, 0), "red")
            K.wheel(b, K.mesh("tyre", lambda: pt.tube(0.0405, 0.0372, 0.0075)), (0, -HOUSE_Y - 0.0255, 0), (0, -1, 0),
                    "black")
            # crank webs
            for yy in (-0.012, 0.012):
                K.strip(b, (0, yy, 0), (0, yy, 0.0127), (0, 1, 0), "nickel", n=2, nuts=False)
        if name == "crowd_drive":
            K.wheel(b, K.mesh("sprocket12", lambda: pt.sprocket(12)), (0, HOUSE_Y + 0.012, 0), (0, 1, 0), "brass")
            K.wheel(b, K.mesh("pulley1", lambda: pt.pulley(0.0254)), (0, -HOUSE_Y - 0.01, 0), (0, -1, 0), "brass")

    # twin brass cylinders on the engine bed, with piston rods down to the crank
    zc = zr(4) + 0.020
    for yy in (-0.014, 0.014):
        ET.SubElement(s, "geom", type="cylinder", fromto=f(0.034, yy, zc, 0.004, yy, zc), size="0.0085",
                      material="brass", group="1", contype="0", conaffinity="0", density="8500")
        for xe in (0.034, 0.004):
            K.rod(s, (xe, yy, zc), (xe + (0.002 if xe > 0.01 else -0.002), yy, zc), r=0.0098, mat="brass")
        K.rod(s, (0.004, yy, zc), (-0.012, yy, zc), r=0.0015, mat="steel")
    # crosshead guide and valve gear (a little visual business)
    K.strip(s, (0.0, -0.022, zc + 0.012), (0.0, 0.022, zc + 0.012), (0, 0, 1), "nickel", nuts=False)
    ET.SubElement(s, "geom", type="cylinder", fromto=f(0.019, 0, zc - 0.002, 0.019, 0, zc + 0.022), size="0.004",
                  material="brass", group="1", contype="0", conaffinity="0")
    # front stop for the boom foot
    for side in (1, -1):
        K.girder(s, (DECK_FRONT, side * (BOOM_HW + 0.0035), DT), (DECK_FRONT, side * (BOOM_HW + 0.0035), DT + 0.03),
                 (-1, 0, 0), zb=(0, side, 0), n=3, nuts=False)
    return A1, A2


def aframe(K, s):
    legs_foot = np.array([DECK_FRONT - P, 0, DT])
    rear_foot = np.array([DECK_REAR, 0, DT])
    for side in (1, -1):
        y = side * DECK_HW
        K.girder(s, legs_foot + [0, y, 0], APEX + [0, y, 0], (0, -side, 0), zb=(-1, 0, 0))
        K.girder(s, rear_foot + [0, y, 0], APEX + [0, y, 0], (0, -side, 0), zb=(0, 0, 1))
        # inner back-stay, as in the photograph (a second member just below the first)
        K.girder(s, rear_foot + [0.0254, y * 0.72, 0], APEX + [-0.005, y * 0.72, -0.02], (0, -side, 0),
                 zb=(0, 0, 1))
        # tie between leg and back-stay
        a = legs_foot + 0.45 * (APEX - legs_foot) + [0, y + side * 0.0012, 0]
        bb = rear_foot + 0.62 * (APEX - rear_foot) + [0, y + side * 0.0012, 0]
        K.strip(s, a, bb, (0, side, 0))
        c = legs_foot + 0.25 * (APEX - legs_foot) + [0, y + side * 0.0012, 0]
        d = rear_foot + 0.35 * (APEX - rear_foot) + [0, y + side * 0.0012, 0]
        K.strip(s, c, d, (0, side, 0))
    # apex cross shaft and a cross brace between the legs
    K.rod(s, APEX + [0.004, -DECK_HW - 0.012, 0.004], APEX + [0.004, DECK_HW + 0.012, 0.004], r=0.002)
    for side in (1, -1):
        K.collar(s, APEX + [0.004, side * (DECK_HW + 0.01), 0.004], (0, 1, 0))
    p0 = legs_foot + 0.62 * (APEX - legs_foot)
    K.strip(s, p0 + [-0.0128, -DECK_HW, 0], p0 + [-0.0128, DECK_HW, 0], (-1, 0, 0.2))
    # boom suspension: four rods from the apex shaft to the boom head
    tip = BOOM_TIP + [0, 0, 0.002]
    for yy in (-0.026, -0.016, 0.016, 0.026):
        K.rod(s, APEX + [0.004, yy, 0.004], tip + [-0.01, yy * 0.75, 0.0], r=0.0012, mat="steel")


def boom(K, s):
    F, u = BOOM_FOOT, BOOM_U
    n = BOOM_HOLES
    perp = np.array([-u[2], 0, u[0]])            # up-and-back, perpendicular to the boom
    for side in (1, -1):
        y = side * BOOM_HW
        K.girder(s, F + [0, y, 0], F + (n - 1) * P * u + [0, y, 0], (0, -side, 0), zb=-perp)
        # a lower chord, making each side a narrow lattice girder
        lo = F - perp * 0.0254 + [0, y, 0] + u * 3 * P
        K.girder(s, lo, lo + (n - 6) * P * u, (0, -side, 0), zb=perp)
        # lattice strips between the chords
        for k in range(3, n - 3, 4):
            a = F + k * P * u + [0, y + side * 0.0013, 0] - perp * 0.002
            b = F + (k + 2) * P * u - perp * 0.0254 + [0, y + side * 0.0013, 0]
            K.strip(s, a, b, (0, side, 0), n=3, nuts=False)
    for k in range(4, n - 2, 5):                   # cross strips over the top
        c = F + k * P * u + perp * 0.0012
        K.strip(s, c + [0, -1.5 * P, 0], c + [0, 1.5 * P, 0], perp, n=4)
    # foot pin, head axle and sheave
    K.rod(s, F + [0, -BOOM_HW - 0.012, 0.004], F + [0, BOOM_HW + 0.012, 0.004])
    head = BOOM_TIP - perp * 0.0064
    K.rod(s, head + [0, -BOOM_HW - 0.01, 0], head + [0, BOOM_HW + 0.01, 0])
    sb = ET.SubElement(s, "body", name="sheave", pos=f(head))
    ET.SubElement(sb, "joint", name="sheave_spin", type="hinge", axis="0 1 0", armature="0.0005", damping="0.001")
    K.wheel(sb, K.mesh("sheave", lambda: pt.pulley(2 * SHEAVE_R, holes=6, face=0.006)), (0, 0, 0), (0, 1, 0), "cream")
    for yy in (-BOOM_HW - 0.006, BOOM_HW + 0.006):
        K.collar(s, head + [0, yy, 0], (0, 1, 0))
    # where the rope leaves the sheave (on its back, toward the drum) and comes off the front
    back = unit(np.array([0.0, 0, 0]) + [-u[0], 0, -u[1] if False else -u[2]])
    ET.SubElement(s, "site", name="tip_in", pos=f(head + perp * SHEAVE_R * 0.9 + u * 0.004 + [0, 0.004, 0]),
                  size="0.002", group="4")
    ET.SubElement(s, "site", name="tip_out", pos=f(head + np.array([SHEAVE_R * 0.95, -0.004, 0])),
                  size="0.002", group="4")
    # shipper shaft with the black chain wheel, and the crowd chain
    S = SHIPPER
    K.rod(s, S + [0, -STICK_HW - 0.03, 0], S + [0, STICK_HW + 0.03, 0], r=0.002)
    sp = ET.SubElement(s, "body", name="shipper_wheel", pos=f(S))
    ET.SubElement(sp, "joint", name="shipper_spin", type="hinge", axis="0 1 0", armature="0.0005", damping="0.001")
    yw = STICK_HW + 0.02
    K.wheel(sp, K.mesh("sprocket15", lambda: pt.sprocket(15)), (0, yw, 0), (0, 1, 0), "black")
    chain(K, s, np.array([2 * P, yw, DT + P / 2 + 4 * P]), S + [0, yw, 0], 0.0095 / (2 * math.sin(math.pi / 12)),
          0.0095 / (2 * math.sin(math.pi / 15)))


def chain(K, s, c0, c1, r0, r1):
    """Sprocket chain: two straight runs tangent to both wheels (the wrap round the wheels is hidden)."""
    d = c1 - c0
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[2], 0, u[0]])
    link = K.mesh("chainlink", pt.chain_link)
    for sign in (1, -1):
        a = c0 + sign * nrm * r0
        b = c1 + sign * nrm * r1
        run = b - a
        n = int(np.linalg.norm(run) / 0.0095)
        ru = unit(run)
        for k in range(n):
            K.put(s, link, a + ru * (k * 0.0095), ru, [0, 1, 0], "steel")


def boiler_and_motor(K, s):
    bx, by = -15 * P, -0.006
    h = 0.17
    K.put(s, K.mesh("boiler", lambda: pt.boiler(0.064, h)), (bx, by, DT + 0.001), (1, 0, 0), (0, 0, 1), "red")
    K.put(s, K.mesh("chimney", lambda: pt.tube(0.0095, 0.008, 0.042)), (bx + 0.006, by, DT + h), (1, 0, 0),
          (0, 0, 1), "green")
    K.put(s, K.mesh("chimneycap", lambda: pt.tube(0.0115, 0.008, 0.004)), (bx + 0.006, by, DT + h + 0.039),
          (1, 0, 0), (0, 0, 1), "green")
    # the electric motor at the back: cream-coloured field coil between end plates
    mx, my, mz = -18 * P, 0.038, DT + 0.026
    ET.SubElement(s, "geom", type="cylinder", fromto=f(mx + 0.024, my, mz, mx - 0.024, my, mz), size="0.016",
                  material="cream", group="1", contype="0", conaffinity="0", density="2500")
    for xe, mat in ((mx + 0.027, "brass"), (mx - 0.027, "black")):
        ET.SubElement(s, "geom", type="cylinder", fromto=f(xe - 0.003, my, mz, xe + 0.003, my, mz), size="0.0175",
                      material=mat, group="1", contype="0", conaffinity="0", density="7800")
    for zz in (mz - 0.0125, mz + 0.0125):
        K.strip(s, (mx + 0.03, my - 0.018, zz), (mx - 0.03, my - 0.018, zz), (0, -1, 0), "nickel", n=6, nuts=False)
    K.plate(s, (mx + 2.5 * P, my - 1.5 * P, DT + 0.0012), (-1, 0, 0), (0, 1, 0), 5, 3, "green", flanged=True)
    # motor shaft forward to the gearing, with a worm
    K.rod(s, (mx + 0.027, my - 0.012, mz), (-7.6 * P, my - 0.012, mz), r=0.002)
    ms = ET.SubElement(s, "body", name="motor_shaft", pos=f(-9 * P, my - 0.012, mz))
    ET.SubElement(ms, "joint", name="motor_spin", type="hinge", axis="1 0 0", armature="0.0005", damping="0.001")
    K.wheel(ms, K.mesh("pulley1", lambda: pt.pulley(0.0254)), (0, 0, 0), (1, 0, 0), "brass", spin_hint=(0, 1, 0))
    # flex to the motor terminals
    ET.SubElement(s, "geom", type="capsule", fromto=f(mx - 0.03, my + 0.004, mz + 0.012, mx - 0.07, my + 0.03, 0.004),
                  size="0.0009", material="wire", group="1", contype="0", conaffinity="0")
    ET.SubElement(s, "geom", type="capsule", fromto=f(mx - 0.03, my - 0.004, mz + 0.012, mx - 0.08, my - 0.02, 0.004),
                  size="0.0009", material="wire", group="1", contype="0", conaffinity="0")


# ---------------------------------------------------------------- stick, bucket, block
def stick(K, s):
    rot = ET.SubElement(s, "body", name="stick_rot", pos=f(SHIPPER), quat=f(quat_axis_angle([0, 1, 0], STICK_ANGLE)))
    ET.SubElement(rot, "joint", name="shipper", type="hinge", axis="0 1 0", range="-1.2 1.1", damping="0.05",
                  armature="0.002")
    # saddle blocks either side of each stick member, riding on the shipper shaft
    for side in (1, -1):
        for yy in (STICK_HW - 0.003, STICK_HW + 0.0157):
            ET.SubElement(rot, "geom", type="box", size="0.016 0.0008 0.011", pos=f(0, side * yy, 0.001),
                          material="nickel", group="1", contype="0", conaffinity="0", density="7800")
        K.nut(rot, (0, side * (STICK_HW + 0.0168), 0.0), (0, side, 0))
    st = ET.SubElement(rot, "body", name="stick", pos="0 0 0")
    ET.SubElement(st, "joint", name="crowd", type="slide", axis="1 0 0", range="-0.07 0.06", damping="5",
                  armature="0.02")
    n = int(round((STICK_END - STICK_BACK) / P)) + 1
    for side in (1, -1):
        K.girder(st, (STICK_BACK, side * STICK_HW, -P / 2), (STICK_END, side * STICK_HW, -P / 2), (0, side, 0),
                 zb=(0, 0, 1), n=n)
    for side in (1, -1):
        y = side * (STICK_HW + 0.0010)
        K.flat_girder(st, (STICK_BACK + 0.5 * P, y, 0.0), (STICK_END - 9 * P, y, 0.0), (0, 0, 1), (0, side, 0), "black")
    for x in (STICK_BACK + P, STICK_BACK + 8 * P, STICK_END - 2 * P):
        K.strip(st, (x, -STICK_HW - 0.0005, P / 2), (x, STICK_HW + 0.0005, P / 2), (0, 0, 1), "green", n=6)
    ET.SubElement(st, "site", name="trip_stick", pos=f(STICK_END - 7 * P, STICK_HW + 0.001, -0.006), size="0.002",
                  group="4")
    # brass trip lever where the cord is tied
    K.wheel(st, K.mesh("pulley1", lambda: pt.pulley(0.0254)), (STICK_END - 7 * P, STICK_HW + 0.008, -0.004),
            (0, 1, 0), "brass")
    bucket(K, st)


def bucket(K, st):
    bk = ET.SubElement(st, "body", name="bucket", pos=f(STICK_END + 0.004, 0, -P / 2),
                       quat=f(quat_axis_angle([0, 1, 0], BUCKET_TILT)))
    D = BKT
    hw = 3 * P
    # sides, top and bottom: red plates 7 x 7 holes (the inside is 6 x 6 pitches)
    for side in (1, -1):
        K.plate(bk, (0, side * (hw + 0.001), -D), (1, 0, 0), (0, 0, 1), 7, 7, "red")
    K.plate(bk, (0, -hw, 0.001), (1, 0, 0), (0, 1, 0), 7, 7, "red")
    K.plate(bk, (0, -hw, -D - 0.001), (1, 0, 0), (0, 1, 0), 7, 7, "red")
    # green angle girders round the mouth and the back edges
    for side in (1, -1):
        K.girder(bk, (D + 0.006, side * (hw + 0.0015), -D - 0.004), (D + 0.006, side * (hw + 0.0015), 0.004),
                 (-1, 0, 0), zb=(0, side, 0), n=7)
        K.girder(bk, (0.001, side * (hw + 0.0015), 0.004), (D + 0.006, side * (hw + 0.0015), 0.004),
                 (0, -side, 0), zb=(0, 0, 1), n=7)
    K.girder(bk, (D + 0.006, -hw - 0.003, -D - 0.003), (D + 0.006, hw + 0.003, -D - 0.003), (-1, 0, 0),
             zb=(0, 0, -1), n=7)
    # teeth along the lip
    for k in range(6):
        y = -2.5 * P + k * P
        a = np.array([D + 0.004, y, -D - 0.002])
        K.rod(bk, a, a + [0.011, 0, -0.003], r=0.0021, mat="nickel")
        K.nut(bk, a + [0.0, 0, 0.003], (0, 0, 1))
    # bail: two strips from the mouth corners up to the pulley block pin
    pin = np.array([D * 0.62, 0, 0.034])
    for side in (1, -1):
        K.strip(bk, (D - 0.004, side * (hw + 0.004), -0.004), pin + [0, side * 0.012, 0], (0, side, 0), n=4)
    K.rod(bk, pin + [0, -0.016, 0], pin + [0, 0.016, 0], r=0.002)
    ET.SubElement(bk, "site", name="teeth", pos=f(D + 0.012, 0, -D - 0.004), size="0.003", group="4")
    ET.SubElement(bk, "site", name="bucket_mid", pos=f(D / 2, 0, -D / 2), size="0.003", group="4")
    # colliders (the door has its own)
    w = 0.0025
    collider(bk, "bkt_top", (D / 2 + 0.006, hw + w, w), (D / 2, 0, w))
    collider(bk, "bkt_bottom", (D / 2 + 0.006, hw + w, w), (D / 2, 0, -D - w))
    for side in (1, -1):
        collider(bk, f"bkt_side{side}", (D / 2 + 0.006, w, D / 2 + w), (D / 2, side * (hw + w), -D / 2))
    collider(bk, "bkt_teeth", (0.006, hw, 0.003), (D + 0.009, 0, -D - 0.003))

    # the door: hinged along the top of the back, swings out to dump
    dr = ET.SubElement(bk, "body", name="door", pos="-0.002 0 0")
    ET.SubElement(dr, "joint", name="door", type="hinge", axis="0 1 0", range="0 1.6", damping="0.02",
                  armature="0.0005")
    K.plate(dr, (-0.001, -hw, -D), (0, 1, 0), (0, 0, 1), 7, 7, "red")
    K.girder(dr, (-0.003, -hw - 0.002, -D - 0.002), (-0.003, hw + 0.002, -D - 0.002), (1, 0, 0), zb=(-1, 0, 0),
             n=7)
    # the catch the trip cord pulls
    K.strip(dr, (-0.004, hw - P, -D + 0.002), (-0.004, hw - P, -D + 3 * P), (-1, 0, 0), "nickel", n=4)
    ET.SubElement(dr, "site", name="trip_door", pos=f(-0.006, hw - P, -D + 0.004), size="0.002", group="4")
    collider(dr, "door_plate", (w, hw + w, D / 2 + w), (-w, 0, -D / 2))

    # pulley block on the bail pin
    blk = ET.SubElement(bk, "body", name="block", pos=f(pin))
    ET.SubElement(blk, "joint", name="block", type="hinge", axis="0 1 0", damping="0.002", armature="0.0002")
    K.wheel(blk, K.mesh("block_pulley", lambda: pt.pulley(0.03, holes=0, face=0.006)), (0, 0, 0.026), (0, 1, 0), "red")
    for side in (1, -1):
        K.strip(blk, (0, side * 0.006, 0), (0, side * 0.006, 0.0254), (0, side, 0), "nickel", n=3)
    K.rod(blk, (0, -0.009, 0.026), (0, 0.009, 0.026), r=0.002)
    ET.SubElement(blk, "site", name="block_rope", pos=f(0, 0, 0.026 + 0.0145), size="0.002", group="4")


# ---------------------------------------------------------------- gravel
TRAY_X0, TRAY_X1, TRAY_HW, TRAY_H = 0.40, 0.49, 0.06, 0.032


def tray(K, world):
    """A three-sided bin of Meccano plates, open toward the shovel, so the gravel stands
    as a bank face to dig into instead of spreading over the floor."""
    t = ET.SubElement(world, "body", name="tray", pos="0 0 0")
    x1, hw, h = TRAY_X1, TRAY_HW, TRAY_H
    nb = int(round(2 * hw / P)) + 1
    K.plate(t, (x1 + 0.002, -hw + P / 2, P / 2), (0, 1, 0), (0, 0, 1), nb - 1, 2, "red")
    nl = int(round((x1 - TRAY_X0) / P)) + 1
    for side in (1, -1):
        K.plate(t, (TRAY_X0 + P / 2, side * (hw + 0.002), P / 2), (1, 0, 0), (0, 0, 1), nl - 1, 2, "red")
        K.girder(t, (TRAY_X0, side * (hw + 0.003), h), (x1 + 0.004, side * (hw + 0.003), h), (0, side, 0),
                 zb=(0, 0, -1))
    K.girder(t, (x1 + 0.004, -hw - 0.004, h), (x1 + 0.004, hw + 0.004, h), (1, 0, 0), zb=(0, 0, -1))
    collider(t, "tray_back", (0.003, hw + 0.006, h / 2), (x1 + 0.005, 0, h / 2))
    for side in (1, -1):
        collider(t, f"tray_side{side}", ((x1 - TRAY_X0) / 2 + 0.006, 0.003, h / 2),
                 ((x1 + TRAY_X0) / 2, side * (hw + 0.005), h / 2))


def gravel_heap(world, n, seed, pitch=0.0135):
    """Stones: small spheres of random size, with rolling resistance so they stand in a
    heap the way gravel does rather than rolling like marbles. They start stacked in
    the tray and slump into a bank while the shovel lowers its bucket."""
    rng = np.random.default_rng(seed)
    spots = []
    layer = 0
    while len(spots) < n:
        off = (layer % 2) * pitch / 2
        z = 0.008 + layer * pitch
        x = TRAY_X0 + 0.022 + off
        while x < TRAY_X1 - 0.006:
            y = -TRAY_HW + 0.007 + off
            while y < TRAY_HW - 0.006:
                spots.append((x, y, z))
                y += pitch
            x += pitch
        layer += 1
    spots = spots[:n]
    for i, (x, y, z) in enumerate(spots):
        jitter = rng.normal(0, 0.001, 3)
        b = ET.SubElement(world, "body", name=f"g{i}", pos=f(*(np.array([x, y, z]) + jitter * [1, 1, 0])))
        ET.SubElement(b, "freejoint")
        r = rng.uniform(0.0045, 0.0062)
        shade = 0.8 + 0.4 * rng.random()
        rgba = np.clip(np.array([0.60, 0.55, 0.47]) * shade + rng.normal(0, 0.035, 3), 0, 1)
        ET.SubElement(b, "geom", type="sphere", size=f(r), rgba=f(*rgba, 1), density="2200",
                      friction="0.9 0.02 0.004", condim="6", group="2", contype="1", conaffinity="1",
                      solref="0.004 1")


# ---------------------------------------------------------------- compile
def xml_string(**kw) -> str:
    root = build(**kw)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def load(**kw) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml_string(**kw))


if __name__ == "__main__":
    import sys
    m = load()
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    (out / "shovel.xml").write_text(xml_string())
    print(f"{m.ngeom} geoms, {m.nmesh} meshes, {m.nbody} bodies, {m.njnt} joints, mass {m.body_subtreemass[0]:.2f} kg")
