"""A Technic orrery in MuJoCo: Sun, Earth and Moon driven by real gear trains.

The mechanism (all axles vertical):

  crank   turns the central shaft once per sidereal day (one turn of the Earth
          relative to the stars).
  year    a fixed gear train coiled round the centre reduces the central shaft to
          the hub of the year arm, by ~366.2564 (sidereal days in a year).
  spin    on the arm, a 1:1 chain of gears carries the central shaft's turning out to
          the Earth. Its axle turns relative to the arm by (shaft - arm), so absolutely
          the Earth turns with the shaft: once per sidereal day. Relative to the Sun
          (the arm) it turns once per solar day without any gear for it.
  moon    a second train on the arm reduces (shaft - arm) by ~29.5306 to the Moon's
          carrier, so the Moon goes round once per synodic month relative to the
          Earth-Sun line, and once per sidereal month relative to the stars.

Every mesh is a constraint between the turning of two axles, written as a fixed
tendon (a linear combination of joint angles) held at zero, so a mesh between a
gear on the moving arm and one on the fixed frame is expressed exactly:
    N_a (theta_a - theta_carrier_a) + N_b (theta_b - theta_carrier_b) = 0.
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .gears import ratio

MM = 0.001
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# the trains chosen in the post (driver teeth, driven teeth) per stage
YEAR_TRAIN = ((16, 24), (16, 40), (16, 40), (16, 40), (16, 40), (16, 40), (16, 40))   # 46875/128
MOON_TRAIN = ((16, 24), (16, 24), (16, 24), (16, 24), (16, 40), (12, 28))             # 945/32
SPIN_CHAIN = (24, 40, 24, 40, 24)                                                      # 1:1, four meshes

COL = {"lbg": (0.63, 0.65, 0.66), "dbg": (0.35, 0.37, 0.39), "black": (0.10, 0.10, 0.11),
       "tan": (0.80, 0.70, 0.50), "dtan": (0.56, 0.48, 0.34), "red": (0.70, 0.10, 0.06),
       "yellow": (1.0, 0.80, 0.20), "blue": (0.10, 0.35, 0.80), "white": (0.95, 0.95, 0.93)}
GEAR_COL = {8: "dbg", 12: "black", 16: "lbg", 20: "tan", 24: "lbg", 28: "dtan", 36: "black", 40: "lbg"}

LAYER = 8.0          # mm between gear layers
FACE = 6.0           # gear face width, mm


# ------------------------------------------------------------------ meshes
def gear_mesh(teeth: int):
    """Module-1 gear: pitch diameter = teeth mm, with a cross-axle hole and a few holes."""
    import shapely.geometry as sg
    import shapely.ops as so
    import trimesh
    m = 1.0
    pr = teeth * m / 2
    ra, rr = pr + m, pr - 1.25 * m
    pts = []
    for k in range(teeth):
        a0, da = 2 * math.pi * k / teeth, 2 * math.pi / teeth
        for frac, r in ((0.0, rr), (0.15, rr), (0.32, ra), (0.52, ra), (0.69, rr), (1.0, rr)):
            pts.append((r * math.cos(a0 + frac * da), r * math.sin(a0 + frac * da)))
    g = sg.Polygon(pts).buffer(0)
    cross = so.unary_union([sg.box(-2.4, -0.9, 2.4, 0.9), sg.box(-0.9, -2.4, 0.9, 2.4)])
    cut = [cross]
    if teeth >= 24:
        rh = (rr + 4) / 2
        n = 4 if teeth < 40 else 6
        cut += [sg.Point(rh * math.cos(2 * math.pi * i / n + 0.4), rh * math.sin(2 * math.pi * i / n + 0.4))
                .buffer(2.4 if teeth < 40 else 3.2, quad_segs=6) for i in range(n)]
    mesh = trimesh.creation.extrude_polygon(g.difference(so.unary_union(cut)), FACE)
    mesh.apply_translation([0, 0, -FACE / 2])
    mesh.apply_scale(MM)
    return mesh


def liftarm_mesh(holes: int, thick: float = 4.0):
    """Thin Technic liftarm: rounded ends, round holes on the 8 mm pitch."""
    import shapely.geometry as sg
    import shapely.ops as so
    import trimesh
    L = (holes - 1) * 8.0
    outline = sg.LineString([(0, 0), (max(L, 1e-3), 0)]).buffer(3.6, quad_segs=8)
    hole = so.unary_union([sg.Point(8.0 * i, 0).buffer(2.4, quad_segs=6) for i in range(holes)])
    mesh = trimesh.creation.extrude_polygon(outline.difference(hole), thick)
    mesh.apply_translation([0, 0, -thick / 2])
    mesh.apply_scale(MM)
    return mesh


# ------------------------------------------------------------------ layout
def cyclic_polygon(sides):
    """Vertices of a convex polygon inscribed in a circle with the given side lengths,
    starting at angle 0. Used to coil the year train round the centre so that it starts
    on the central shaft and ends back on the coaxial hub of the year arm."""
    sides = np.asarray(sides, float)
    lo, hi = sides.max() / 2 + 1e-9, sides.sum()
    for _ in range(200):
        R = (lo + hi) / 2
        total = (2 * np.arcsin(np.clip(sides / (2 * R), -1, 1))).sum()
        lo, hi = (R, hi) if total > 2 * math.pi else (lo, R)
    R = (lo + hi) / 2
    angles = np.concatenate([[0], np.cumsum(2 * np.arcsin(sides / (2 * R)))])
    return R, angles


def centre_distance(a, b):
    return (a + b) / 2.0          # mm, module 1


# ------------------------------------------------------------------ model
def f(*v):
    return " ".join(f"{x:.6g}" for x in v)


class Builder:
    def __init__(self):
        self.meshes = {}
        self.tendons = []          # (name, [(joint, coef), ...])
        self.joint_eq = []         # (joint1, joint2, coef): q1 = coef * q2

    def gear(self, body, teeth, z, name=None):
        mname = f"gear{teeth}"
        if mname not in self.meshes:
            self.meshes[mname] = lambda t=teeth: gear_mesh(t)
        g = ET.SubElement(body, "geom", type="mesh", mesh=mname, pos=f(0, 0, z * MM),
                          rgba=f(*COL[GEAR_COL[teeth]], 1), group="1", contype="0", conaffinity="0",
                          density="1050")
        if name:
            g.set("name", name)

    def axle(self, body, z0, z1, colour="black"):
        r = 2.4 * MM
        ET.SubElement(body, "geom", type="box", size=f(r, 0.9 * MM, (z1 - z0) / 2 * MM), pos=f(0, 0, (z0 + z1) / 2 * MM),
                      rgba=f(*COL[colour], 1), group="1", contype="0", conaffinity="0", density="1050")
        ET.SubElement(body, "geom", type="box", size=f(0.9 * MM, r, (z1 - z0) / 2 * MM), pos=f(0, 0, (z0 + z1) / 2 * MM),
                      rgba=f(*COL[colour], 1), group="1", contype="0", conaffinity="0", density="1050")

    def liftarm(self, body, p0, p1, z, colour="dbg", thick=4.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = int(round(np.linalg.norm(p1 - p0) / 8.0)) + 1
        mname = f"liftarm{n}_{thick:g}"
        if mname not in self.meshes:
            self.meshes[mname] = lambda n=n, t=thick: liftarm_mesh(n, t)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        ET.SubElement(body, "geom", type="mesh", mesh=mname, pos=f(p0[0] * MM, p0[1] * MM, z * MM),
                      euler=f(0, 0, ang), rgba=f(*COL[colour], 1), group="1", contype="0", conaffinity="0",
                      density="1050")

    def mesh_constraint(self, name, terms):
        self.tendons.append((name, terms))


def build(year=YEAR_TRAIN, moon=MOON_TRAIN, spin=SPIN_CHAIN, idler=True) -> tuple[str, dict]:
    """MJCF for the orrery, and a dict of design facts (ratios, layout)."""
    B = Builder()
    root = ET.Element("mujoco", model="technic_orrery")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(ASSETS), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 0")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080", fovy="35")
    ET.SubElement(vis, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(vis, "headlight", ambient="0.10 0.10 0.11", diffuse="0.10 0.10 0.10", specular="0.02 0.02 0.02")
    ET.SubElement(vis, "map", znear="0.005", shadowclip="0.8")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.10 0.11 0.16",
                  rgb2="0.01 0.01 0.03", width="256", height="256")
    ET.SubElement(asset, "texture", name="earth", type="cube", builtin="checker", rgb1="0.12 0.35 0.78",
                  rgb2="0.20 0.55 0.25", width="64", height="64")
    ET.SubElement(asset, "material", name="earth", texture="earth", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="moon", rgba="0.78 0.78 0.76 1", specular="0.05")
    ET.SubElement(asset, "material", name="sun", rgba="1 0.82 0.25 1", emission="1")
    ET.SubElement(asset, "material", name="table", rgba="0.22 0.19 0.16 1", specular="0.15", reflectance="0")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="room", directional="true", dir="-0.3 -0.4 -1", diffuse="0.40 0.40 0.42",
                  specular="0.25 0.25 0.25", castshadow="true")
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 0.01", material="table", group="0")

    # ---- the year train, coiled round the centre on the base
    # shaft -> a1 -> ... -> a_(n-1) -> [idler] -> hub. The idler (a plain 24-tooth gear) adds a
    # mesh without changing the ratio, so the arm turns the same way as the shaft.
    drivers = [a for a, _ in year]                     # driver of stage i on vertex i-1
    drivens = [b for _, b in year[:-1]]                # driven gears on the intermediate axles
    hub_teeth = year[-1][1]
    sides = [centre_distance(drivers[i], year[i][1]) for i in range(len(year) - 1)]
    if idler:
        drivens.append(24)
        sides += [centre_distance(drivers[-1], 24), centre_distance(24, hub_teeth)]
    else:
        sides.append(centre_distance(drivers[-1], hub_teeth))
    R, ang = cyclic_polygon(sides)
    circ_c = np.array([-R, 0.0])                       # circle centre so that vertex 0 sits at the origin
    verts = [circ_c + R * np.array([math.cos(a), math.sin(a)]) for a in ang]   # verts[0] == verts[-1] == origin

    base = ET.SubElement(world, "body", name="base")
    z_base = 12.0
    # a frame of liftarms under the train, on four posts
    span = 2 * R + 30
    for y in (-span / 2, span / 2):
        B.liftarm(base, (-2 * R - 15, y), (15, y), z_base, "lbg", thick=7.2)
    for x in (-2 * R - 15, 15):
        B.liftarm(base, (x, -span / 2), (x, span / 2), z_base + 7.2, "lbg", thick=7.2)
    for x in (-2 * R - 15, 15):
        for y in (-span / 2, span / 2):
            ET.SubElement(base, "geom", type="box", size=f(3.6 * MM, 3.6 * MM, z_base / 2 * MM),
                          pos=f(x * MM, y * MM, z_base / 2 * MM), rgba=f(*COL["dbg"], 1), group="1",
                          contype="0", conaffinity="0")
    z0 = z_base + 20.0                                  # first gear layer

    # crank / central shaft (absolute)
    shaft = ET.SubElement(base, "body", name="shaft", pos="0 0 0")
    ET.SubElement(shaft, "joint", name="shaft", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")
    B.axle(shaft, z_base - 6, 230)
    B.gear(shaft, drivers[0], z0)
    # crank handle under the base
    ET.SubElement(shaft, "geom", type="box", size=f(16 * MM, 2 * MM, 2 * MM), pos=f(16 * MM, 0, (z_base - 6) * MM),
                  rgba=f(*COL["red"], 1), group="1", contype="0", conaffinity="0")
    ET.SubElement(shaft, "geom", type="cylinder", size=f(2.5 * MM, 6 * MM), pos=f(30 * MM, 0, (z_base - 12) * MM),
                  rgba=f(*COL["red"], 1), group="1", contype="0", conaffinity="0")

    joints_year = ["shaft"]
    n_axles = len(sides) - 1                            # intermediate vertices
    for i in range(1, n_axles + 1):
        p = verts[i]
        b = ET.SubElement(base, "body", name=f"y{i}", pos=f(p[0] * MM, p[1] * MM, 0))
        ET.SubElement(b, "joint", name=f"y{i}", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")
        zl = z0 + (i - 1) * LAYER
        B.axle(b, z_base, zl + LAYER + 4)
        is_idler = idler and i == n_axles
        B.gear(b, drivens[i - 1], zl)                   # driven by the previous axle, on its layer
        if not is_idler:
            B.gear(b, drivers[i], zl + LAYER)
        joints_year.append(f"y{i}")
        # mesh: the previous axle's driver with this axle's gear (an idler meshes on its own layer)
        B.mesh_constraint(f"mesh_y{i}", [(joints_year[-2], drivers[i - 1]), (f"y{i}", drivens[i - 1])])
    # the year arm's hub (absolute), meshing the last axle (or the idler)
    z_hub = z0 + (n_axles - 1 if idler else n_axles) * LAYER
    arm = ET.SubElement(base, "body", name="arm", pos=f(0, 0, 0))
    ET.SubElement(arm, "joint", name="arm", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")
    B.gear(arm, hub_teeth, z_hub)
    last = joints_year[-1]
    B.mesh_constraint("mesh_hub", [(last, 24 if idler else drivers[-1]), ("arm", hub_teeth)])
    if idler:                                          # the idler sits on the hub's layer
        idl = world.find(".//body[@name='%s']" % last)
        for g in idl.findall("geom"):
            if g.get("mesh", "").startswith("gear"):
                g.set("pos", f(0, 0, z_hub * MM))

    # ---- the arm: two liftarm rails out to the Earth
    L_moon = sum(centre_distance(a, b) for a, b in moon)
    L_spin = sum(centre_distance(spin[i], spin[i + 1]) for i in range(len(spin) - 1))
    L = L_moon
    assert abs(L_moon - L_spin) < 1e-9, (L_moon, L_spin)
    z_arm = z_hub + 6
    for y in (-9.0, 9.0):
        B.liftarm(arm, (-8, y), (L + 8, y), z_arm, "dbg")
    for x in (0.0, L):
        B.liftarm(arm, (x, -9), (x, 9), z_arm + 4, "dbg")

    # spin chain (layer 1 above the arm) and moon train (layers above that)
    zs = z_arm + 10
    B.gear(shaft, spin[0], zs)
    x = 0.0
    prev = ("shaft", spin[0])
    for i in range(1, len(spin)):
        x += centre_distance(spin[i - 1], spin[i])
        b = ET.SubElement(arm, "body", name=f"s{i}", pos=f(x * MM, 0, 0))
        ET.SubElement(b, "joint", name=f"s{i}", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")
        B.axle(b, z_arm, zs + 4)
        B.gear(b, spin[i], zs)
        # relative to the arm: N_prev (q_prev - q_arm if prev is the shaft) + N_i q_i = 0
        terms = [(f"s{i}", spin[i])]
        if prev[0] == "shaft":
            terms += [("shaft", prev[1]), ("arm", -prev[1])]
        else:
            terms += [(prev[0], prev[1])]
        B.mesh_constraint(f"mesh_s{i}", terms)
        prev = (f"s{i}", spin[i])
    earth_axle = prev[0]

    zm = zs + LAYER
    B.gear(shaft, moon[0][0], zm)
    x = 0.0
    prev = ("shaft", moon[0][0])
    for i, (a, b_) in enumerate(moon, start=1):
        x += centre_distance(a, b_)
        zl = zm + (i - 1) * LAYER
        b = ET.SubElement(arm, "body", name=f"m{i}", pos=f(x * MM, 0, 0))
        ET.SubElement(b, "joint", name=f"m{i}", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")
        B.axle(b, zs + 4 if i < len(moon) else zs + 4, zl + LAYER + 4)
        B.gear(b, b_, zl)
        if i < len(moon):
            B.gear(b, moon[i][0], zl + LAYER)
        terms = [(f"m{i}", b_)]
        if prev[0] == "shaft":
            terms += [("shaft", a), ("arm", -a)]
        else:
            terms += [(prev[0], a)]
        B.mesh_constraint(f"mesh_m{i}", terms)
        prev = (f"m{i}", b_)
    moon_axle = prev[0]
    z_top = zm + len(moon) * LAYER

    # ---- the Sun on the shaft's top (it turns with the shaft, as the real Sun turns), Earth, Moon
    sun_r, earth_r, moon_r, moon_arm = 34.0, 13.0, 4.5, 34.0
    sun_z = z_top + 55
    ET.SubElement(base, "geom", name="sun", type="sphere", size=f(sun_r * MM), pos=f(0, 0, sun_z * MM),
                  material="sun", group="1", contype="0", conaffinity="0")
    ET.SubElement(base, "light", name="sunlight", pos=f(0, 0, sun_z * MM), directional="false",
                  diffuse="1.05 1.0 0.9", specular="0.4 0.4 0.4", attenuation="1 0 0", castshadow="false",
                  cutoff="180", exponent="0")       # no shadows: the light sits inside the Sun's sphere
    # MuJoCo lights are spotlights, so the Sun's light on the Earth and Moon is a spotlight riding
    # on the year arm, at the Sun's surface, aimed along the arm: it always faces the Earth.
    ET.SubElement(arm, "light", name="sun_on_earth", pos=f((sun_r + 1) * MM, 0, sun_z * MM), dir="1 0 0",
                  directional="false", diffuse="1.5 1.45 1.3", specular="0.5 0.5 0.5", cutoff="22", exponent="2",
                  castshadow="true")
    # Earth: on the spin chain's last axle, raised on a column to the Sun's height
    eb = ET.SubElement(world.find(".//body[@name='%s']" % earth_axle), "body", name="earth", pos=f(0, 0, 0))
    B.axle(eb, zs + 4, sun_z - earth_r)
    ET.SubElement(eb, "geom", name="earth", type="sphere", size=f(earth_r * MM), pos=f(0, 0, sun_z * MM),
                  material="earth", group="1", contype="0", conaffinity="0")
    # Moon: carrier on the moon train's last axle, which is coaxial with the Earth's
    mb = world.find(".//body[@name='%s']" % moon_axle)
    B.liftarm(mb, (0, 0), (moon_arm, 0), z_top + 2, "lbg")
    ET.SubElement(mb, "geom", type="cylinder", size=f(1.6 * MM, (sun_z - z_top) / 2 * MM),
                  pos=f(moon_arm * MM, 0, (z_top + sun_z) / 2 * MM), rgba=f(*COL["black"], 1), group="1",
                  contype="0", conaffinity="0")
    ET.SubElement(mb, "geom", name="moon", type="sphere", size=f(moon_r * MM), pos=f(moon_arm * MM, 0, sun_z * MM),
                  material="moon", group="1", contype="0", conaffinity="0")
    ET.SubElement(mb, "site", name="moon", pos=f(moon_arm * MM, 0, sun_z * MM), size="0.001", group="4")

    # cameras: overview, and one riding on the arm behind the Earth
    look = lambda name, pos, tgt, fovy, parent=world: ET.SubElement(parent, "camera", name=name, pos=f(*pos),
                                                                    xyaxes=f(*_xyaxes(pos, tgt)), fovy=str(fovy))
    look("overview", (0.36, -0.40, 0.40), (0.02, 0.01, 0.14), 38)
    # riding on the arm, beside the Earth, so the Earth shows its day and night halves
    cam_pos = np.array([(L + 30) * MM, -165 * MM, (sun_z + 40) * MM])
    look("earthcam", tuple(cam_pos), ((L + 5) * MM, 0, (sun_z - 8) * MM), 34, parent=arm)
    ET.SubElement(world, "camera", name="top", pos=f(0, 0, 0.85), xyaxes="1 0 0 0 1 0", fovy="35")

    # ---- the meshes as fixed tendons held at zero
    ten = ET.SubElement(root, "tendon")
    eq = ET.SubElement(root, "equality")
    for name, terms in B.tendons:
        t = ET.SubElement(ten, "fixed", name=name)
        merged = {}
        for j, c in terms:
            merged[j] = merged.get(j, 0) + c
        for j, c in merged.items():
            if c:
                ET.SubElement(t, "joint", joint=j, coef=str(c))
        ET.SubElement(eq, "tendon", tendon1=name, polycoef="0 0 0 0 0", solref="0.004 1", solimp="0.99 0.999 0.001")
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "velocity", name="crank", joint="shaft", kv="0.05", ctrlrange="-100 100")

    # assets
    ASSETS.mkdir(exist_ok=True)
    for name, maker in B.meshes.items():
        path = ASSETS / f"{name}.stl"
        if not path.exists():
            maker().export(path)
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl", inertia="exact")

    year_ratio = ratio(year)
    moon_ratio = ratio(moon)
    facts = dict(year_ratio=year_ratio, moon_ratio=moon_ratio, R_coil=R, arm_length=L, earth_axle=earth_axle,
                 moon_axle=moon_axle, n_gears=sum(1 for e in root.iter("geom") if e.get("mesh", "").startswith("gear")),
                 sun_z=sun_z)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode"), facts


def _xyaxes(pos, target):
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return (*right, *up)


def sunlight_only(m, on: bool = True):
    """Turn the room light off (and the headlight down) so only the Sun lights the scene."""
    i = m.light("room").id
    if on:
        m.light_diffuse[i] = 0
        m.light_specular[i] = 0
        m.vis.headlight.ambient[:] = 0.04
        m.vis.headlight.diffuse[:] = 0.0


def load(**kw):
    xml, facts = build(**kw)
    return mujoco.MjModel.from_xml_string(xml), facts


def angles(m, d, facts):
    """Absolute angles (rad) of the shaft (Earth's sidereal turning), arm, Earth and Moon carrier."""
    q = lambda j: d.qpos[m.jnt_qposadr[m.joint(j).id]]
    arm = q("arm")
    return dict(shaft=q("shaft"), arm=arm, earth=arm + q(facts["earth_axle"]), moon=arm + q(facts["moon_axle"]))
