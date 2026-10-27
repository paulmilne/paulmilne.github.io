"""A LEGO Technic orrery of the whole solar system, in MuJoCo.

One turn of the crank is one Earth year: the Earth's arm is fixed to the crank
shaft. Each other planet has its own gear train, chosen by exhaustive search
over Technic gears (gears.reachable_mixed), coiled round the shaft on its own
tier of a gear tower: it starts with a gear on the shaft at the bottom of the
tier and ends on that planet's hub, coaxial with the shaft, at the top. The
outer planets are on the lowest tiers with the longest arms, the inner planets
on the highest with the shortest, so no planet's post ever meets another arm.

Every mesh is a fixed tendon held at zero, as in the Earth-Moon orrery. The
planets start at their mean longitudes for a given date (JPL elements), baked
into the angle at which each arm is fixed to its hub.
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from . import gears as G
from .orrery import ASSETS, COL, MM, Builder, centre_distance, cyclic_polygon, f, _xyaxes

ORDER_OUT_IN = ("Neptune", "Uranus", "Saturn", "Jupiter", "Mars", "Earth", "Venus", "Mercury")
RADIUS = {"Mercury": 70, "Venus": 100, "Earth": 135, "Mars": 170, "Jupiter": 215, "Saturn": 262,
          "Uranus": 305, "Neptune": 345}                          # arm lengths, mm (not to scale)
SIZE = {"Mercury": 5, "Venus": 8, "Earth": 8.5, "Mars": 6, "Jupiter": 17, "Saturn": 14, "Uranus": 10,
        "Neptune": 10}                                            # sphere radii, mm (not to scale)
LOOK = {"Mercury": (0.62, 0.60, 0.58), "Venus": (0.93, 0.83, 0.58), "Earth": None, "Mars": (0.80, 0.35, 0.18),
        "Jupiter": (0.85, 0.70, 0.52), "Saturn": (0.90, 0.80, 0.55), "Uranus": (0.55, 0.85, 0.88),
        "Neptune": (0.25, 0.42, 0.85)}
LAYER = 6.8
JD_START = 2461340.5          # 2026 October 27, 0h UT


def choose_trains(max_stages: int = 8, tolerance_deg: float = 2.0):
    """For each planet: the shortest train within tolerance_deg of drift per century, or else the
    shortest train that reaches the best accuracy possible with max_stages."""
    levels = G.reachable_mixed(max_stages)
    out = {}
    for p in G.PLANETS:
        if p == "Earth":
            out[p] = (Fraction(1), (), 0.0)
            continue
        x = G.planet_ratio(p)
        best = G.best_reachable(x, levels)
        drift = {k: 360 * (100 / x) * (1 / (1 + best[k][2]) - 1) for k in best if k}
        ok = [k for k in sorted(drift) if abs(drift[k]) < tolerance_deg]
        if ok:
            k = ok[0]
        else:
            top = min(abs(v) for v in drift.values())
            k = min(k for k in drift if abs(abs(drift[k]) - top) < 1e-9)
        r, st, e = best[k]
        out[p] = (r, st, drift[k])
    return out


def ring_mesh(r_in, r_out, thick=0.6):
    import shapely.geometry as sg
    import trimesh
    ring = sg.Point(0, 0).buffer(r_out, quad_segs=24).difference(sg.Point(0, 0).buffer(r_in, quad_segs=24))
    m = trimesh.creation.extrude_polygon(ring, thick)
    m.apply_translation([0, 0, -thick / 2])
    m.apply_scale(MM)
    return m


def build(trains=None, jd=JD_START):
    trains = trains or choose_trains()
    B = Builder()
    root = ET.Element("mujoco", model="technic_solar_system")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(ASSETS), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 0")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080", fovy="35")
    ET.SubElement(vis, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(vis, "headlight", ambient="0.20 0.20 0.22", diffuse="0.25 0.25 0.25", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "map", znear="0.005", shadowclip="1.6")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.10 0.11 0.16",
                  rgb2="0.01 0.01 0.03", width="256", height="256")
    ET.SubElement(asset, "texture", name="earth", type="cube", builtin="checker", rgb1="0.12 0.35 0.78",
                  rgb2="0.20 0.55 0.25", width="64", height="64")
    ET.SubElement(asset, "material", name="earth", texture="earth", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="sun", rgba="1 0.82 0.25 1", emission="1")
    ET.SubElement(asset, "material", name="table", rgba="0.22 0.19 0.16 1", specular="0.15")
    ET.SubElement(asset, "material", name="ring", rgba="0.85 0.78 0.60 0.75", specular="0.2")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="room", directional="true", dir="-0.3 -0.4 -1", diffuse="0.75 0.75 0.77",
                  specular="0.3 0.3 0.3", castshadow="true")
    ET.SubElement(world, "geom", name="table", type="plane", size="2 2 0.01", material="table", group="0")
    base = ET.SubElement(world, "body", name="base")

    shaft = ET.SubElement(base, "body", name="shaft")
    ET.SubElement(shaft, "joint", name="shaft", type="hinge", axis="0 0 1", damping="0.0001", armature="1e-5")

    z = 30.0                            # bottom of the first tier, mm
    info = {}
    hub_bodies = {}
    n_gears = 0
    for t, p in enumerate(ORDER_OUT_IN):
        r, stages, drift = trains[p]
        if p == "Earth":
            hub_bodies[p] = shaft
            info[p] = dict(z=z, ratio=r, stages=(), drift=0.0, coil_R=0.0)
            z_arm = z + 4
            z += 14
        else:
            # vertices: shaft, a1 .. a_(n-1), [idler], hub; even number of meshes keeps the direction
            n = len(stages)
            idler = n % 2 == 1
            drivers = [a for a, _ in stages]
            drivens = [b for _, b in stages[:-1]]
            hub_teeth = stages[-1][1]
            sides = [centre_distance(drivers[i], stages[i][1]) for i in range(n - 1)]
            if idler:
                idl = 24 if drivers[-1] in G.SPUR else 20
                drivens.append(idl)
                sides += [centre_distance(drivers[-1], idl), centre_distance(idl, hub_teeth)]
            else:
                sides.append(centre_distance(drivers[-1], hub_teeth))
            R, ang = cyclic_polygon(sides)
            rot = 2 * math.pi * t / len(ORDER_OUT_IN) * 3       # spread the coils round the tower
            cc = R * np.array([math.cos(rot + math.pi), math.sin(rot + math.pi)])
            verts = [cc + R * np.array([math.cos(a + rot), math.sin(a + rot)]) for a in ang]
            B.gear(shaft, drivers[0], z)
            n_gears += 1
            prev = ("shaft", drivers[0])
            n_axles = len(sides) - 1
            for i in range(1, n_axles + 1):
                v = verts[i]
                is_idler = idler and i == n_axles
                zl = z + (i - 1) * LAYER
                b = ET.SubElement(base, "body", name=f"{p}{i}", pos=f(v[0] * MM, v[1] * MM, 0))
                ET.SubElement(b, "joint", name=f"{p}{i}", type="hinge", axis="0 0 1", damping="0.0001",
                              armature="1e-5")
                B.axle(b, zl - 4, zl + (0 if is_idler else LAYER) + 4)
                B.gear(b, drivens[i - 1], zl)
                n_gears += 1
                if not is_idler:
                    B.gear(b, drivers[i], zl + LAYER)
                    n_gears += 1
                B.mesh_constraint(f"mesh_{p}{i}", [(prev[0], prev[1]), (f"{p}{i}", drivens[i - 1])])
                prev = (f"{p}{i}", drivers[i] if not is_idler else drivens[i - 1])
            z_hub = z + (n_axles - 1 if idler else n_axles) * LAYER
            hub = ET.SubElement(base, "body", name=f"{p}_hub")
            ET.SubElement(hub, "joint", name=f"{p}_hub", type="hinge", axis="0 0 1", damping="0.0001",
                          armature="1e-5")
            B.gear(hub, hub_teeth, z_hub)
            n_gears += 1
            B.mesh_constraint(f"mesh_{p}_hub", [(prev[0], prev[1]), (f"{p}_hub", hub_teeth)])
            hub_bodies[p] = hub
            info[p] = dict(z=z_hub, ratio=r, stages=stages, drift=drift, coil_R=R, idler=idler)
            z_arm = z_hub + 6
            z = z_hub + 10
        # the arm, set at the planet's longitude on the start date, and the post up to the planet
        L0 = math.radians(G.mean_longitude(p, jd))
        hb = hub_bodies[p]
        arm = ET.SubElement(hb, "body", name=f"{p}_arm", euler=f(0, 0, L0))
        Rp = RADIUS[p]
        B.liftarm(arm, (0, 0), (Rp, 0), z_arm, "dbg" if p != "Earth" else "lbg")
        info[p]["z_arm"] = z_arm
        info[p]["L0"] = L0
    top = z + 10
    plane_z = top + 40                                  # all planets ride at this height
    for p in ORDER_OUT_IN:
        arm = world.find(f".//body[@name='{p}_arm']")
        Rp, rp = RADIUS[p], SIZE[p]
        za = info[p]["z_arm"]
        ET.SubElement(arm, "geom", type="cylinder", size=f(1.6 * MM, (plane_z - za) / 2 * MM),
                      pos=f(Rp * MM, 0, (za + plane_z) / 2 * MM), rgba=f(*COL["black"], 1), group="1",
                      contype="0", conaffinity="0")
        g = ET.SubElement(arm, "geom", name=p, type="sphere", size=f(rp * MM), pos=f(Rp * MM, 0, plane_z * MM),
                          group="1", contype="0", conaffinity="0")
        if p == "Earth":
            g.set("material", "earth")
        else:
            g.set("rgba", f(*LOOK[p], 1))
        if p == "Saturn":
            if "saturn_ring" not in B.meshes:
                B.meshes["saturn_ring"] = lambda: ring_mesh(rp * 1.35, rp * 2.2)
            ET.SubElement(arm, "geom", type="mesh", mesh="saturn_ring", pos=f(Rp * MM, 0, plane_z * MM),
                          euler=f(0.45, 0, 0), material="ring", group="1", contype="0", conaffinity="0")
        ET.SubElement(arm, "site", name=p, pos=f(Rp * MM, 0, plane_z * MM), size="0.001", group="4")
    # shaft: full height, and the Sun on top
    B.axle(shaft, 8, plane_z - 22)
    ET.SubElement(base, "geom", name="sun", type="sphere", size=f(24 * MM), pos=f(0, 0, plane_z * MM),
                  material="sun", group="1", contype="0", conaffinity="0")
    ET.SubElement(base, "light", name="sunlight", pos=f(0, 0, (plane_z + 30) * MM), dir="0 0 -1",
                  directional="false", diffuse="0.6 0.55 0.45", cutoff="80", exponent="1", castshadow="false")
    # a round base of liftarms and the crank
    for k in range(8):
        a = 2 * math.pi * k / 8
        B.liftarm(base, (0, 0), (90 * math.cos(a), 90 * math.sin(a)), 4, "lbg", thick=7.2)
    ET.SubElement(shaft, "geom", type="box", size=f(20 * MM, 2 * MM, 2 * MM), pos=f(-24 * MM, 0, 12 * MM),
                  rgba=f(*COL["red"], 1), group="1", contype="0", conaffinity="0")

    ET.SubElement(world, "camera", name="overview", pos=f(0.50, -0.66, 0.56),
                  xyaxes=f(*_xyaxes((0.50, -0.66, 0.56), (0.02, 0.02, 0.21))), fovy="44")
    ET.SubElement(world, "camera", name="top", pos=f(0, 0, 1.55), xyaxes="1 0 0 0 1 0", fovy="40")
    ET.SubElement(world, "camera", name="tower", pos=f(0.30, -0.36, 0.30),
                  xyaxes=f(*_xyaxes((0.30, -0.36, 0.30), (0.0, 0.0, 0.17))), fovy="40")

    ten = ET.SubElement(root, "tendon")
    eq = ET.SubElement(root, "equality")
    for name, terms in B.tendons:
        tdn = ET.SubElement(ten, "fixed", name=name)
        merged = {}
        for j, c in terms:
            merged[j] = merged.get(j, 0) + c
        for j, c in merged.items():
            if c:
                ET.SubElement(tdn, "joint", joint=j, coef=str(c))
        ET.SubElement(eq, "tendon", tendon1=name, polycoef="0 0 0 0 0", solref="0.004 1", solimp="0.99 0.999 0.001")
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "velocity", name="crank", joint="shaft", kv="0.2", ctrlrange="-100 100")
    ASSETS.mkdir(exist_ok=True)
    for name, maker in B.meshes.items():
        path = ASSETS / f"{name}.stl"
        if not path.exists():
            maker().export(path)
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl", inertia="exact")
    ET.indent(root)
    facts = dict(info=info, n_gears=n_gears, n_meshes=len(B.tendons), height=plane_z, jd=jd)
    return ET.tostring(root, encoding="unicode"), facts


def load(**kw):
    xml, facts = build(**kw)
    return mujoco.MjModel.from_xml_string(xml), facts


def longitudes(m, d, facts):
    """Heliocentric longitude (deg, 0-360) of each planet in the model."""
    out = {}
    for p, inf in facts["info"].items():
        j = "shaft" if p == "Earth" else f"{p}_hub"
        q = d.qpos[m.jnt_qposadr[m.joint(j).id]]
        out[p] = math.degrees(inf["L0"] + q) % 360
    return out


def turns(m, d, facts):
    """Turns made by each planet since the start (signed, unwrapped)."""
    return {p: d.qpos[m.jnt_qposadr[m.joint("shaft" if p == "Earth" else f"{p}_hub").id]] / (2 * math.pi)
            for p in facts["info"]}
