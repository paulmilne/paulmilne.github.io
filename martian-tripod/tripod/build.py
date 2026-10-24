"""Generate the MJCF (MuJoCo XML) for the tripod from tripod.params.

    python -m tripod.build            # writes models/tripod.xml

The model:
  * hood   - free-floating body (6 DOF), an ellipsoid with a Heat-Ray arm
  * 3 legs - hip_yaw (z), hip_pitch (y), knee (y): 9 torque-controlled joints
  * 2 tentacles - chains of capsules with passive spring/damper joints
  * a "stand" keyframe computed by inverse kinematics
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .params import DEFAULT, Tripod, leg_ik

ROOT = Path(__file__).resolve().parent.parent


def f(*xs: float) -> str:
    return " ".join(f"{x:.6g}" for x in xs)


# ---------------------------------------------------------------- the village
# Everything the Heat-Ray can set alight has a geom name starting "burn_".
# The tripod starts at the origin and walks along +x; the village lies ahead,
# clear of the feet (which stay within about 25 m of the line of march).

def cottage(wb, name: str, x: float, y: float, yaw: float = 0.0, w: float = 8.0, d: float = 6.0, h: float = 5.0):
    b = ET.SubElement(wb, "body", name=name, pos=f(x, y, 0), euler=f(0, 0, yaw))
    ET.SubElement(b, "geom", name=f"burn_{name}_walls", type="box", size=f(w / 2, d / 2, h / 2), pos=f(0, 0, h / 2),
                  rgba="0.80 0.74 0.62 1")
    r = 0.72
    for side in (1, -1):
        ET.SubElement(b, "geom", name=f"burn_{name}_roof{'n' if side > 0 else 's'}", type="box",
                      size=f(w / 2 + 0.3, d / 2 / math.cos(r) / 2 + 0.3, 0.25),
                      pos=f(0, side * d / 4, h + d / 4 * math.tan(r)), euler=f(-side * r, 0, 0), rgba="0.47 0.21 0.15 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_chimney", type="box", size="0.5 0.5 1.4", pos=f(w / 3, 0, h + 2.2),
                  rgba="0.55 0.3 0.25 1")


def pine(wb, name: str, x: float, y: float, h: float = 14.0):
    b = ET.SubElement(wb, "body", name=name, pos=f(x, y, 0))
    ET.SubElement(b, "geom", name=f"burn_{name}_trunk", type="cylinder", size=f(0.35, h * 0.3), pos=f(0, 0, h * 0.3),
                  rgba="0.35 0.24 0.16 1")
    for k, (z, r) in enumerate(((0.45, 3.0), (0.65, 2.3), (0.82, 1.5))):
        ET.SubElement(b, "geom", name=f"burn_{name}_crown{k}", type="ellipsoid", size=f(r, r, h * 0.16),
                      pos=f(0, 0, h * z), rgba="0.16 0.30 0.18 1")


def church(wb, name: str, x: float, y: float):
    b = ET.SubElement(wb, "body", name=name, pos=f(x, y, 0))
    ET.SubElement(b, "geom", name=f"burn_{name}_nave", type="box", size="9 4.5 4.5", pos="0 0 4.5", rgba="0.70 0.68 0.62 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_naveroof", type="box", size="9.2 3.4 0.3", pos="0 1.7 10.2",
                  euler="-0.6 0 0", rgba="0.36 0.36 0.38 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_naveroof2", type="box", size="9.2 3.4 0.3", pos="0 -1.7 10.2",
                  euler="0.6 0 0", rgba="0.36 0.36 0.38 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_tower", type="box", size="3 3 9", pos="-11.5 0 9", rgba="0.68 0.66 0.60 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_spire", type="ellipsoid", size="2.4 2.4 7", pos="-11.5 0 23",
                  rgba="0.30 0.30 0.33 1")


def haystack(wb, name: str, x: float, y: float):
    b = ET.SubElement(wb, "body", name=name, pos=f(x, y, 0))
    ET.SubElement(b, "geom", name=f"burn_{name}", type="cylinder", size="2.2 1.6", pos="0 0 1.6", rgba="0.78 0.66 0.35 1")
    ET.SubElement(b, "geom", name=f"burn_{name}_top", type="ellipsoid", size="2.3 2.3 1.6", pos="0 0 3.2",
                  rgba="0.78 0.66 0.35 1")


def village(wb):
    cottage(wb, "lodge", -30, 18)                     # by the start, for scale
    cottage(wb, "cottage_a", 55, 34, yaw=0.2)
    cottage(wb, "cottage_b", 68, 40, yaw=-0.1)
    cottage(wb, "cottage_c", 82, -36, yaw=0.15)
    cottage(wb, "inn", 98, 42, yaw=0.05, w=12, d=7, h=6.5)
    church(wb, "church", 120, -40)
    for k, (x, y) in enumerate(((40, -40), (45, -46), (51, -42), (36, -50), (58, -48), (47, -55))):
        pine(wb, f"pine{k}", x, y, h=12 + 3 * ((k * 7) % 3))
    for k, (x, y) in enumerate(((75, 52), (80, 49), (86, 54))):
        haystack(wb, f"hay{k}", x, y)


def build(p: Tripod = DEFAULT) -> ET.Element:
    m = ET.Element("mujoco", model="martian_tripod")
    ET.SubElement(m, "compiler", angle="radian", autolimits="true")
    ET.SubElement(m, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 -9.81")
    ET.SubElement(m, "size", memory="64M")
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080", azimuth="135", elevation="-12")
    ET.SubElement(vis, "map", haze="0.4")
    ET.SubElement(vis, "headlight", ambient="0.25 0.25 0.25", diffuse="0.5 0.5 0.5")
    ET.SubElement(vis, "quality", shadowsize="8192")

    a = ET.SubElement(m, "asset")
    ET.SubElement(a, "texture", type="skybox", builtin="gradient", rgb1="0.55 0.6 0.62",
                  rgb2="0.12 0.13 0.14", width="512", height="3072")
    ET.SubElement(a, "texture", name="fields", type="2d", builtin="checker", rgb1="0.33 0.38 0.27",
                  rgb2="0.29 0.34 0.24", width="512", height="512", mark="edge", markrgb="0.25 0.28 0.2")
    ET.SubElement(a, "material", name="ground", texture="fields", texrepeat="40 40", reflectance="0")
    ET.SubElement(a, "material", name="brass", rgba="0.62 0.5 0.27 1", specular="0.8", shininess="0.6", reflectance="0.15")
    ET.SubElement(a, "material", name="steel", rgba="0.42 0.44 0.46 1", specular="0.6", shininess="0.5")
    ET.SubElement(a, "material", name="flesh", rgba="0.36 0.4 0.33 1", specular="0.3", shininess="0.2")
    ET.SubElement(a, "material", name="ray", rgba="0.9 0.85 0.6 1", emission="0.6")

    dflt = ET.SubElement(m, "default")
    leg_d = ET.SubElement(dflt, "default", **{"class": "leg"})
    ET.SubElement(leg_d, "joint", type="hinge", damping="2e6", armature="1e5")
    ET.SubElement(leg_d, "geom", material="steel", contype="0", conaffinity="0", group="1")
    ET.SubElement(leg_d, "motor", ctrlrange=f(-p.max_torque, p.max_torque))
    t_d = ET.SubElement(dflt, "default", **{"class": "tentacle"})
    ET.SubElement(t_d, "joint", type="hinge", stiffness=f(p.tentacle.stiffness),
                  damping=f(p.tentacle.damping), range="-0.6 0.6")
    ET.SubElement(t_d, "geom", type="capsule", material="flesh", contype="0", conaffinity="0", group="1")

    wb = ET.SubElement(m, "worldbody")
    # a high spotlight standing in for the sun (MuJoCo's spot shadows are cleaner than directional ones)
    ET.SubElement(wb, "light", name="sun", mode="fixed", pos="90 -70 160", dir="-0.45 0.35 -0.82",
                  cutoff="70", exponent="0", castshadow="true", diffuse="0.95 0.9 0.8", attenuation="1 0 0")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 1", material="ground",
                  friction="1.0 0.005 0.0001")
    village(wb)

    hood = ET.SubElement(wb, "body", name="hood", pos=f(0, 0, p.stance_height))
    # an onlooker at eye height beyond the cottage, always looking at the hood
    ET.SubElement(wb, "camera", name="witness", mode="targetbody", target="hood", pos="150 75 1.7", fovy="42")
    ET.SubElement(hood, "freejoint", name="root")
    ET.SubElement(hood, "geom", name="hood", type="ellipsoid", size=f(p.hood_radius, p.hood_radius, p.hood_height),
                  mass=f(p.hood_mass), material="brass", group="1")
    ET.SubElement(hood, "geom", name="cowl", type="cylinder", size=f(p.hip_radius + 0.6, 0.5),
                  pos=f(0, 0, -p.hip_drop + 0.5), material="steel", mass="1", group="1")  # cosmetic
    # the Heat-Ray "camera" on a jointed arm at the front (+x)
    arm = ET.SubElement(hood, "body", name="heatray", pos=f(p.hood_radius * 0.8, 0, p.hood_height * 0.4))
    # yaw is unlimited: the projector can turn right round on its mount
    ET.SubElement(arm, "joint", name="ray_yaw", type="hinge", axis="0 0 1", damping="2e4")
    ET.SubElement(arm, "joint", name="ray_pitch", type="hinge", axis="0 1 0", range="-0.4 1.4", damping="2e4")
    ET.SubElement(arm, "geom", type="capsule", fromto="0 0 0 2.2 0 0", size="0.3", material="steel",
                  mass=f(p.heatray_mass * 2 / 3), group="1", contype="0", conaffinity="0")
    ET.SubElement(arm, "geom", type="cylinder", pos="2.6 0 0", euler="0 1.5708 0", size="0.55 0.45", material="ray",
                  mass=f(p.heatray_mass / 3), group="1", contype="0", conaffinity="0")
    ET.SubElement(arm, "site", name="muzzle", pos="3.1 0 0", size="0.25", rgba="1 0.9 0.6 1")
    ET.SubElement(hood, "camera", name="martian", pos=f(p.hood_radius + 1.0, 0, 0.5), xyaxes="0 -1 0 0.15 0 1", fovy="70")
    ET.SubElement(hood, "camera", name="track", mode="trackcom", pos=f(-70, -70, -5), xyaxes="0.7071 -0.7071 0 0.12 0.12 0.985", fovy="45")
    ET.SubElement(hood, "site", name="imu", pos="0 0 0", size="0.2")

    q_hip, q_knee = leg_ik(p)
    for i, az_deg in enumerate(p.leg_azimuths_deg):
        az = math.radians(az_deg)
        hx, hy = p.hip_radius * math.cos(az), p.hip_radius * math.sin(az)
        mount = ET.SubElement(hood, "body", name=f"leg{i}_mount", pos=f(hx, hy, -p.hip_drop), euler=f(0, 0, az))
        ET.SubElement(mount, "joint", name=f"leg{i}_yaw", axis="0 0 1", range="-0.8 0.8", **{"class": "leg"})
        ET.SubElement(mount, "geom", type="sphere", size=f(p.leg.radius * 1.9), mass=f(p.leg.mount_mass), material="brass", **{"class": "leg"})
        thigh = ET.SubElement(mount, "body", name=f"leg{i}_thigh")
        ET.SubElement(thigh, "joint", name=f"leg{i}_hip", axis="0 1 0", range="-1.2 1.4", **{"class": "leg"})
        ET.SubElement(thigh, "geom", type="capsule", fromto=f(0, 0, 0, p.leg.thigh, 0, 0), size=f(p.leg.radius),
                      mass=f(p.leg.thigh_mass), **{"class": "leg"})
        shin = ET.SubElement(thigh, "body", name=f"leg{i}_shin", pos=f(p.leg.thigh, 0, 0))
        ET.SubElement(shin, "joint", name=f"leg{i}_knee", axis="0 1 0", range="-0.3 2.6", **{"class": "leg"})
        ET.SubElement(shin, "geom", type="sphere", size=f(p.leg.radius * 1.5), mass=f(p.leg.knee_mass), material="brass", **{"class": "leg"})
        ET.SubElement(shin, "geom", type="capsule", fromto=f(0, 0, 0, p.leg.shin, 0, 0), size=f(p.leg.radius * 0.8),
                      mass=f(p.leg.shin_mass), **{"class": "leg"})
        ET.SubElement(shin, "geom", name=f"foot{i}", type="sphere", pos=f(p.leg.shin, 0, 0),
                      size=f(p.leg.foot_radius), mass=f(p.leg.foot_mass), material="brass", contype="1", conaffinity="1", group="1",
                      friction="1.2 0.01 0.001")
        ET.SubElement(shin, "site", name=f"foot{i}", pos=f(p.leg.shin, 0, 0), size=f(p.leg.foot_radius * 1.3), rgba="0 0 0 0")

    # two tentacles hanging from under the front of the hood
    t = p.tentacle
    for side, y in (("l", 1.6), ("r", -1.6)):
        parent = ET.SubElement(hood, "body", name=f"tent_{side}0", pos=f(p.hood_radius * 0.55, y, -p.hood_height * 0.6))
        body = parent
        for k in range(t.links):
            ET.SubElement(body, "joint", name=f"tent_{side}{k}_a", axis="0 1 0", **{"class": "tentacle"})
            ET.SubElement(body, "joint", name=f"tent_{side}{k}_b", axis="1 0 0", **{"class": "tentacle"})
            r = t.radius * (1 - 0.6 * k / t.links)
            ET.SubElement(body, "geom", fromto=f(0, 0, 0, 0, 0, -t.link_length), size=f(r),
                          mass=f(t.link_mass), **{"class": "tentacle"})
            if k < t.links - 1:
                body = ET.SubElement(body, "body", name=f"tent_{side}{k + 1}", pos=f(0, 0, -t.link_length))

    act = ET.SubElement(m, "actuator")
    for i in range(3):
        for j in ("yaw", "hip", "knee"):
            ET.SubElement(act, "motor", name=f"leg{i}_{j}", joint=f"leg{i}_{j}", **{"class": "leg"})
    ET.SubElement(act, "position", name="ray_yaw", joint="ray_yaw", kp="5e6", kv="6e5")
    ET.SubElement(act, "position", name="ray_pitch", joint="ray_pitch", kp="5e6", kv="6e5")

    sens = ET.SubElement(m, "sensor")
    for i in range(3):
        ET.SubElement(sens, "touch", name=f"foot{i}_touch", site=f"foot{i}")
    ET.SubElement(sens, "framepos", name="hood_pos", objtype="body", objname="hood")

    # keyframe: standing pose from IK. qpos = 7 (free) + 2 (ray) + 9 (legs) + tentacle joints
    n_tent = 2 * t.links * 2
    qpos = [0, 0, p.stance_height, 1, 0, 0, 0, 0, 0.3]
    ctrl = []
    for _ in range(3):
        qpos += [0, q_hip, q_knee]
        ctrl += [0, 0, 0]
    # order in qpos follows the tree: hood free joint, heatray (2), then legs, then tentacles
    qpos += [0] * n_tent
    kf = ET.SubElement(m, "keyframe")
    ET.SubElement(kf, "key", name="stand", qpos=f(*qpos), ctrl=f(*ctrl, 0, 0.3))
    return m


def write(path: Path | None = None, p: Tripod = DEFAULT) -> Path:
    path = path or ROOT / "models" / "tripod.xml"
    tree = ET.ElementTree(build(p))
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="unicode", xml_declaration=False)
    return path


if __name__ == "__main__":
    out = write()
    print(f"wrote {out}")
