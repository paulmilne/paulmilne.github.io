"""Single source of truth for the tripod's geometry and mass.

Everything else (the MuJoCo model, the stand test, later the Blender rig)
is generated from these numbers, so a change here propagates everywhere.
Units are SI: metres, kilograms, seconds, radians.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Leg:
    thigh: float = 16.0          # hip -> knee length (m)
    shin: float = 28.0           # knee -> foot length (m)
    radius: float = 0.55         # structural tube radius (m)
    thigh_mass: float = 13e3     # kg
    shin_mass: float = 11e3      # kg
    foot_radius: float = 1.1     # contact sphere (m)
    mount_mass: float = 2000.0   # hip housing, kg
    knee_mass: float = 500.0     # knee housing, kg
    foot_mass: float = 1500.0    # kg


@dataclass(frozen=True)
class Tentacle:
    links: int = 10
    link_length: float = 1.3
    radius: float = 0.22
    link_mass: float = 150.0
    stiffness: float = 2.0e5     # N m / rad, passive spring per joint
    damping: float = 6.0e4       # N m s / rad


@dataclass(frozen=True)
class Tripod:
    hood_radius: float = 4.5          # the "brazen hood" (m)
    hood_height: float = 3.2          # vertical semi-axis of the hood ellipsoid (m)
    hood_mass: float = 160e3          # kg
    heatray_mass: float = 1200.0      # arm + projector, kg
    # the carrier: "the great metallic carrier which projected behind him ... much as
    # a workman's basket hangs over his shoulder" (Book II, ch. I)
    carrier_mass: float = 9000.0      # kg, empty
    carrier_radius: float = 2.6       # m
    carrier_height: float = 3.4       # m
    carrier_gap: float = 1.2          # m between the hood's rim and the basket
    carrier_drop: float = 1.0         # m the basket's floor sits below the hood centre
    hip_radius: float = 2.6           # hips sit on this circle under the hood (m)
    hip_drop: float = 2.0             # hips are this far below hood centre (m)
    stance_height: float = 30.0       # hood centre above ground when standing (m)
    stance_radius: float = 22.0       # feet on this circle, measured from body axis (m)
    leg: Leg = field(default_factory=Leg)
    tentacle: Tentacle = field(default_factory=Tentacle)
    leg_azimuths_deg: tuple[float, float, float] = (0.0, 120.0, 240.0)
    knee_up: bool = True              # knees above the hip-foot line, spider-fashion

    # leg actuators: torque motors; kp/kv are the joint PD gains for a swinging leg
    kp: float = 6.0e8                 # N m / rad
    kv: float = 6.0e7                 # N m s / rad
    max_torque: float = 4.0e7         # N m, motor saturation (about 6x the standing load)

    @property
    def total_mass(self) -> float:
        l = self.leg
        legs = 3 * (l.thigh_mass + l.shin_mass + l.mount_mass + l.knee_mass + l.foot_mass)
        tents = 2 * self.tentacle.links * self.tentacle.link_mass
        return self.hood_mass + self.heatray_mass + self.carrier_mass + legs + tents


def planar_ik(L1: float, L2: float, dx: float, dz: float, knee_up: bool = True) -> tuple[float, float]:
    """Two-link IK in the leg plane. Hinge axes along +y, so positive angles
    swing the links downward; at zero both links point along +x.
    Returns (q_hip, q_knee). Raises ValueError if the point is out of reach."""
    d2 = dx * dx + dz * dz
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if not -1.0 <= c <= 1.0:
        raise ValueError(f"foot out of reach: distance {math.sqrt(d2):.2f} m, legs {L1}+{L2} m")
    beta = math.acos(c)
    beta = -beta if knee_up else beta         # sign picks knee above or below the hip-foot line
    phi1 = math.atan2(dz, dx) - math.atan2(L2 * math.sin(beta), L1 + L2 * math.cos(beta))
    phi2 = phi1 + beta                        # elevation angles of thigh and shin
    return -phi1, -(phi2 - phi1)


def leg_ik(p: Tripod) -> tuple[float, float]:
    """Hip-pitch and knee angles for the standing pose (feet on the stance circle)."""
    dx = p.stance_radius - p.hip_radius
    dz = -(p.stance_height - p.hip_drop - p.leg.foot_radius)
    return planar_ik(p.leg.thigh, p.leg.shin, dx, dz, p.knee_up)


DEFAULT = Tripod()
