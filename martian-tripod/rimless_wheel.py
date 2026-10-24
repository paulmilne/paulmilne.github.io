"""Wells's 'milking stool bowled along' as a three-spoke rimless wheel.

    uv run rimless_wheel.py

The planar rimless wheel (McGeer 1990) is the simplest model of a body
vaulting over rigid legs: mass M, N spokes of length L, radius of gyration
about the hub k*L, rolling down a slope gamma.

Between footfalls the stance spoke is an inverted pendulum with pivot
inertia J = M L^2 (1 + k^2). At a footfall angular momentum about the new
foot is conserved, which gives the impact ratio

    r = w+ / w- = (k^2 + cos 2a) / (1 + k^2),     a = pi / N.

For N = 3, cos 2a = -1/2. With the mass concentrated at the hub (k^2 small)
r is negative: the footfall does not just slow the wheel, it stops it dead.
A three-legged stool only rolls passively if k^2 > 1/2, i.e. most of its
inertia is out in the legs. This script

  1. tabulates the shallowest slope on which each wheel keeps rolling,
  2. checks the analytic limit cycle against a MuJoCo contact simulation,
  3. applies it to the tripod model's own mass distribution, and estimates
     the power its legs must supply to keep it rolling across flat ground.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tripod.params import DEFAULT

G = 9.81
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- analytic
def ratio(n: int, k2: float) -> float:
    return (k2 + math.cos(2 * math.pi / n)) / (1 + k2)


def limit_cycle(n: int, gamma: float, k2: float, L: float = 1.0):
    """Fixed point of the stride map. Returns (w_minus, w_plus, rolls)."""
    a = math.pi / n
    r = ratio(n, k2)
    if r <= 1e-12:
        return math.nan, math.nan, False
    # energy over one stance phase: 1/2 J (w-^2 - w+^2) = M g L 2 sin a sin gamma
    w_minus = math.sqrt(4 * G * math.sin(a) * math.sin(gamma) / (L * (1 + k2) * (1 - r * r)))
    w_plus = r * w_minus
    # vault condition: the hub must get over the top of the stance spoke
    rolls = 0.5 * (1 + k2) * L * w_plus ** 2 > G * (1 - math.cos(a - gamma))
    return w_minus, w_plus, rolls


def min_slope(n: int, k2: float) -> float:
    if ratio(n, k2) <= 1e-12:
        return math.nan
    lo, hi = 1e-7, math.pi / n
    if not limit_cycle(n, hi, k2)[2]:
        return math.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if limit_cycle(n, mid, k2)[2] else (mid, hi)
    return hi


def stance_time(n: int, gamma: float, k2: float, w_plus: float, L: float) -> float:
    """theta from -a to +a under theta'' = g sin(theta + gamma) / (L (1 + k^2))."""
    a = math.pi / n
    th, w, t, dt = -a, w_plus, 0.0, 1e-4
    while th < a:
        w += G * math.sin(th + gamma) / (L * (1 + k2)) * dt
        th += w * dt
        t += dt
        if w <= 0:
            return math.inf
    return t


# ---------------------------------------------------------------- MuJoCo
def wheel_xml(n: int, gamma: float, k2: float, L: float = 1.0, M: float = 10.0) -> str:
    rf = 0.02 * L
    parts = []
    for j in range(n):
        ang = 2 * math.pi * j / n
        x, z = -(L - rf) * math.sin(ang), -(L - rf) * math.cos(ang)
        parts.append(f'<geom type="capsule" fromto="0 0 0 {x:.6f} 0 {z:.6f}" size="{rf / 2}" contype="0" conaffinity="0" rgba=".6 .5 .3 1"/>')
        parts.append(f'<geom name="foot{j}" type="sphere" pos="{x:.6f} 0 {z:.6f}" size="{rf}" friction="3 0.005 0.0001"/>')
    I = max(k2, 1e-4) * M * L * L
    gx, gz = G * math.sin(gamma), -G * math.cos(gamma)
    return f"""
<mujoco>
  <option timestep="1e-4" gravity="{gx:.6f} 0 {gz:.6f}" integrator="implicitfast"/>
  <default><geom solref="0.001 1" solimp="0.99 0.999 0.0001" density="0"/></default>
  <worldbody>
    <geom name="slope" type="plane" size="0 0 1"/>
    <body name="hub" pos="0 0 {(L - rf) * math.cos(math.pi / n) + rf:.6f}">
      <inertial pos="0 0 0" mass="{M}" diaginertia="{I:.6g} {I:.6g} {I:.6g}"/>
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="th" type="hinge" axis="0 1 0"/>
      {''.join(parts)}
    </body>
  </worldbody>
</mujoco>"""


def simulate(n: int, gamma: float, k2: float, w0: float, impacts_wanted: int = 30, L: float = 1.0):
    """Start just after a footfall and roll downhill (+x). Returns (t, w, pre-impact speeds)."""
    model = mujoco.MjModel.from_xml_string(wheel_xml(n, gamma, k2, L))
    data = mujoco.MjData(model)
    a = math.pi / n
    # spoke 0 points down at th = 0; th = -a puts its foot a*L downhill of the hub
    data.qpos[2] = -a
    data.qvel[2] = w0
    data.qvel[0] = w0 * L * math.cos(a)
    data.qvel[1] = w0 * L * math.sin(a)
    mujoco.mj_forward(model, data)
    feet = {model.geom(f"foot{j}").id for j in range(n)}
    ts, ws, pre = [], [], []
    stance = model.geom("foot0").id
    every = 10
    k = 0
    while len(pre) < impacts_wanted and data.time < 20:
        w_before = data.qvel[2]
        mujoco.mj_step(model, data)
        touching = {g for c in data.contact[: data.ncon] for g in (c.geom1, c.geom2) if g in feet}
        new = touching - {stance}
        # a footfall: a foot other than the current stance foot lands ahead of the hub
        ahead = [g for g in new if data.geom_xpos[g][0] > data.body("hub").xpos[0]]
        if ahead:
            pre.append(w_before)
            stance = ahead[0]
        if k % every == 0:
            ts.append(data.time); ws.append(data.qvel[2])
        k += 1
        if data.qvel[2] < -1e-3 and data.time > 1:   # rolled backwards: it has stalled
            break
    return np.array(ts), np.array(ws), np.array(pre)


def footfall_ratio(n: int, k2: float, w: float = 2.0, L: float = 1.0) -> float:
    """One footfall on flat ground: measured w+/w- from MuJoCo, to compare with ratio().

    Starts 0.02 rad before the next foot lands, rotating at w about the stance
    foot, so gravity barely acts during the measurement.
    """
    a, rf, delta = math.pi / n, 0.02 * L, 0.02
    model = mujoco.MjModel.from_xml_string(wheel_xml(n, 0.0, k2, L))
    data = mujoco.MjData(model)
    th = a - delta
    data.qpos[1] = (L - rf) * (math.cos(th) - math.cos(a))
    data.qpos[2] = th
    data.qvel[0] = w * L * math.cos(th)
    data.qvel[1] = -w * L * math.sin(th)
    data.qvel[2] = w
    mujoco.mj_forward(model, data)
    stance = model.geom("foot0").id
    feet = {model.geom(f"foot{j}").id for j in range(n)}
    t_hit = w_pre = None
    while data.time < 1:
        w_before = data.qvel[2]
        mujoco.mj_step(model, data)
        if t_hit is None:
            touching = {g for c in data.contact[: data.ncon] for g in (c.geom1, c.geom2) if g in feet}
            if touching - {stance}:
                t_hit, w_pre = data.time, w_before
        elif data.time > t_hit + 0.006:
            return data.qvel[2] / w_pre
    return math.nan


def main():
    p = DEFAULT
    L = math.hypot(p.stance_radius - p.hip_radius, p.stance_height - p.hip_drop - p.leg.foot_radius) + p.hip_drop
    M = p.total_mass
    # tripod inertia about the hub with legs locked: each leg a uniform rod hub->foot plus a foot mass at L
    m_leg = p.leg.thigh_mass + p.leg.shin_mass
    I_hub = 3 * (m_leg * L * L / 3 + p.leg.foot_mass * L * L)
    k2_tripod = I_hub / (M * L * L)

    k2s = (0.0, k2_tripod, 0.5, 1.0)
    print("Shallowest slope on which an N-spoke rimless wheel keeps rolling (— = never)")
    print("   N  " + "".join(f"  k²={k:<5.2f}" for k in k2s))
    for n in (3, 4, 5, 6, 8, 12):
        cells = []
        for k2 in k2s:
            s = min_slope(n, k2)
            cells.append(f"  {('—' if math.isnan(s) else f'{math.degrees(s):5.1f}°'):>8s}")
        print(f"  {n:2d}  " + "".join(cells))
    print(f"(k²={k2_tripod:.2f} is the tripod model with its legs locked)")

    # ---- validation 1: the impact law, one footfall of a three-spoke wheel on flat ground
    print("\nValidation 1 - the impact law: one footfall on flat ground, w+/w- (L = 1 m)")
    print("   k²      N=3 analytic   N=3 MuJoCo     N=6 analytic   N=6 MuJoCo")
    for k2 in (2.0, 1.0, 0.5, k2_tripod, 0.0):
        print(f"  {k2:4.2f}   {ratio(3, k2):12.3f}   {footfall_ratio(3, k2):10.3f}     "
              f"{ratio(6, k2):12.3f}   {footfall_ratio(6, k2):10.3f}")
    print("  A negative analytic ratio would need the trailing foot to pull on the ground.")
    print("  It can't, so in MuJoCo both feet end up planted and the wheel stops dead.")
    print("  MuJoCo reads low (most for N=3, where little speed survives the footfall):")
    print("  soft contacts and the finite foot radius both bleed a little extra energy.")

    # ---- validation 2: a full limit cycle, six spokes on a 15° slope (gentle enough not to take off)
    n6, g6 = 6, math.radians(15)
    wm, wp, _ = limit_cycle(n6, g6, 0.0)
    ts, ws, pre = simulate(n6, g6, 0.0, w0=1.3 * wp, impacts_wanted=20)
    tail = pre[-10:]
    print(f"\nValidation 2 - six-spoke wheel, k² = 0, 15° slope: limit cycle w- before each footfall")
    print(f"  analytic {wm:.4f} rad/s, MuJoCo {tail.mean():.4f} ± {tail.std():.4f} rad/s ({abs(tail.mean() - wm) / wm:.1%})")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(ts, ws, lw=1.1, color="#23459a", label="MuJoCo hinge speed")
    ax.axhline(wm, color="#b5301c", ls="--", lw=1, label="analytic $\\omega^-$ (before footfall)")
    ax.axhline(wp, color="#128a46", ls=":", lw=1.2, label="analytic $\\omega^+$ (after footfall)")
    ax.set_xlim(0, ts[-1])
    ax.set_xlabel("time (s)"); ax.set_ylabel("angular speed (rad/s)")
    ax.set_title("Six-spoke rimless wheel settling onto its limit cycle, 15° slope, L = 1 m")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "rimless_wheel_limit_cycle.png", dpi=160)

    n = 3
    # ---- flat ground: the legs must put back what each footfall destroys
    a = math.pi / n
    r = ratio(n, k2_tripod)
    w_min = math.sqrt(2 * G * (1 - math.cos(a)) / (L * (1 + k2_tripod)))   # just enough to vault the top
    d = 2 * L * math.sin(a)
    print(f"\nTripod on flat ground: L = {L:.1f} m, mass {M / 1e3:.0f} t, impact ratio r = {r:.2f}")
    print("  (r < 0: every footfall kills the forward roll; the legs must supply the next vault)")
    print("  speed over vault minimum   stride   speed (mph)   power (MW)   cost of transport   Froude")
    for kx in (1.05, 1.2, 1.5, 2.0):
        w = kx * w_min
        T = stance_time(n, 0.0, k2_tripod, w, L)
        J = M * L * L * (1 + k2_tripod)
        dE = 0.5 * J * w * w - 0.5 * J * (max(r, 0) * w) ** 2   # energy to restore after the footfall
        v = d / T
        print(f"  {kx:5.2f}x                     {T:5.2f} s   {v * 2.237:9.1f}   {dE / T / 1e6:9.1f}   {dE / (M * G * d):14.3f}   {v * v / (G * L):7.2f}")
    print(f"\nplot: {OUT / 'rimless_wheel_limit_cycle.png'}")


if __name__ == "__main__":
    main()
