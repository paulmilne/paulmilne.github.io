"""Walk test: run the tumbling gait for a few cycles and report how it went.

    uv run walk_test.py                       # defaults
    uv run walk_test.py --spin 15 --cycles 6  # pirouetting
"""
import argparse
import math

import numpy as np
import mujoco

from tripod.build import write
from tripod.params import DEFAULT
from tripod.gait import GaitConfig, TumbleGait

ap = argparse.ArgumentParser()
ap.add_argument("--cycles", type=int, default=3)
ap.add_argument("--stride", type=float, default=GaitConfig.stride)
ap.add_argument("--spin", type=float, default=0.0)
ap.add_argument("--heading", type=float, default=0.0)
ap.add_argument("--margin", type=float, default=GaitConfig.margin)
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)

cfg = GaitConfig(stride=args.stride, spin_deg=args.spin,
                 heading_deg=args.heading, margin=args.margin)
gait = TumbleGait(DEFAULT, model, data, cfg)
hood = model.body("hood").id
start = gait.centroid()
T = args.cycles * 3 * (cfg.transfer_time + cfg.swing_time + cfg.settle_time) * 1.6
max_tilt = 0.0
min_z = 1e9
peak_tau = 0.0
two_leg_time = 0.0
fell = False
while gait.steps_done < 3 * args.cycles and data.time - gait.t0 < T:
    info = gait(data)
    mujoco.mj_step(model, data)
    R = data.xmat[hood].reshape(3, 3)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    max_tilt = max(max_tilt, tilt)
    z = data.xpos[hood][2]
    min_z = min(min_z, z)
    peak_tau = max(peak_tau, np.abs(data.actuator_force[:9]).max())
    if info["two_legged"]:
        two_leg_time += model.opt.timestep
    if z < 15:
        fell = True
        break

end = gait.centroid()
planned = cfg.stride * gait.steps_done / 3
moved = float(np.dot(end - start, gait.dir))
R = data.xmat[hood].reshape(3, 3)
yaw_turned = math.degrees(math.atan2(R[1, 0], R[0, 0]) - gait.yaw0)
print(f"gait: {cfg.transfer_time + cfg.swing_time + cfg.settle_time:.1f} s/step planned, stride {cfg.stride:.1f} m/cycle, spin {cfg.spin_deg:.0f}°/step, edge margin {cfg.margin:.1f} m")
print(f"time walked        {data.time - gait.t0:6.1f} s  ({gait.steps_done} steps)")
print(f"feet advanced      {moved:6.1f} m of {planned:.1f} m planned  ->  {moved / (data.time - gait.t0) * 2.237:.1f} mph")
print(f"time on two legs   {two_leg_time / (data.time - gait.t0):6.0%}")
print(f"max hood tilt      {max_tilt:6.1f}°")
print(f"hood height        min {min_z:.1f} m")
print(f"hood yaw turned    {yaw_turned:6.1f}°")
print(f"peak servo torque  {peak_tau / 1e6:6.1f} MN m")
ok = not fell and moved > 0.8 * planned and max_tilt < 25
print("FELL" if fell else ("PASS" if ok else "FAIL"))
