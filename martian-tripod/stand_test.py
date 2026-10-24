"""Stand test: hold the tripod in its stance under the balance controller.

    uv run stand_test.py

Reports hood height drift, foot normal forces against the machine's weight,
and the joint torques the leg motors need to hold the pose.
"""
import numpy as np
import mujoco

from tripod.build import write
from tripod.params import DEFAULT
from tripod.control import BodyController
from tripod.kinematics import rotz

model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)
ctl = BodyController(DEFAULT, model)
goal, _ = ctl.com_state(data)

hood = model.body("hood").id
weight = model.body_subtreemass[hood] * -model.opt.gravity[2]
z0 = data.body(hood).xpos[2]
T = 10.0
peak_tau = np.zeros(9)
while data.time < T:
    ctl(data, [0, 1, 2], goal, np.zeros(3), rotz(0.0))
    mujoco.mj_step(model, data)
    if data.time > 2:
        peak_tau = np.maximum(peak_tau, np.abs(data.ctrl[:9]))

z1 = data.body(hood).xpos[2]
forces = np.array([data.sensor(f"foot{i}_touch").data[0] for i in range(3)])
print(f"machine mass        {model.body_subtreemass[hood] / 1e3:8.1f} t")
print(f"hood height         {z0:8.2f} m  ->  {z1:8.2f} m after {T:.0f} s")
print(f"foot normal forces  {', '.join(f'{x / 1e6:.2f}' for x in forces)} MN  (sum {forces.sum() / 1e6:.2f}, weight {weight / 1e6:.2f})")
print("holding torque by joint (MN m):")
for i in range(9):
    print(f"  {model.actuator(i).name:10s} {peak_tau[i] / 1e6:8.2f}")
ok = abs(z1 - z0) < 0.5 and abs(forces.sum() / weight - 1) < 0.05
print("PASS" if ok else "FAIL")
