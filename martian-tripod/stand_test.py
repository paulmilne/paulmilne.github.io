"""Stand test: drop the tripod into its IK stance and check it holds.

    uv run stand_test.py

Reports hood height drift, foot normal forces against the machine's weight,
and the peak joint torques the servos need to hold the pose.
"""
import numpy as np
import mujoco

from tripod.build import write

model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.mj_forward(model, data)

hood = model.body("hood").id
weight = model.body_subtreemass[hood] * -model.opt.gravity[2]
z0 = data.body(hood).xpos[2]
T = 10.0
peak_tau = np.zeros(model.nu)
while data.time < T:
    mujoco.mj_step(model, data)
    peak_tau = np.maximum(peak_tau, np.abs(data.actuator_force))

z1 = data.body(hood).xpos[2]
forces = np.array([data.sensor(f"foot{i}_touch").data[0] for i in range(3)])
print(f"machine mass        {model.body_subtreemass[hood] / 1e3:8.1f} t")
print(f"hood height         {z0:8.2f} m  ->  {z1:8.2f} m after {T:.0f} s")
print(f"foot normal forces  {', '.join(f'{x / 1e6:.2f}' for x in forces)} MN  (sum {forces.sum() / 1e6:.2f}, weight {weight / 1e6:.2f})")
print("peak servo torque by joint (MN m):")
for i in range(model.nu):
    print(f"  {model.actuator(i).name:10s} {peak_tau[i] / 1e6:8.2f}")
ok = abs(z1 - z0) < 0.5 and abs(forces.sum() / weight - 1) < 0.05
print("PASS" if ok else "FAIL")
