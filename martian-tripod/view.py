"""Open the tripod in MuJoCo's interactive viewer, standing in its IK pose.

    uv run view.py

Double-click a body to select it; Ctrl+right-drag applies a force to it.
The Control panel on the right has a slider for every leg servo, and the
camera list includes 'witness' (eye height, by the cottage) and 'martian'.
"""
import mujoco
import mujoco.viewer

from tripod.build import write

model = mujoco.MjModel.from_xml_path(str(write()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
mujoco.viewer.launch(model, data)
