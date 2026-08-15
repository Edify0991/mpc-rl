"""Pure-MuJoCo G1 model access for offline tools.

This module intentionally avoids ``mjlab`` imports.  It can therefore be used
by conversion/replay tools without triggering task-entry-point registration.
"""

from __future__ import annotations

from pathlib import Path

import mujoco


G1_XML = Path(__file__).parent / "g1" / "xmls" / "g1.xml"


def compile_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(G1_XML))


def actuated_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
  return tuple(
    name for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)) is not None
  )
