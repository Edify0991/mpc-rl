"""Pure-MuJoCo Jingchu01 model access for offline tools."""

from __future__ import annotations

from pathlib import Path

import mujoco


JINGCHU01_XML = Path(__file__).parent / "jingchu01" / "xmls" / "jingchu01.xml"


def compile_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(JINGCHU01_XML))


def actuated_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
  return tuple(
    name for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)) is not None
  )
