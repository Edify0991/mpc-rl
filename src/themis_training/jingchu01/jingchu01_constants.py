"""Jingchu01 28-DOF articulation, actuator, collision and MPC constants.

The numerical actuator data are the Jingchu01 values used by the local
BeyondMimic configuration.  The committed MJCF is a self-contained copy of
``/home/user/wmd/jc01-model`` with only its root body renamed ``Robotbase`` to
match the named reference-motion contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


@dataclass(frozen=True)
class MotorSpec:
  stiffness: float
  damping: float
  effort_limit: float
  velocity_limit: float
  armature: float


_OMEGA_N = 2.0 * 3.141592653589793 * 6.0
_DAMPING_RATIO = 1.1


def _motor(armature: float, effort_limit: float, velocity_limit: float, *, stiffness: float | None = None, multiplier: float = 1.0) -> MotorSpec:
  armature *= multiplier
  return MotorSpec(
    stiffness=stiffness if stiffness is not None else armature * _OMEGA_N**2,
    damping=2.0 * _DAMPING_RATIO * armature * _OMEGA_N,
    effort_limit=effort_limit,
    velocity_limit=velocity_limit,
    armature=armature,
  )


# Reflected inertias and gain construction from BeyondMimic's JC01 parameters.
MOTOR_HIP_ROLL_PITCH = _motor(0.2773762228, 330.0, 12.0)
MOTOR_HIP_YAW = _motor(0.07001770124, 150.0, 18.0)
MOTOR_KNEE = _motor(300.0 / _OMEGA_N**2, 306.0, 12.0, stiffness=300.0)
MOTOR_ANKLE = _motor(0.0485578476, 90.0, 20.0, multiplier=2.0)
MOTOR_SHOULDER = _motor(0.0485578476, 90.0, 20.0)
MOTOR_ELBOW = _motor(0.03960461065, 60.0, 24.0)
MOTOR_WRIST = _motor(0.02422284137, 36.0, 28.0)


JINGCHU01_JOINT_NAMES: tuple[str, ...] = (
  "right_hip_roll", "right_hip_yaw", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
  "left_hip_roll", "left_hip_yaw", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
  "waist_roll", "waist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch", "right_elbow_yaw", "right_wrist_pitch", "right_wrist_roll",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch", "left_elbow_yaw", "left_wrist_pitch", "left_wrist_roll",
)

JINGCHU01_ANCHOR_BODY_NAME = "Robotbase"
JINGCHU01_BODY_NAMES: tuple[str, ...] = (JINGCHU01_ANCHOR_BODY_NAME,) + JINGCHU01_JOINT_NAMES
# Keep the contact order left, right, which is the order assumed by the
# two-contact centroidal MPC state and by all reference-contact landmarks.
JINGCHU01_FEET_BODY_NAMES = ("left_ankle_roll", "right_ankle_roll")
JINGCHU01_FEET_SITE_NAMES = ("left_foot_site", "right_foot_site")
JINGCHU01_FEET_GEOM_PATTERN = r"^(right|left)_ankle_roll_collision_[0-9]+$"
JINGCHU01_COLLISION_GEOM_PATTERN = r".*_collision_[0-9]+$"

_MOTOR_GROUPS: tuple[tuple[MotorSpec, tuple[str, ...]], ...] = (
  (MOTOR_HIP_ROLL_PITCH, (r".*_hip_roll$", r".*_hip_pitch$", r"waist_roll$")),
  (MOTOR_HIP_YAW, (r".*_hip_yaw$", r"waist_yaw$")),
  (MOTOR_KNEE, (r".*_knee_pitch$",)),
  (MOTOR_ANKLE, (r".*_ankle_pitch$", r".*_ankle_roll$")),
  (MOTOR_SHOULDER, (r".*_shoulder_(pitch|roll)$",)),
  (MOTOR_ELBOW, (r".*_shoulder_yaw$", r".*_elbow_pitch$")),
  (MOTOR_WRIST, (r".*_elbow_yaw$", r".*_wrist_(pitch|roll)$")),
)

STIFFNESS: dict[str, float] = {}
DAMPING: dict[str, float] = {}
EFFORT_LIMIT: dict[str, float] = {}
VELOCITY_LIMIT: dict[str, float] = {}
ARMATURE: dict[str, float] = {}
for _joint in JINGCHU01_JOINT_NAMES:
  for _motor_spec, _patterns in _MOTOR_GROUPS:
    if any(re.fullmatch(pattern, _joint) for pattern in _patterns):
      STIFFNESS[_joint] = _motor_spec.stiffness
      DAMPING[_joint] = _motor_spec.damping
      EFFORT_LIMIT[_joint] = _motor_spec.effort_limit
      VELOCITY_LIMIT[_joint] = _motor_spec.velocity_limit
      ARMATURE[_joint] = _motor_spec.armature
      break
  else:  # pragma: no cover
    raise RuntimeError(f"No Jingchu01 motor assignment for {_joint}")


_HERE = Path(__file__).parent
JINGCHU01_XML: Path = _HERE / "xmls" / "jingchu01.xml"
assert JINGCHU01_XML.exists(), f"Missing Jingchu01 MJCF: {JINGCHU01_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, JINGCHU01_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(JINGCHU01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def _position_actuator(motor: MotorSpec, patterns: tuple[str, ...]) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=patterns,
    stiffness=motor.stiffness,
    damping=motor.damping,
    effort_limit=motor.effort_limit,
    armature=motor.armature,
  )


def _effort_actuator(motor: MotorSpec, patterns: tuple[str, ...]) -> BuiltinMotorActuatorCfg:
  return BuiltinMotorActuatorCfg(
    target_names_expr=patterns,
    effort_limit=motor.effort_limit,
    armature=motor.armature,
  )


JINGCHU01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(_position_actuator(motor, patterns) for motor, patterns in _MOTOR_GROUPS),
  soft_joint_pos_limit_factor=0.9,
)
JINGCHU01_EFFORT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(_effort_actuator(motor, patterns) for motor, patterns in _MOTOR_GROUPS),
  soft_joint_pos_limit_factor=0.9,
)

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    r".*_hip_pitch$": -0.24,
    r".*_knee_pitch$": 0.48,
    r".*_ankle_pitch$": -0.24,
    r".*_elbow_pitch$": 0.3,
    r".*_shoulder_roll$": 0.2,
    r".*_shoulder_pitch$": 0.2,
  },
  joint_vel={r".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(JINGCHU01_COLLISION_GEOM_PATTERN,),
  condim={JINGCHU01_FEET_GEOM_PATTERN: 3, JINGCHU01_COLLISION_GEOM_PATTERN: 1},
  priority={JINGCHU01_FEET_GEOM_PATTERN: 1},
  friction={JINGCHU01_FEET_GEOM_PATTERN: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(JINGCHU01_FEET_GEOM_PATTERN,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_jingchu01_robot_cfg() -> EntityCfg:
  """Return a fresh position-actuated 28-DOF Jingchu01 configuration."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=JINGCHU01_ARTICULATION,
  )


def get_jingchu01_effort_robot_cfg() -> EntityCfg:
  """Return torque-actuated Jingchu01 for explicit HybridMimic PD control."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=JINGCHU01_EFFORT_ARTICULATION,
  )


JINGCHU01_ACTION_SCALE: dict[str, float] = {
  joint: 0.25 * EFFORT_LIMIT[joint] / STIFFNESS[joint]
  for joint in JINGCHU01_JOINT_NAMES
}

# Exact sum of the 29 inertial bodies in the committed MJCF.  The matrix is
# the full composite inertia about the system CoM, evaluated with MuJoCo at
# STANDING_KEYFRAME. It initializes the MPC angular-momentum estimate; its
# configuration dependence is still deliberately not modeled by the QP.
JINGCHU01_TOTAL_MASS = 57.00294
JINGCHU01_CENTROIDAL_INERTIA_BODY: tuple[tuple[float, float, float], ...] = (
  (8.686559, -0.000549, 0.285403),
  (-0.000549, 7.616278, 0.000288),
  (0.285403, 0.000288, 1.642220),
)
