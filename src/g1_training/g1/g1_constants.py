"""Unitree G1 29-DOF articulation, motor, and collision configuration.

The MJCF and motor grouping are ported from the local G1 model used by the
project's BeyondMimic setup.  This module is intentionally independent of the
THEMIS constants: a task must choose exactly one of the two articulations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from ..model import G1_XML


@dataclass(frozen=True)
class MotorSpec:
  """Joint-side motor data used by MuJoCo and the explicit PD controller."""

  stiffness: float
  damping: float
  effort_limit: float
  velocity_limit: float
  armature: float


# Unitree motor reflected inertias.  The gains follow the same 10 Hz,
# damping-ratio-two construction used by the G1 source configuration.
_OMEGA_N = 2.0 * 3.141592653589793 * 10.0
_DAMPING_RATIO = 2.0


def _motor(armature: float, effort: float, velocity: float, multiplier: float = 1.0) -> MotorSpec:
  joint_armature = armature * multiplier
  return MotorSpec(
    stiffness=joint_armature * _OMEGA_N**2,
    damping=2.0 * _DAMPING_RATIO * joint_armature * _OMEGA_N,
    effort_limit=effort * multiplier,
    velocity_limit=velocity,
    armature=joint_armature,
  )


MOTOR_5020 = _motor(0.003609725, 25.0, 37.0)
MOTOR_7520_14 = _motor(0.010177520, 88.0, 32.0)
MOTOR_7520_22 = _motor(0.025101925, 139.0, 20.0)
MOTOR_5010_16 = _motor(0.0021812, 10.0, 22.0)
# G1 ankle and waist pitch/roll are parallel mechanisms driven by two 5020
# motors.  Until an identified configuration-dependent transmission is
# available, this is the standard nominal 1:1 effective-joint approximation.
MOTOR_5020_DUAL = _motor(0.003609725, 25.0, 37.0, multiplier=2.0)


G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Full inertial body set, distinct from the sparse pose-tracking body set in
# g1_env_cfgs.  Use it for mass-consistent reference centroidal quantities.
G1_CENTROIDAL_BODY_NAMES: tuple[str, ...] = (
  "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
  "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
  "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
  "waist_yaw_link", "waist_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
  "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
  "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
  "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)


def _joint_map(motor: MotorSpec, *patterns: str) -> dict[str, float]:
  return {pattern: motor.stiffness for pattern in patterns}


_MOTOR_GROUPS: tuple[tuple[MotorSpec, tuple[str, ...]], ...] = (
  (MOTOR_7520_14, (r".*_hip_yaw_joint", "waist_yaw_joint")),
  # The public MJCF/source configuration uses the 7520-22 group for all
  # hip pitch/roll and knee joints; retaining it avoids silently changing the
  # validated tracking controller while porting models.
  (MOTOR_7520_22, (r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_knee_joint")),
  (MOTOR_5020_DUAL, (r".*_ankle_pitch_joint", r".*_ankle_roll_joint", "waist_pitch_joint", "waist_roll_joint")),
  (MOTOR_5020, (r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_elbow_joint", r".*_wrist_roll_joint")),
  (MOTOR_5010_16, (r".*_wrist_pitch_joint", r".*_wrist_yaw_joint")),
)

STIFFNESS: dict[str, float] = {}
DAMPING: dict[str, float] = {}
EFFORT_LIMIT: dict[str, float] = {}
VELOCITY_LIMIT: dict[str, float] = {}
ARMATURE: dict[str, float] = {}
for _name in G1_JOINT_NAMES:
  for _motor_spec, _patterns in _MOTOR_GROUPS:
    if any(re.fullmatch(pattern, _name) for pattern in _patterns):
      STIFFNESS[_name] = _motor_spec.stiffness
      DAMPING[_name] = _motor_spec.damping
      EFFORT_LIMIT[_name] = _motor_spec.effort_limit
      VELOCITY_LIMIT[_name] = _motor_spec.velocity_limit
      ARMATURE[_name] = _motor_spec.armature
      break
  else:  # pragma: no cover - protects future joint-list changes.
    raise RuntimeError(f"No G1 motor assignment for {_name}")


assert G1_XML.exists(), f"Missing G1 MJCF: {G1_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
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


G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(_position_actuator(motor, patterns) for motor, patterns in _MOTOR_GROUPS),
  soft_joint_pos_limit_factor=0.9,
)

# HybridMimic computes tau_PD itself and consequently needs effort motors.
G1_EFFORT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(_effort_actuator(motor, patterns) for motor, patterns in _MOTOR_GROUPS),
  soft_joint_pos_limit_factor=0.9,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.78),
  joint_pos={
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={r".*": 0.0},
)

# All body collisions stay enabled for general motion imitation.  Foot geoms
# have condim=3 and valid friction; other self contacts retain condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, r".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_g1_robot_cfg() -> EntityCfg:
  """Return a fresh position-actuated G1 configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_effort_robot_cfg() -> EntityCfg:
  """Return G1 with effort motors for :class:`HybridMimicAction`."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_EFFORT_ARTICULATION,
  )


# Position-action perturbation is a quarter of the nominal torque range in
# displacement units.  HybridMimic does not use this table; it uses STIFFNESS
# and DAMPING directly in its explicit PD law.
G1_ACTION_SCALE: dict[str, float] = {
  joint: 0.25 * EFFORT_LIMIT[joint] / STIFFNESS[joint]
  for joint in G1_JOINT_NAMES
}

# Centroidal-MPC parameters. The 33.341142 kg mass is the exact sum of the
# inertial masses in the committed G1 MJCF. The inertia is deliberately a
# documented initial composite-inertia approximation, not an identified full
# configuration-dependent centroidal inertia.
G1_TOTAL_MASS = 33.341142
G1_CENTROIDAL_INERTIA_BODY: tuple[tuple[float, float, float], ...] = (
  (1.20, 0.0, 0.0),
  (0.0, 1.45, 0.0),
  (0.0, 0.0, 0.75),
)
