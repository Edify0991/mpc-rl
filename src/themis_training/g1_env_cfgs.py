"""Training configurations for the Unitree G1 29-DOF model.

These factories deliberately do not mutate the THEMIS factories.  They expose
the same motion-reference and MPC-landmark interfaces, so a G1 task can be
selected at registration time without accidentally resolving THEMIS joint or
body names.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import SceneEntityCfg, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import hybrid_mimic, mimic_mdp, mpc_grf_mdp
from .env_cfgs import _apply_mpc_grf_features, _apply_mpc_grf_v2_features
from .g1.g1_constants import (
  DAMPING,
  G1_ACTION_SCALE,
  G1_CENTROIDAL_INERTIA_BODY,
  G1_JOINT_NAMES,
  G1_TOTAL_MASS,
  STIFFNESS,
  get_g1_effort_robot_cfg,
  get_g1_robot_cfg,
)
from .hybrid_mimic import HybridMimicActionCfg
from .mimic_mdp import MotionReferenceCommandCfg
from .mpc_grf_mdp import LocoMPCCommandCfg


G1_FOOT_SITES = ("left_foot", "right_foot")
G1_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

# Ordered only where an input reference has matching G1 body channels.  A
# MotionReferenceCommand performs strict name checking at load time, rather
# than quietly applying a G1 clip to a different morphology.
G1_TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Base G1 velocity task with G1-specific contacts, joint order and gains."""
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 1000
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 96
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=G1_JOINT_NAMES,
    scale=G1_ACTION_SCALE,
    use_default_offset=True,
    preserve_order=True,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES, preserve_order=True)},
    )
    cfg.observations[group].terms["joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES, preserve_order=True)},
    )

  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, self_collision)

  cfg.viewer.body_name = "torso_link"
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.15
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = G1_FOOT_SITES
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = G1_FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = {r".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.15, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15, r".*waist_roll.*": 0.1, r".*waist_pitch.*": 0.1,
    r".*shoulder.*": 0.15, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.25, r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25, r".*waist_roll.*": 0.1, r".*waist_pitch.*": 0.1,
    r".*shoulder.*": 0.25, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
  }
  # This repository's velocity base calls the uprightness term ``upright``
  # (some upstream mjlab versions call the same term body_orientation_l2).
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  for term in ("foot_clearance", "foot_slip"):
    cfg.rewards[term].params["asset_cfg"].site_names = G1_FOOT_SITES
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
  return cfg


def g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-ground G1 baseline used by all G1 motion-imitation tasks."""
  cfg = g1_rough_env_cfg(play=play)
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  if cfg.curriculum is not None:
    cfg.curriculum.pop("terrain_levels", None)
  return cfg


def _add_motion_reference(
  cfg: ManagerBasedRlEnvCfg,
  motion_file: str | None,
  *,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> None:
  # The mjlab registry constructs task configs without function arguments.
  # Keep ``motion_file`` programmatic for experiments, but also allow normal
  # CLI training to select a G1 retargeted clip without editing source.
  motion_file = motion_file or os.environ.get("THEMIS_G1_MOTION_FILE")
  if motion_file is not None and not Path(motion_file).is_file():
    raise FileNotFoundError(
      "THEMIS_G1_MOTION_FILE / motion_file must point to an existing G1-retargeted NPZ: "
      f"{motion_file}"
    )
  cfg.commands["motion"] = MotionReferenceCommandCfg(
    asset_cfg=SceneEntityCfg("robot"),
    motion_file=motion_file,
    joint_names=G1_JOINT_NAMES,
    body_names=G1_TRACKING_BODY_NAMES,
    anchor_body_name="pelvis",
    centroidal_body_names=centroidal_body_names or G1_TRACKING_BODY_NAMES,
    contact_body_names=G1_FOOT_BODIES,
    random_start=False,
    debug_vis=False,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["motion_reference"] = ObservationTermCfg(
      func=mimic_mdp.motion_reference, params={"command_name": "motion"}
    )
    cfg.observations[group].terms["motion_anchor_error"] = ObservationTermCfg(
      func=mimic_mdp.motion_anchor_error, params={"command_name": "motion"}
    )
  cfg.observations["critic"].terms["motion_body_targets"] = ObservationTermCfg(
    func=mimic_mdp.motion_body_targets, params={"command_name": "motion"}
  )
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=mimic_mdp.motion_joint_error_exp, weight=2.0, params={"command_name": "motion", "std": 0.4}
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=mimic_mdp.motion_joint_vel_error_exp, weight=1.0, params={"command_name": "motion", "std": 2.0}
  )
  cfg.rewards["motion_anchor"] = RewardTermCfg(
    func=mimic_mdp.motion_anchor_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}
  )
  cfg.rewards["motion_body"] = RewardTermCfg(
    func=mimic_mdp.motion_relative_body_position_error_exp, weight=2.0, params={"command_name": "motion", "std": 0.3}
  )


def g1_motion_tracker_env_cfg(play: bool = False, motion_file: str | None = None) -> ManagerBasedRlEnvCfg:
  """BeyondMimic-style G1 tracker with standard position actions."""
  cfg = g1_flat_env_cfg(play=play)
  _add_motion_reference(cfg, motion_file)
  cfg.commands["motion"].random_start = not play
  return cfg


def g1_mpc_rl_mimic_contact_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 Phase-1 MPC-RL mimic teacher with future contact-plan action."""
  cfg = _apply_mpc_grf_v2_features(_apply_mpc_grf_features(g1_flat_env_cfg(play=play), play=play))
  cfg.scene.entities = {"robot": get_g1_effort_robot_cfg()}
  _add_motion_reference(cfg, motion_file, centroidal_body_names=centroidal_body_names)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionReferenceCommandCfg)
  motion.loop = False
  motion.random_start = not play

  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, LocoMPCCommandCfg)
  loco_mpc.mass = G1_TOTAL_MASS
  loco_mpc.inertia_body = G1_CENTROIDAL_INERTIA_BODY
  loco_mpc.motion_command_name = "motion"
  loco_mpc.use_reference_contact_schedule = True
  loco_mpc.use_policy_contact_plan = True
  loco_mpc.policy_contact_plan_residual_scale = 0.75
  loco_mpc.run_every_n_steps = 1
  loco_mpc.mpc_dt = 0.07

  cfg.actions = {
    "hybrid_mimic": HybridMimicActionCfg(
      entity_name="robot",
      actuator_names=G1_JOINT_NAMES,
      motion_command_name="motion",
      mpc_command_name="loco_mpc",
      contact_site_names=G1_FOOT_SITES,
      joint_target_residual_scale=0.30,
      enable_contact_state_action=True,
      contact_plan_horizon=loco_mpc.mpc_horizon,
      kp=STIFFNESS,
      kd=DAMPING,
    )
  }
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("motion_reference", None)
    cfg.observations[group].terms.pop("motion_anchor_error", None)
  cfg.observations["actor"].terms["motion_reference_preview"] = ObservationTermCfg(
    func=mimic_mdp.motion_reference_preview,
    params={"command_name": "motion", "frame_offset": 1},
  )
  cfg.observations["critic"].terms["motion_reference_centroidal"] = ObservationTermCfg(
    func=mimic_mdp.motion_reference_centroidal, params={"command_name": "motion"}
  )
  for name, func in (
    ("mpc_com_ref", mpc_grf_mdp.mpc_com_ref),
    ("mpc_k_ref", mpc_grf_mdp.mpc_ang_mom_ref),
    ("mpc_contact_force_ref", mpc_grf_mdp.mpc_contact_force_ref),
    ("mpc_contact_plan_ref", mpc_grf_mdp.mpc_contact_plan_ref),
    ("mpc_contact_plan_valid", mpc_grf_mdp.mpc_contact_plan_valid),
  ):
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=func, params={"command_name": "loco_mpc"}
    )

  cfg.terminations["motion_clip_end"] = TerminationTermCfg(
    func=mimic_mdp.motion_clip_complete, params={"command_name": "motion"}
  )
  # A nonperiodic reference replaces velocity-command / phase-clock rewards.
  for name in ("track_linear_velocity", "track_angular_velocity", "foot_gait", "mpc_foot_placement"):
    if name in cfg.rewards:
      cfg.rewards[name].weight = 0.0
  cfg.rewards["mpc_com_tracking"].weight = 1.0
  cfg.rewards["mpc_com_vel_tracking"].weight = 1.0
  cfg.rewards["mpc_ang_mom"].weight = 0.05
  cfg.rewards["mpc_ang_vel_tracking"].weight = 0.0
  cfg.rewards["mpc_grf_tracking"].weight = 0.05
  cfg.rewards["hybrid_torque"] = RewardTermCfg(
    func=hybrid_mimic.hybrid_torque_l2, weight=-2.0e-5, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["residual_action"] = RewardTermCfg(
    func=hybrid_mimic.residual_action_l2, weight=-0.01, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["future_contact_plan"] = RewardTermCfg(
    func=mpc_grf_mdp.FutureContactPlanTracking,
    weight=0.5,
    params={
      "command_name": "loco_mpc", "sensor_name": "feet_ground_contact", "std": 0.25,
      "horizon_discount": 0.9, "force_on_threshold": 20.0, "force_off_threshold": 10.0,
    },
  )
  return cfg


def g1_mpc_rl_mimic_student_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 Phase-2 causal DAgger/PPO student: only one 29-D joint action."""
  cfg = g1_mpc_rl_mimic_contact_env_cfg(
    play=play, motion_file=motion_file, centroidal_body_names=centroidal_body_names
  )
  action = cfg.actions["hybrid_mimic"]
  assert isinstance(action, HybridMimicActionCfg)
  action.enable_contact_state_action = False
  cfg.commands.pop("loco_mpc", None)
  for name in list(cfg.rewards):
    if name.startswith("mpc_") or name in {"future_contact_plan", "policy_contact_state"}:
      cfg.rewards.pop(name)
  for group in ("actor", "critic"):
    for name in list(cfg.observations[group].terms):
      if name.startswith("mpc_"):
        cfg.observations[group].terms.pop(name)
  return cfg


def g1_hierarchical_hybrid_mimic_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 joint tracker plus slow 16-D bounded MPC-parameter action.

  This preserves the repository's detached-landmark training rule: both the
  fast joint/contact-plan action and the held high-level parameter action are
  optimized by PPO from the common imitation return; gradients never pass
  through the QP or the simulator.
  """
  cfg = g1_mpc_rl_mimic_contact_env_cfg(
    play=play, motion_file=motion_file, centroidal_body_names=centroidal_body_names
  )
  action = cfg.actions["hybrid_mimic"]
  assert isinstance(action, HybridMimicActionCfg)
  action.hierarchical_mpc_parameters = True
  action.high_level_decimation = 5
  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, LocoMPCCommandCfg)
  loco_mpc.use_hierarchical_parameters = True
  # The high-level parameters are held for five low-level policy steps.  The
  # contact-plan residual remains horizon-valued and is sampled each step.
  loco_mpc.run_every_n_steps = action.high_level_decimation
  for group in ("actor", "critic"):
    for name, func in (
      ("mpc_com_ref", mpc_grf_mdp.mpc_com_ref),
      ("mpc_k_ref", mpc_grf_mdp.mpc_ang_mom_ref),
      ("mpc_contact_force_ref", mpc_grf_mdp.mpc_contact_force_ref),
      ("mpc_hierarchical_parameter_state", mpc_grf_mdp.mpc_hierarchical_parameter_state),
    ):
      cfg.observations[group].terms[name] = ObservationTermCfg(
        func=func, params={"command_name": "loco_mpc"}
      )
  cfg.rewards["hierarchical_mpc_parameter"] = RewardTermCfg(
    func=mpc_grf_mdp.hierarchical_mpc_parameter_l2,
    weight=-1.0e-3,
    params={"action_name": "hybrid_mimic"},
  )
  return cfg
