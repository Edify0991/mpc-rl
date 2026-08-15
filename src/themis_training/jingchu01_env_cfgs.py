"""Jingchu01 28-DOF velocity and MPC-guided motion-imitation tasks."""

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

from . import hybrid_mimic, mimic_mdp, mpc_grf_mimic_mdp
from .env_cfgs import _apply_mpc_grf_features, _apply_mpc_grf_v2_features
from .hybrid_mimic import HybridMimicActionCfg
from .jingchu01.jingchu01_constants import (
  DAMPING,
  JINGCHU01_ACTION_SCALE,
  JINGCHU01_ANCHOR_BODY_NAME,
  JINGCHU01_BODY_NAMES,
  JINGCHU01_CENTROIDAL_INERTIA_BODY,
  JINGCHU01_FEET_BODY_NAMES,
  JINGCHU01_FEET_GEOM_PATTERN,
  JINGCHU01_FEET_SITE_NAMES,
  JINGCHU01_JOINT_NAMES,
  JINGCHU01_TOTAL_MASS,
  STIFFNESS,
  get_jingchu01_effort_robot_cfg,
  get_jingchu01_robot_cfg,
)
from .mimic_mdp import MotionReferenceCommandCfg
from .mpc_grf_mimic_mdp import MimicLocoMPCCommandCfg


def jingchu01_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Base Jingchu01 velocity configuration with mesh-foot contact sensing."""
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 128
  cfg.scene.entities = {"robot": get_jingchu01_robot_cfg()}
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = JINGCHU01_ANCHOR_BODY_NAME

  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=JINGCHU01_JOINT_NAMES,
    scale=JINGCHU01_ACTION_SCALE,
    use_default_offset=True,
    preserve_order=True,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=JINGCHU01_JOINT_NAMES, preserve_order=True)},
    )
    cfg.observations[group].terms["joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=JINGCHU01_JOINT_NAMES, preserve_order=True)},
    )

  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll|right_ankle_roll)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=JINGCHU01_ANCHOR_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=JINGCHU01_ANCHOR_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, self_collision)
  cfg.viewer.body_name = JINGCHU01_ANCHOR_BODY_NAME
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.25
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = JINGCHU01_FEET_SITE_NAMES
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (JINGCHU01_FEET_GEOM_PATTERN,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)

  # Keep the current repository's velocity-base term name; it is ``upright``
  # rather than ``body_orientation_l2`` in the upstream AMP configuration.
  cfg.rewards["upright"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)
  for term in ("foot_clearance", "foot_slip"):
    cfg.rewards[term].params["asset_cfg"].site_names = JINGCHU01_FEET_SITE_NAMES
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.5,
    params={"sensor_name": self_collision.name, "force_threshold": 10.0},
  )
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  return cfg


def jingchu01_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-ground Jingchu01 baseline used by all registered JC01 tasks."""
  cfg = jingchu01_rough_env_cfg(play=play)
  cfg.sim.njmax = 1500
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
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
  motion_file = motion_file or os.environ.get("THEMIS_JINGCHU01_MOTION_FILE")
  if motion_file is not None and not Path(motion_file).is_file():
    raise FileNotFoundError(
      "THEMIS_JINGCHU01_MOTION_FILE / motion_file must point to a 28-DOF Jingchu01 reference: "
      f"{motion_file}"
    )
  cfg.commands["motion"] = MotionReferenceCommandCfg(
    asset_cfg=SceneEntityCfg("robot"),
    motion_file=motion_file,
    joint_names=JINGCHU01_JOINT_NAMES,
    body_names=JINGCHU01_BODY_NAMES,
    anchor_body_name=JINGCHU01_ANCHOR_BODY_NAME,
    centroidal_body_names=centroidal_body_names or JINGCHU01_BODY_NAMES,
    reference_frame_alignment="initial_anchor",
    contact_body_names=JINGCHU01_FEET_BODY_NAMES,
    contact_point_offsets_b={
      "left_ankle_roll": (0.0, 0.0, -0.04),
      "right_ankle_roll": (0.0, 0.0, -0.04),
    },
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


def jingchu01_motion_tracker_env_cfg(play: bool = False, motion_file: str | None = None) -> ManagerBasedRlEnvCfg:
  """BeyondMimic-style whole-body 28-DOF Jingchu01 tracker."""
  cfg = jingchu01_flat_env_cfg(play=play)
  _add_motion_reference(cfg, motion_file)
  cfg.commands["motion"].random_start = not play
  return cfg


def jingchu01_mpc_rl_mimic_contact_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """JC01 Phase-1 MPC-RL teacher with horizon contact-plan residual action."""
  cfg = _apply_mpc_grf_v2_features(_apply_mpc_grf_features(jingchu01_flat_env_cfg(play=play), play=play))
  cfg.scene.entities = {"robot": get_jingchu01_effort_robot_cfg()}
  _add_motion_reference(cfg, motion_file, centroidal_body_names=centroidal_body_names)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionReferenceCommandCfg)
  motion.loop = False
  motion.random_start = not play
  cfg.commands["loco_mpc"] = mpc_grf_mimic_mdp.as_mimic_loco_mpc_cfg(cfg.commands["loco_mpc"])
  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, MimicLocoMPCCommandCfg)
  loco_mpc.asset_cfg = SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES)
  loco_mpc.left_foot_site_name = "left_foot_site"
  loco_mpc.right_foot_site_name = "right_foot_site"
  loco_mpc.mass = JINGCHU01_TOTAL_MASS
  loco_mpc.inertia_body = JINGCHU01_CENTROIDAL_INERTIA_BODY
  loco_mpc.motion_command_name = "motion"
  loco_mpc.use_reference_contact_schedule = True
  loco_mpc.use_policy_contact_plan = True
  loco_mpc.policy_contact_plan_residual_scale = 0.75
  loco_mpc.run_every_n_steps = 1
  loco_mpc.mpc_dt = 0.07

  cfg.actions = {
    "hybrid_mimic": HybridMimicActionCfg(
      entity_name="robot",
      actuator_names=JINGCHU01_JOINT_NAMES,
      motion_command_name="motion",
      mpc_command_name="loco_mpc",
      contact_site_names=JINGCHU01_FEET_SITE_NAMES,
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
    func=mimic_mdp.motion_reference_preview, params={"command_name": "motion", "frame_offset": 1}
  )
  cfg.observations["critic"].terms["motion_reference_centroidal"] = ObservationTermCfg(
    func=mimic_mdp.motion_reference_centroidal, params={"command_name": "motion"}
  )
  for name, func in (
    ("mpc_com_ref", mpc_grf_mimic_mdp.mpc_com_ref),
    ("mpc_k_ref", mpc_grf_mimic_mdp.mpc_ang_mom_ref),
    ("mpc_contact_force_ref", mpc_grf_mimic_mdp.mpc_contact_force_ref),
    ("mpc_contact_plan_ref", mpc_grf_mimic_mdp.mpc_contact_plan_ref),
    ("mpc_contact_plan_valid", mpc_grf_mimic_mdp.mpc_contact_plan_valid),
  ):
    cfg.observations["critic"].terms[name] = ObservationTermCfg(func=func, params={"command_name": "loco_mpc"})
  cfg.terminations["motion_clip_end"] = TerminationTermCfg(
    func=mimic_mdp.motion_clip_complete, params={"command_name": "motion"}
  )
  for name in ("track_linear_velocity", "track_angular_velocity", "foot_gait", "mpc_foot_placement"):
    if name in cfg.rewards:
      cfg.rewards[name].weight = 0.0
  cfg.rewards["mpc_com_tracking"].weight = 0.0
  cfg.rewards["mpc_com_vel_tracking"].weight = 0.0
  cfg.rewards["mpc_ang_mom"].weight = 0.0
  cfg.rewards["mpc_ang_vel_tracking"].weight = 0.0
  cfg.rewards["mpc_exact_centroidal_landmark"] = RewardTermCfg(
    func=mpc_grf_mimic_mdp.MpcExactCentroidalLandmarkTracking,
    weight=1.0,
    params={
      "command_name": "loco_mpc", "w_com": 4.0, "w_com_vel": 1.0,
      "w_linear_momentum": 0.02, "w_angular_momentum": 0.10,
    },
  )
  cfg.rewards["mpc_grf_tracking"].weight = 0.05
  cfg.rewards["hybrid_torque"] = RewardTermCfg(
    func=hybrid_mimic.hybrid_torque_l2, weight=-2.0e-5, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["residual_action"] = RewardTermCfg(
    func=hybrid_mimic.residual_action_l2, weight=-0.01, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["future_contact_plan"] = RewardTermCfg(
    func=mpc_grf_mimic_mdp.FutureContactPlanTracking,
    weight=0.5,
    params={"command_name": "loco_mpc", "sensor_name": "feet_ground_contact", "std": 0.25,
            "horizon_discount": 0.9, "force_on_threshold": 20.0, "force_off_threshold": 10.0},
  )
  return cfg


def jingchu01_hierarchical_hybrid_mimic_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """JC01 contact-plan teacher augmented by a slow 16-D MPC-parameter action."""
  cfg = jingchu01_mpc_rl_mimic_contact_env_cfg(
    play=play, motion_file=motion_file, centroidal_body_names=centroidal_body_names
  )
  action = cfg.actions["hybrid_mimic"]
  assert isinstance(action, HybridMimicActionCfg)
  action.hierarchical_mpc_parameters = True
  action.high_level_decimation = 5
  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, MimicLocoMPCCommandCfg)
  loco_mpc.use_hierarchical_parameters = True
  loco_mpc.run_every_n_steps = action.high_level_decimation
  for group in ("actor", "critic"):
    for name, func in (
      ("mpc_com_ref", mpc_grf_mimic_mdp.mpc_com_ref),
      ("mpc_k_ref", mpc_grf_mimic_mdp.mpc_ang_mom_ref),
      ("mpc_contact_force_ref", mpc_grf_mimic_mdp.mpc_contact_force_ref),
      ("mpc_hierarchical_parameter_state", mpc_grf_mimic_mdp.mpc_hierarchical_parameter_state),
    ):
      cfg.observations[group].terms[name] = ObservationTermCfg(func=func, params={"command_name": "loco_mpc"})
  cfg.rewards["hierarchical_mpc_parameter"] = RewardTermCfg(
    func=mpc_grf_mimic_mdp.hierarchical_mpc_parameter_l2,
    weight=-1.0e-3,
    params={"action_name": "hybrid_mimic"},
  )
  return cfg


def jingchu01_mpc_rl_mimic_student_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """JC01 Phase-2 causal student: a single 28-D joint-target action."""
  cfg = jingchu01_mpc_rl_mimic_contact_env_cfg(
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
