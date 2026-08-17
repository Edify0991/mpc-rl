"""Training configurations for the Unitree G1 29-DOF model.

These factories deliberately do not mutate the THEMIS factories.  They expose
the same motion-reference and MPC-landmark interfaces, so a G1 task can be
selected at registration time without accidentally resolving THEMIS joint or
body names.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import fields

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

from . import hybrid_mimic, mimic_mdp
from . import mpc_grf_mimic_mdp as g1_mpc_mimic_mdp
from . import mpc_grf_parameterized_mdp as g1_mpc_parameterized_mdp
from . import mpc_grf_mdp as g1_mpc_grf_mdp
from .g1.g1_constants import (
  DAMPING,
  G1_ACTION_SCALE,
  G1_CENTROIDAL_INERTIA_BODY,
  G1_CENTROIDAL_BODY_NAMES,
  G1_JOINT_NAMES,
  G1_MPC_FOOT_X_HEEL,
  G1_MPC_FOOT_X_TOE,
  G1_MPC_FOOT_Y_HALF,
  G1_MPC_FZ_MAX_FOOT,
  G1_MPC_MU_FOOT,
  G1_MPC_MU_FOOT_YAW,
  G1_TOTAL_MASS,
  STIFFNESS,
  get_g1_effort_robot_cfg,
  get_g1_robot_cfg,
)
from .hybrid_mimic import HybridMimicActionCfg
from .mimic_mdp import MotionReferenceCommandCfg
from .mpc_grf_mimic_mdp import MimicLocoMPCCommandCfg
from .mpc_grf_mdp import (
  LocoMPCCommandCfg as G1LocoMPCCommandCfg,
  LocoManipMPCCommandCfg as G1LocoManipMPCCommandCfg,
)
from .robot_loco_manipulation import add_push_box_loco_manipulation


G1_FOOT_SITES = ("left_foot", "right_foot")
G1_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def _apply_g1_mpc_grf_features(
  cfg: ManagerBasedRlEnvCfg, *, command_cfg: G1LocoMPCCommandCfg,
) -> ManagerBasedRlEnvCfg:
  """Install the paper-compatible G1 CD-MPC rewards and critic landmarks."""
  cfg.commands["loco_mpc"] = command_cfg
  cfg.rewards["mpc_grf_tracking"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.mpc_grf_tracking, weight=0.0,
    params={"command_name": "loco_mpc", "grf_sensor_name": "feet_ground_contact"},
  )
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["mpc_ang_mom"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.MpcAngMomTracking, weight=0.05,
    params={"command_name": "loco_mpc", "w_k": 1.0, "w_kdot": 0.01,
            "q_k": (1.0, 1.0, 0.5), "q_kdot": (0.1, 0.1, 0.05)},
  )
  cfg.rewards["mpc_com_tracking"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.mpc_com_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "w_pos": 1.0, "w_vel": 0.5},
  )
  cfg.rewards["foot_flat_orientation"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.foot_flat_orientation, weight=0.3,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=G1_FOOT_SITES), "sigma": 0.15},
  )
  for name, func in (
    ("mpc_com_ref", g1_mpc_grf_mdp.mpc_com_ref),
    ("mpc_k_ref", g1_mpc_grf_mdp.mpc_ang_mom_ref),
  ):
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=func, params={"command_name": "loco_mpc"},
    )
  return cfg


def _apply_g1_mpc_grf_v2_features(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Apply the paper's MPC-primary V2 reward weights to the G1 task."""
  for name in ("track_linear_velocity", "track_angular_velocity", "body_ang_vel", "air_time", "foot_slip"):
    cfg.rewards.pop(name, None)
  cfg.rewards["mpc_com_tracking"].weight = 2.0
  cfg.rewards["mpc_com_tracking"].params.update({
    "w_pos": 1.0, "w_vel": 0.0,
    "lookahead_fracs": (0.0, 0.25, 0.5, 0.75, 1.0),
    "lookahead_weights": (0.5, 0.25, 0.15, 0.07, 0.03),
  })
  landmark_params = {
    "command_name": "loco_mpc",
    "lookahead_fracs": (0.0, 0.25, 0.5, 0.75, 1.0),
    "lookahead_weights": (0.5, 0.25, 0.15, 0.07, 0.03),
  }
  cfg.rewards["mpc_com_vel_tracking"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.mpc_com_vel_tracking, weight=2.0,
    params=landmark_params | {"w_vel": 4.0},
  )
  cfg.rewards["mpc_ang_vel_tracking"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.mpc_ang_vel_tracking, weight=4.0,
    params=landmark_params | {"w_ang": 2.0},
  )
  cfg.rewards["mpc_ang_mom"].weight = 0.05
  cfg.rewards["mpc_ang_mom"].params.update({
    "q_k": (1.0, 1.0, 1.0),
    "lookahead_fracs": landmark_params["lookahead_fracs"],
    "lookahead_weights": landmark_params["lookahead_weights"],
  })
  cfg.rewards["mpc_grf_tracking"].weight = 0.002
  cfg.rewards["foot_flat_orientation"].weight = 1.0
  cfg.rewards["mpc_foot_placement"] = RewardTermCfg(
    func=g1_mpc_grf_mdp.mpc_foot_placement_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "sigma": 0.2},
  )
  command = cfg.commands["loco_mpc"]
  command.mpc_horizon = 10
  command.run_every_n_steps = 5
  command.solver_type = "jax_pimpc"
  return cfg


def _g1_mpc_foot_wrench_kwargs() -> dict[str, float]:
  """Return MJCF-derived foot-wrench limits for the G1 centroidal QP."""
  return {
    "foot_x_toe": G1_MPC_FOOT_X_TOE,
    "foot_x_heel": G1_MPC_FOOT_X_HEEL,
    "foot_y_half": G1_MPC_FOOT_Y_HALF,
    "mu_foot": G1_MPC_MU_FOOT,
    "mu_foot_yaw": G1_MPC_MU_FOOT_YAW,
    "fz_max_foot": G1_MPC_FZ_MAX_FOOT,
  }

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


def g1_mpc_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Robot-local port of the paper's velocity-command MPC locomotion task."""
  command = G1LocoMPCCommandCfg(
    debug_vis=play, asset_cfg=SceneEntityCfg("robot", site_names=G1_FOOT_SITES),
    left_foot_site_name="left_foot", right_foot_site_name="right_foot",
    mpc_dt=0.07, mpc_horizon=10, mass=G1_TOTAL_MASS,
    inertia_body=G1_CENTROIDAL_INERTIA_BODY, hip_width=0.15,
    **_g1_mpc_foot_wrench_kwargs(),
    gait_period=0.9, duty_factor=0.5, vel_cmd_name="twist",
    grf_sensor_name="feet_ground_contact", run_every_n_steps=5,
  )
  cfg = _apply_g1_mpc_grf_features(g1_flat_env_cfg(play=play), command_cfg=command)
  return _apply_g1_mpc_grf_v2_features(cfg)


def g1_mpc_loco_manipulation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 port of the original THEMIS MPC-guided push-box task.

  G1 supplies physical palm sites and hand collision geoms, so the hand-force
  part of the centroidal MPC has the same contact interpretation as THEMIS.
  """
  cfg = g1_mpc_locomotion_env_cfg(play=play)
  command = G1LocoManipMPCCommandCfg(
    debug_vis=play,
    mpc_dt=0.07,
    mpc_horizon=10,
    mass=G1_TOTAL_MASS,
    inertia_body=G1_CENTROIDAL_INERTIA_BODY,
    hip_width=0.15,
    **_g1_mpc_foot_wrench_kwargs(),
    gait_period=0.8,
    run_every_n_steps=5,
    asset_cfg=SceneEntityCfg("robot", site_names=G1_FOOT_SITES),
    left_foot_site_name="left_foot",
    right_foot_site_name="right_foot",
    solver_type="jax_pimpc",
  )
  return add_push_box_loco_manipulation(
    cfg,
    command_cfg=command,
    left_hand_site="left_palm",
    right_hand_site="right_palm",
    left_hand_geom="left_hand_collision",
    right_hand_geom="right_hand_collision",
    body_box_geoms=G1_FOOT_GEOMS,
    mpc_mdp=g1_mpc_grf_mdp,
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
  # All three upstream foot rewards receive an empty site list by default.
  # ``foot_swing_height`` is stateful and must have the same two-site order as
  # ``feet_ground_contact``; otherwise it allocates [B, 0] peak heights while
  # the contact sensor emits [B, 2] air/contact flags.
  for term in ("foot_clearance", "foot_swing_height", "foot_slip"):
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
    centroidal_body_names=centroidal_body_names or G1_CENTROIDAL_BODY_NAMES,
    reference_frame_alignment="initial_anchor",
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
  cfg.rewards["tracking_success"] = RewardTermCfg(
    func=mimic_mdp.tracking_success, weight=0.25, params={"command_name": "motion"}
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
  cfg = g1_mpc_locomotion_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_effort_robot_cfg()}
  _add_motion_reference(cfg, motion_file, centroidal_body_names=centroidal_body_names)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionReferenceCommandCfg)
  motion.loop = False
  motion.random_start = not play

  cfg.commands["loco_mpc"] = g1_mpc_parameterized_mdp.as_parameterized_mimic_loco_mpc_cfg(
    g1_mpc_mimic_mdp.as_mimic_loco_mpc_cfg(cfg.commands["loco_mpc"])
  )
  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, g1_mpc_parameterized_mdp.ParameterizedMimicLocoMPCCommandCfg)
  loco_mpc.mass = G1_TOTAL_MASS
  loco_mpc.inertia_body = G1_CENTROIDAL_INERTIA_BODY
  loco_mpc.motion_command_name = "motion"
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
    ("mpc_com_ref", g1_mpc_mimic_mdp.mpc_com_ref),
    ("mpc_k_ref", g1_mpc_mimic_mdp.mpc_ang_mom_ref),
    ("mpc_contact_force_ref", g1_mpc_mimic_mdp.mpc_contact_force_ref),
    ("mpc_contact_plan_ref", g1_mpc_mimic_mdp.mpc_contact_plan_ref),
    ("mpc_contact_plan_valid", g1_mpc_mimic_mdp.mpc_contact_plan_valid),
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
  cfg.rewards["mpc_com_tracking"].weight = 0.0
  cfg.rewards["mpc_com_vel_tracking"].weight = 0.0
  cfg.rewards["mpc_ang_mom"].weight = 0.0
  cfg.rewards["mpc_ang_vel_tracking"].weight = 0.0
  cfg.rewards["mpc_exact_centroidal_landmark"] = RewardTermCfg(
    func=g1_mpc_mimic_mdp.MpcExactCentroidalLandmarkTracking,
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
    func=g1_mpc_parameterized_mdp.FutureContactPlanTracking,
    weight=0.5,
    params={
      "command_name": "loco_mpc", "sensor_name": "feet_ground_contact", "std": 0.25,
      "horizon_discount": 0.9, "force_on_threshold": 20.0, "force_off_threshold": 10.0,
    },
  )
  return cfg


def g1_mpc_rl_mimic_reference_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Parameter-free fixed-reference Mimic MPC validation task.

  It retains reference centroidal/contact landmarks but removes every learned
  contact-plan or MPC-parameter action.  This is the isolated control before
  testing the parameterized extension.
  """
  cfg = g1_mpc_rl_mimic_contact_env_cfg(
    play=play, motion_file=motion_file, centroidal_body_names=centroidal_body_names
  )
  current = cfg.commands["loco_mpc"]
  cfg.commands["loco_mpc"] = MimicLocoMPCCommandCfg(
    **{item.name: getattr(current, item.name) for item in fields(MimicLocoMPCCommandCfg)}
  )
  action = cfg.actions["hybrid_mimic"]
  assert isinstance(action, HybridMimicActionCfg)
  action.enable_contact_state_action = False
  action.contact_plan_horizon = 1
  action.hierarchical_mpc_parameters = False
  cfg.rewards.pop("future_contact_plan", None)
  cfg.rewards.pop("hierarchical_mpc_parameter", None)
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("mpc_hierarchical_parameter_state", None)
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
  assert isinstance(loco_mpc, g1_mpc_parameterized_mdp.ParameterizedMimicLocoMPCCommandCfg)
  loco_mpc.use_hierarchical_parameters = True
  # The high-level parameters are held for five low-level policy steps.  The
  # contact-plan residual remains horizon-valued and is sampled each step.
  loco_mpc.run_every_n_steps = action.high_level_decimation
  for group in ("actor", "critic"):
    for name, func in (
      ("mpc_com_ref", g1_mpc_mimic_mdp.mpc_com_ref),
      ("mpc_k_ref", g1_mpc_mimic_mdp.mpc_ang_mom_ref),
      ("mpc_contact_force_ref", g1_mpc_mimic_mdp.mpc_contact_force_ref),
      ("mpc_hierarchical_parameter_state", g1_mpc_parameterized_mdp.mpc_hierarchical_parameter_state),
    ):
      cfg.observations[group].terms[name] = ObservationTermCfg(
        func=func, params={"command_name": "loco_mpc"}
      )
  cfg.rewards["hierarchical_mpc_parameter"] = RewardTermCfg(
    func=g1_mpc_parameterized_mdp.hierarchical_mpc_parameter_l2,
    weight=-1.0e-3,
    params={"action_name": "hybrid_mimic"},
  )
  return cfg
