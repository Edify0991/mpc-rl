"""Jingchu01 28-DOF velocity and MPC-guided motion-imitation tasks."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import SceneEntityCfg, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.envs.mdp import dr
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import hybrid_mimic, mimic_mdp
from . import mpc_grf_mimic_mdp as jingchu01_mpc_mimic_mdp
from . import mpc_grf_parameterized_mdp as jingchu01_mpc_parameterized_mdp
from . import mpc_grf_mdp as jingchu01_mpc_grf_mdp
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
  JINGCHU01_MPC_FOOT_X_HEEL,
  JINGCHU01_MPC_FOOT_X_TOE,
  JINGCHU01_MPC_FOOT_Y_HALF,
  JINGCHU01_MPC_FZ_MAX_FOOT,
  JINGCHU01_MPC_MU_FOOT,
  JINGCHU01_MPC_MU_FOOT_YAW,
  JINGCHU01_TOTAL_MASS,
  STIFFNESS,
  get_jingchu01_effort_robot_cfg,
  get_jingchu01_robot_cfg,
)
from .mimic_mdp import MotionReferenceCommandCfg
from .mpc_grf_mimic_mdp import MimicLocoMPCCommandCfg
from .mpc_grf_mdp import (
  LocoMPCCommandCfg as Jingchu01LocoMPCCommandCfg,
  LocoManipMPCCommandCfg as Jingchu01LocoManipMPCCommandCfg,
)
from .robot_loco_manipulation import add_push_box_loco_manipulation


def _apply_jingchu01_mpc_grf_features(
  cfg: ManagerBasedRlEnvCfg, *, command_cfg: Jingchu01LocoMPCCommandCfg,
) -> ManagerBasedRlEnvCfg:
  """Install the paper-compatible Jingchu01 CD-MPC rewards and landmarks."""
  cfg.commands["loco_mpc"] = command_cfg
  cfg.rewards["mpc_grf_tracking"] = RewardTermCfg(
    func=jingchu01_mpc_grf_mdp.mpc_grf_tracking, weight=0.0,
    params={"command_name": "loco_mpc", "grf_sensor_name": "feet_ground_contact"},
  )
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["mpc_ang_mom"] = RewardTermCfg(
    func=jingchu01_mpc_grf_mdp.MpcAngMomTracking, weight=0.05,
    params={"command_name": "loco_mpc", "w_k": 1.0, "w_kdot": 0.01,
            "q_k": (1.0, 1.0, 0.5), "q_kdot": (0.1, 0.1, 0.05)},
  )
  cfg.rewards["mpc_com_tracking"] = RewardTermCfg(
    func=jingchu01_mpc_grf_mdp.mpc_com_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "w_pos": 1.0, "w_vel": 0.5},
  )
  cfg.rewards["foot_flat_orientation"] = RewardTermCfg(
    func=jingchu01_mpc_grf_mdp.foot_flat_orientation, weight=0.3,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES), "sigma": 0.15},
  )
  for name, func in (
    ("mpc_com_ref", jingchu01_mpc_grf_mdp.mpc_com_ref),
    ("mpc_k_ref", jingchu01_mpc_grf_mdp.mpc_ang_mom_ref),
  ):
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=func, params={"command_name": "loco_mpc"},
    )
  return cfg


def _apply_jingchu01_mpc_grf_v2_features(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Apply the paper's MPC-primary V2 reward weights to Jingchu01."""
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
    func=jingchu01_mpc_grf_mdp.mpc_com_vel_tracking, weight=2.0,
    params=landmark_params | {"w_vel": 4.0},
  )
  cfg.rewards["mpc_ang_vel_tracking"] = RewardTermCfg(
    func=jingchu01_mpc_grf_mdp.mpc_ang_vel_tracking, weight=4.0,
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
    func=jingchu01_mpc_grf_mdp.mpc_foot_placement_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "sigma": 0.2},
  )
  command = cfg.commands["loco_mpc"]
  command.mpc_horizon = 10
  command.run_every_n_steps = 5
  command.solver_type = "jax_pimpc"
  return cfg


def jingchu01_mpc_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Robot-local port of the paper's velocity-command MPC locomotion task."""
  command = Jingchu01LocoMPCCommandCfg(
    debug_vis=play, asset_cfg=SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES),
    left_foot_site_name="left_foot_site", right_foot_site_name="right_foot_site",
    left_foot_geom_names=(r"^left_ankle_roll_collision_[0-9]+$",),
    right_foot_geom_names=(r"^right_ankle_roll_collision_[0-9]+$",),
    mpc_dt=0.07, mpc_horizon=10, mass=JINGCHU01_TOTAL_MASS,
    inertia_body=JINGCHU01_CENTROIDAL_INERTIA_BODY, hip_width=0.15,
    foot_x_toe=JINGCHU01_MPC_FOOT_X_TOE,
    foot_x_heel=JINGCHU01_MPC_FOOT_X_HEEL,
    foot_y_half=JINGCHU01_MPC_FOOT_Y_HALF,
    mu_foot=JINGCHU01_MPC_MU_FOOT,
    mu_foot_yaw=JINGCHU01_MPC_MU_FOOT_YAW,
    fz_max_foot=JINGCHU01_MPC_FZ_MAX_FOOT,
    gait_period=0.9, duty_factor=0.5, vel_cmd_name="twist",
    grf_sensor_name="feet_ground_contact", run_every_n_steps=5,
  )
  cfg = _apply_jingchu01_mpc_grf_features(jingchu01_flat_env_cfg(play=play), command_cfg=command)
  return _apply_jingchu01_mpc_grf_v2_features(cfg)


def jingchu01_mpc_loco_manipulation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Jingchu01 port of the original THEMIS MPC-guided push-box task.

  JC01 has no palm body.  Its explicit ``*_wrist_contact`` sites live at the
  wrist-roll origins and make that modelling assumption visible in the MJCF.
  """
  cfg = jingchu01_mpc_locomotion_env_cfg(play=play)
  command = Jingchu01LocoManipMPCCommandCfg(
    debug_vis=play,
    mpc_dt=0.07,
    mpc_horizon=10,
    mass=JINGCHU01_TOTAL_MASS,
    inertia_body=JINGCHU01_CENTROIDAL_INERTIA_BODY,
    hip_width=0.15,
    foot_x_toe=JINGCHU01_MPC_FOOT_X_TOE,
    foot_x_heel=JINGCHU01_MPC_FOOT_X_HEEL,
    foot_y_half=JINGCHU01_MPC_FOOT_Y_HALF,
    mu_foot=JINGCHU01_MPC_MU_FOOT,
    mu_foot_yaw=JINGCHU01_MPC_MU_FOOT_YAW,
    fz_max_foot=JINGCHU01_MPC_FZ_MAX_FOOT,
    gait_period=0.8,
    run_every_n_steps=5,
    asset_cfg=SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES),
    left_foot_site_name="left_foot_site",
    right_foot_site_name="right_foot_site",
    left_foot_geom_names=(r"^left_ankle_roll_collision_[0-9]+$",),
    right_foot_geom_names=(r"^right_ankle_roll_collision_[0-9]+$",),
    solver_type="jax_pimpc",
  )
  return add_push_box_loco_manipulation(
    cfg,
    command_cfg=command,
    left_hand_site="left_wrist_contact",
    right_hand_site="right_wrist_contact",
    left_hand_geom="left_wrist_roll_collision_0",
    right_hand_geom="right_wrist_roll_collision_0",
    body_box_geoms=(
      "left_ankle_roll_collision_0",
      "right_ankle_roll_collision_0",
      "left_knee_pitch_collision_0",
      "right_knee_pitch_collision_0",
    ),
    mpc_mdp=jingchu01_mpc_grf_mdp,
  )


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
  # Same model-parameter DR as THEMIS.  The batched MPC snapshot is refreshed
  # on reset, so its mass and foot friction are not left at nominal values.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup", func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2), "operation": "scale"},
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.5, 1.5), "operation": "scale"},
  )
  cfg.events["body_inertia"] = EventTermCfg(
    mode="startup", func=dr.pseudo_inertia,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "alpha_range": (-0.112, 0.091)},
  )

  # ``variable_posture`` resolves these regex maps once at construction.  The
  # velocity factory leaves them empty because every robot must supply its own
  # joint naming/groups; an empty map produces a [0]-length std vector and
  # crashes against Jingchu01's 28-DOF joint error.  These values mirror the
  # G1 velocity-task convention: tight standing posture, with progressively
  # more freedom for leg swing and upper-body motion at higher commanded speed.
  cfg.rewards["pose"].params["std_standing"] = {r".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.15, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15, r".*waist_roll.*": 0.1,
    r".*shoulder.*": 0.15, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.25, r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25, r".*waist_roll.*": 0.1,
    r".*shoulder.*": 0.25, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
  }

  # Keep the current repository's velocity-base term name; it is ``upright``
  # rather than ``body_orientation_l2`` in the upstream AMP configuration.
  cfg.rewards["upright"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)
  # Keep site ordering consistent with the two-foot contact sensor.  This is
  # also required by the stateful ``foot_swing_height`` reward.
  for term in ("foot_clearance", "foot_swing_height", "foot_slip"):
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
) -> None:
  """Install the joint-space reference as the sole policy command.

  The velocity-task factory contributes a sampled ``twist`` command together
  with observations, rewards, and curricula that consume it.  A nonperiodic
  motion-mimic task must not retain any of these: its command is the current
  ``[q_ref, dq_ref]`` stream owned by :class:`MotionReferenceCommand`.  In
  particular, ``motion_reference`` intentionally contains no global base
  pose, so this does not reinstate global-root imitation.
  """
  # Remove every velocity-command consumer before removing the producer.  Do
  # not merely set their weights to zero: managers may still evaluate a
  # zero-weight term, and then retain a hidden dependency on ``twist``.
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("command", None)
    cfg.observations[group].terms.pop("phase", None)
  for name, term in tuple(cfg.rewards.items()):
    if isinstance(term.params, dict) and term.params.get("command_name") == "twist":
      cfg.rewards.pop(name)
  if cfg.curriculum is not None:
    for name, term in tuple(cfg.curriculum.items()):
      if isinstance(term.params, dict) and term.params.get("command_name") == "twist":
        cfg.curriculum.pop(name)
  cfg.commands.pop("twist", None)

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
    reference_frame_alignment="initial_anchor",
    random_start=False,
    debug_vis=False,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["motion_reference"] = ObservationTermCfg(
      func=mimic_mdp.motion_reference, params={"command_name": "motion"}
    )
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=mimic_mdp.motion_joint_error_exp, weight=2.0, params={"command_name": "motion", "std": 0.4}
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=mimic_mdp.motion_joint_vel_error_exp, weight=1.0, params={"command_name": "motion", "std": 2.0}
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
  cfg = jingchu01_mpc_locomotion_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_jingchu01_effort_robot_cfg()}
  _add_motion_reference(cfg, motion_file)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionReferenceCommandCfg)
  motion.loop = False
  motion.random_start = not play
  cfg.commands["loco_mpc"] = jingchu01_mpc_parameterized_mdp.as_parameterized_mimic_loco_mpc_cfg(
    jingchu01_mpc_mimic_mdp.as_mimic_loco_mpc_cfg(cfg.commands["loco_mpc"])
  )
  loco_mpc = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc, jingchu01_mpc_parameterized_mdp.ParameterizedMimicLocoMPCCommandCfg)
  # The parameterized command inherits this legacy field from the velocity
  # MPC only for configuration compatibility.  MimicLocoMPCCommand never
  # reads it; setting it to None makes an accidental return to a 3-D twist
  # command fail its construction-time invariant.
  loco_mpc.vel_cmd_name = None
  loco_mpc.asset_cfg = SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES)
  loco_mpc.left_foot_site_name = "left_foot_site"
  loco_mpc.right_foot_site_name = "right_foot_site"
  loco_mpc.mass = JINGCHU01_TOTAL_MASS
  loco_mpc.inertia_body = JINGCHU01_CENTROIDAL_INERTIA_BODY
  loco_mpc.motion_command_name = "motion"
  loco_mpc.centroidal_body_names = centroidal_body_names or JINGCHU01_BODY_NAMES
  loco_mpc.contact_body_names = JINGCHU01_FEET_BODY_NAMES
  # An audited 50-Hz clip can carry an explicit [T,2] contact label.  Keeping
  # this opt-in avoids silently treating an unvalidated height/speed heuristic
  # as a ground-truth MPC schedule.  The command samples it at mpc_dt.
  loco_mpc.reference_contact_key = os.environ.get("THEMIS_JINGCHU01_REFERENCE_CONTACT_KEY")
  loco_mpc.contact_point_offsets_b = {
    "left_ankle_roll": (0.0, 0.0, -0.04),
    "right_ankle_roll": (0.0, 0.0, -0.04),
  }
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
  # This task uses BuiltinMotorActuator plus the explicit PD law in
  # HybridMimicAction.  mjlab.dr.pd_gains only accepts position actuators;
  # randomize the actual torque-controller gains instead.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup", func=hybrid_mimic.randomize_hybrid_pd_gains,
    params={"action_name": "hybrid_mimic", "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2)},
  )
  # _add_motion_reference has installed the current [q_ref, dq_ref] command
  # in both groups.  Do not replace it with a one-frame-ahead preview: the
  # action term tracks the current frame, and an actor that only sees t+1 is
  # unnecessarily partially observed.
  cfg.observations["critic"].terms["mpc_reference_centroidal"] = ObservationTermCfg(
    func=jingchu01_mpc_mimic_mdp.mpc_reference_centroidal, params={"command_name": "loco_mpc"}
  )
  for name, func in (
    ("mpc_com_ref", jingchu01_mpc_mimic_mdp.mpc_com_ref),
    ("mpc_k_ref", jingchu01_mpc_mimic_mdp.mpc_ang_mom_ref),
    ("mpc_contact_force_ref", jingchu01_mpc_mimic_mdp.mpc_contact_force_ref),
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
    func=jingchu01_mpc_mimic_mdp.MpcExactCentroidalLandmarkTracking,
    weight=1.0,
    params={
      "command_name": "loco_mpc", "w_com": 4.0, "w_com_vel": 1.0,
      "w_angular_momentum": 0.10,
    },
  )
  cfg.rewards["mpc_grf_tracking"] = RewardTermCfg(
    func=jingchu01_mpc_mimic_mdp.mpc_grf_tracking_mode_aware,
    weight=0.05,
    params={"command_name": "loco_mpc", "grf_sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["hybrid_torque"] = RewardTermCfg(
    func=hybrid_mimic.hybrid_torque_l2, weight=-2.0e-5, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["residual_action"] = RewardTermCfg(
    func=hybrid_mimic.residual_action_l2, weight=-0.01, params={"action_name": "hybrid_mimic"}
  )
  cfg.rewards["future_contact_plan"] = RewardTermCfg(
    func=jingchu01_mpc_parameterized_mdp.FutureContactPlanTracking,
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
  assert isinstance(loco_mpc, jingchu01_mpc_parameterized_mdp.ParameterizedMimicLocoMPCCommandCfg)
  loco_mpc.use_hierarchical_parameters = True
  loco_mpc.run_every_n_steps = action.high_level_decimation
  for group in ("actor", "critic"):
    for name, func in (
      ("mpc_com_ref", jingchu01_mpc_mimic_mdp.mpc_com_ref),
      ("mpc_k_ref", jingchu01_mpc_mimic_mdp.mpc_ang_mom_ref),
      ("mpc_contact_force_ref", jingchu01_mpc_mimic_mdp.mpc_contact_force_ref),
      ("mpc_hierarchical_parameter_state", jingchu01_mpc_parameterized_mdp.mpc_hierarchical_parameter_state),
    ):
      cfg.observations[group].terms[name] = ObservationTermCfg(func=func, params={"command_name": "loco_mpc"})
  cfg.rewards["hierarchical_mpc_parameter"] = RewardTermCfg(
    func=jingchu01_mpc_parameterized_mdp.hierarchical_mpc_parameter_l2,
    weight=-1.0e-3,
    params={"action_name": "hybrid_mimic"},
  )
  return cfg


def jingchu01_mpc_rl_mimic_reference_env_cfg(
  play: bool = False,
  motion_file: str | None = None,
  centroidal_body_names: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Parameter-free fixed-reference Mimic MPC validation task.

  It retains reference centroidal/contact landmarks but removes every learned
  contact-plan or MPC-parameter action.  This is the isolated control before
  testing the parameterized extension.
  """
  cfg = jingchu01_mpc_rl_mimic_contact_env_cfg(
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
