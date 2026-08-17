"""Robot-parameterized port of the THEMIS push-box task scaffold.

This module contains scene/reward plumbing only.  The centroidal MPC command
class and all morphology data are supplied by the robot-local package, keeping
G1 and Jingchu01 from importing THEMIS's MPC or contact names.
"""

from __future__ import annotations

from typing import Any

from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import push_box_mdp


def add_push_box_loco_manipulation(
  cfg: Any,
  *,
  command_cfg: Any,
  left_hand_site: str,
  right_hand_site: str,
  left_hand_geom: str,
  right_hand_geom: str,
  body_box_geoms: tuple[str, ...],
  mpc_mdp: Any,
) -> Any:
  """Add the original THEMIS box-pushing task terms to a robot MPC env.

  ``command_cfg`` must be the robot-local ``LocoManipMPCCommandCfg``.  Contact
  points are named explicitly rather than inferred from a link, so the model
  assumption is auditable: G1 uses palm sites, while JC01 uses documented
  wrist-origin sites.
  """
  cfg.scene.entities["box"] = EntityCfg(spec_fn=push_box_mdp.get_push_box_spec)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.__class__ = push_box_mdp.ModeGatedVelocityCommandCfg
  twist.ranges.lin_vel_x = (-0.5, 0.75)
  twist.ranges.lin_vel_y = (-0.5, 0.5)
  twist.ranges.ang_vel_z = (-0.75, 0.75)
  twist.rel_standing_envs = 0.0

  lhand = ContactSensorCfg(
    name="lhand_box_contact",
    primary=ContactMatch(mode="geom", pattern=left_hand_geom, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found", "force"), reduce="maxforce", num_slots=1,
  )
  rhand = ContactSensorCfg(
    name="rhand_box_contact",
    primary=ContactMatch(mode="geom", pattern=right_hand_geom, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found", "force"), reduce="maxforce", num_slots=1,
  )
  body = ContactSensorCfg(
    name="body_box_contact",
    primary=ContactMatch(mode="geom", pattern=body_box_geoms, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    # ``leg_box_collision_cost`` uses the contact-match count, while the
    # loco-manipulation MPC reads the net force from this same sensor.
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (lhand, rhand, body)

  command_cfg.hand_site_names = (left_hand_site, right_hand_site)
  command_cfg.lhand_box_sensor_name = lhand.name
  command_cfg.rhand_box_sensor_name = rhand.name
  command_cfg.body_box_sensor_name = body.name
  cfg.commands["loco_mpc"] = command_cfg

  cfg.observations["critic"].terms["box_pose_rel"] = ObservationTermCfg(
    func=push_box_mdp.box_pose_rel_priv, params={"box_name": "box", "robot_name": "robot"}
  )
  cfg.observations["critic"].terms["box_lin_vel"] = ObservationTermCfg(
    func=push_box_mdp.box_lin_vel_priv, params={"box_name": "box", "robot_name": "robot"}
  )
  cfg.observations["critic"].terms["box_size"] = ObservationTermCfg(
    func=push_box_mdp.box_size_priv, params={"box_name": "box", "geom_name": "box_geom"}
  )
  cfg.observations["critic"].terms["hand_box_contact"] = ObservationTermCfg(
    func=push_box_mdp.hand_box_contact_priv,
    params={"lhand_sensor": lhand.name, "rhand_sensor": rhand.name},
  )

  cfg.rewards["hand_box_contact"] = RewardTermCfg(
    func=push_box_mdp.hand_box_contact, weight=1.0,
    params={"lhand_sensor": lhand.name, "rhand_sensor": rhand.name, "both_hands_bonus": 0.5},
  )
  cfg.rewards["box_com_tracking"] = RewardTermCfg(
    func=push_box_mdp.box_com_tracking, weight=4.0,
    params={
      "command_name": "loco_mpc", "box_name": "box",
      "lhand_sensor": lhand.name, "rhand_sensor": rhand.name,
      "w_pos": 2.0, "w_vel": 0.5,
      "lookahead_fracs": (0.0, 0.25, 0.5, 0.75, 1.0),
      "lookahead_weights": (0.35, 0.25, 0.20, 0.12, 0.08),
    },
  )
  cfg.rewards["push_velocity_match"] = RewardTermCfg(
    func=push_box_mdp.push_velocity_match, weight=5.0,
    params={"command_name": "twist", "box_name": "box", "robot_name": "robot",
            "lhand_sensor": lhand.name, "rhand_sensor": rhand.name, "sigma": 0.3},
  )
  cfg.rewards["robot_box_velocity_match"] = RewardTermCfg(
    func=push_box_mdp.robot_box_velocity_match, weight=2.5,
    params={"box_name": "box", "robot_name": "robot", "lhand_sensor": lhand.name,
            "rhand_sensor": rhand.name, "sigma": 0.3},
  )
  cfg.rewards["robot_box_xy_distance"] = RewardTermCfg(
    func=push_box_mdp.robot_box_xy_distance_cost, weight=-1.0,
    params={"box_name": "box", "robot_name": "robot", "target_distance": 0.5},
  )
  cfg.rewards["robot_box_yaw"] = RewardTermCfg(
    func=push_box_mdp.robot_box_yaw_cost, weight=-0.5,
    params={"box_name": "box", "robot_name": "robot"},
  )
  cfg.rewards["leg_box_collision"] = RewardTermCfg(
    func=push_box_mdp.leg_box_collision_cost, weight=-1.0,
    params={"sensor_name": body.name},
  )
  cfg.rewards["mpc_hand_force_tracking"] = RewardTermCfg(
    func=mpc_mdp.mpc_hand_force_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "lhand_sensor_name": lhand.name,
            "rhand_sensor_name": rhand.name, "sigma": 50.0},
  )

  cfg.events["init_push_mode"] = EventTermCfg(
    mode="startup", func=push_box_mdp.init_push_mode, params={"push_fraction": 0.5}
  )
  cfg.events["reset_box"] = EventTermCfg(
    mode="reset", func=push_box_mdp.reset_box_pose_and_mass,
    params={"x_range": (0.9, 1.5), "y_range": (0.0, 0.0), "yaw_range": (0.0, 0.0),
            "mass_range": (3.0, 15.0), "z_clearance": 0.03, "box_name": "box", "geom_name": "box_geom"},
  )
  cfg.events["reset_box_friction"] = EventTermCfg(
    mode="reset", func=push_box_mdp.randomize_box_friction_curriculum,
    params={"default_friction_range": (0.05, 0.5), "box_name": "box", "geom_name": "box_geom"},
  )
  cfg.terminations["robot_far_from_box"] = TerminationTermCfg(
    func=push_box_mdp.robot_far_from_box, params={"max_distance": 2.0, "box_name": "box", "robot_name": "robot"}
  )
  cfg.terminations["box_toppled"] = TerminationTermCfg(
    func=push_box_mdp.box_toppled, params={"max_tilt_rad": 1.0, "box_name": "box"}
  )
  cfg.terminations["box_pitched"] = TerminationTermCfg(
    func=push_box_mdp.box_pitched, params={"max_pitch_rad": 0.1745, "box_name": "box"}
  )
  return cfg
