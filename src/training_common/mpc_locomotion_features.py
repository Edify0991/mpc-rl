"""Robot-neutral MPC-landmark reward and observation configuration.

Robot packages construct and pass their own ``LocoMPCCommandCfg``.  This
module only installs the shared paper reward structure; it never imports a
robot-specific MPC implementation or uses a fixed foot name.
"""

from __future__ import annotations

from typing import Any

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import SceneEntityCfg


def apply_mpc_grf_features(
  cfg: Any,
  *,
  command_cfg: Any,
  mpc_mdp: Any,
  foot_sites: tuple[str, str],
) -> Any:
  """Install the paper's v1 CD-MPC reward/critic-landmark scaffold."""
  cfg.commands["loco_mpc"] = command_cfg
  cfg.rewards["mpc_grf_tracking"] = RewardTermCfg(
    func=mpc_mdp.mpc_grf_tracking, weight=0.0,
    params={"command_name": "loco_mpc", "grf_sensor_name": "feet_ground_contact"},
  )
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["mpc_ang_mom"] = RewardTermCfg(
    func=mpc_mdp.MpcAngMomTracking, weight=0.05,
    params={"command_name": "loco_mpc", "w_k": 1.0, "w_kdot": 0.01,
            "q_k": (1.0, 1.0, 0.5), "q_kdot": (0.1, 0.1, 0.05)},
  )
  cfg.rewards["mpc_com_tracking"] = RewardTermCfg(
    func=mpc_mdp.mpc_com_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "w_pos": 1.0, "w_vel": 0.5},
  )
  cfg.rewards["foot_flat_orientation"] = RewardTermCfg(
    func=mpc_mdp.foot_flat_orientation, weight=0.3,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=foot_sites), "sigma": 0.15},
  )
  for name, func in (
    ("mpc_com_ref", mpc_mdp.mpc_com_ref),
    ("mpc_k_ref", mpc_mdp.mpc_ang_mom_ref),
    ("mpc_contact_force_ref", mpc_mdp.mpc_contact_force_ref),
    ("mpc_contact_moment_ref", mpc_mdp.mpc_contact_moment_ref),
  ):
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=func, params={"command_name": "loco_mpc"}
    )
  return cfg


def apply_mpc_grf_v2_features(cfg: Any, *, mpc_mdp: Any) -> Any:
  """Apply the paper's v2 MPC-primary reward weights to a v1 scaffold."""
  for name in ("track_linear_velocity", "track_angular_velocity", "body_ang_vel", "air_time", "foot_slip"):
    cfg.rewards.pop(name, None)
  com = cfg.rewards["mpc_com_tracking"]
  com.weight = 2.0
  com.params.update({
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
    func=mpc_mdp.mpc_com_vel_tracking, weight=2.0,
    params=landmark_params | {"w_vel": 4.0},
  )
  cfg.rewards["mpc_ang_vel_tracking"] = RewardTermCfg(
    func=mpc_mdp.mpc_ang_vel_tracking, weight=4.0,
    params=landmark_params | {"w_ang": 2.0},
  )
  ang = cfg.rewards["mpc_ang_mom"]
  ang.weight = 0.05
  ang.params.update({
    "q_k": (1.0, 1.0, 1.0),
    "lookahead_fracs": landmark_params["lookahead_fracs"],
    "lookahead_weights": landmark_params["lookahead_weights"],
  })
  cfg.rewards["mpc_grf_tracking"].weight = 0.002
  cfg.rewards["foot_flat_orientation"].weight = 1.0
  cfg.rewards["mpc_foot_placement"] = RewardTermCfg(
    func=mpc_mdp.mpc_foot_placement_tracking, weight=1.0,
    params={"command_name": "loco_mpc", "sigma": 0.2},
  )
  command = cfg.commands["loco_mpc"]
  command.mpc_horizon = 10
  command.run_every_n_steps = 5
  command.solver_type = "jax_pimpc"
  return cfg
