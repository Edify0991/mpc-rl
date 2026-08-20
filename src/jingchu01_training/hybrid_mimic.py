"""Hierarchical HybridMimic action composition.

The low-level action supplies a whole-body tracker correction and a continuous
two-foot contact intention.  Its torque is

``tau = Kp(q_tracker + a_rl - q) + Kd(dq_ref-dq)``.

The low-level policy can also output a continuous two-foot contact intention.
It is frozen before the centroidal QP and supervised by simulated contact;
there is no separately trained contact-prediction network.

Optionally, a held high-level action parameterizes the contact timing,
touchdown distribution, and centroidal-momentum correction used by CD-MPC.
The action is decoded *before* the QP; MPC then emits landmarks that shape the
low-level reward.  There is deliberately no MPC/simulator back-propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul

from .mimic_mdp import MotionReferenceCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HybridMimicActionCfg(ActionTermCfg):
  """Motion-tracking joint action plus optional contact/MPC-plan actions."""

  actuator_names: tuple[str, ...] | list[str]
  motion_command_name: str = "motion"
  mpc_command_name: str = "loco_mpc"
  contact_site_names: tuple[str, ...] = ("left_foot", "right_foot")
  tracker_policy_path: str | None = None
  tracker_action_scale: float = 0.5
  # A nonzero value enables an end-to-end trained joint-target correction on
  # top of the motion-tracker target. Zero leaves pure frozen-tracker PD.
  joint_target_residual_scale: float = 0.0
  kp: float | dict[str, float] = 35.0
  kd: float | dict[str, float] = 4.0
  torque_limit: float | None = None
  # Continuous contact intention w in (0, 1), one value per configured foot.
  # It is a policy action, not a force-sensor prediction network.  The MPC
  # combines it with its nominal schedule before assembling the QP and a
  # reward compares it to the simulated contact state.
  enable_contact_state_action: bool = False
  # One value per foot is the legacy current-contact action. A value larger
  # than one emits a full MPC-horizon contact-plan residual [H, feet].
  contact_plan_horizon: int = 1
  # High-level policy output is the 16-D raw vector decoded by
  # MPCParameterNet.decode_parameters.  It is held for this many policy
  # steps, giving the MPC-contact plan a slower time scale than whole-body
  # residual control.
  hierarchical_mpc_parameters: bool = False
  high_level_decimation: int = 5

  def build(self, env: "ManagerBasedRlEnv") -> "HybridMimicAction":
    return HybridMimicAction(self, env)


class HybridMimicAction(ActionTerm):
  """Applies reference PD tracking and exposes contact/MPC-plan actions."""

  cfg: HybridMimicActionCfg

  def __init__(self, cfg: HybridMimicActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    joint_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_joints = len(joint_ids)
    joint_names = [self._entity.joint_names[int(i)] for i in self._joint_ids]
    self._kp_nominal = self._expand_gain(cfg.kp, joint_names)
    self._kd_nominal = self._expand_gain(cfg.kd, joint_names)
    # Unlike a BuiltinPositionActuator, this action computes its PD torque
    # explicitly. Keep per-environment gain buffers so actuator-gain DR can
    # be applied without mutating an incompatible BuiltinMotorActuator.
    self._kp = self._kp_nominal.expand(self.num_envs, -1).clone()
    self._kd = self._kd_nominal.expand(self.num_envs, -1).clone()
    site_ids, resolved_sites = self._entity.find_sites(cfg.contact_site_names, preserve_order=True)
    if len(site_ids) != len(cfg.contact_site_names):
      raise ValueError(f"Could not resolve contact sites {cfg.contact_site_names}; got {resolved_sites}")
    self._site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    if cfg.high_level_decimation < 1:
      raise ValueError("high_level_decimation must be at least one")
    if cfg.contact_plan_horizon < 1:
      raise ValueError("contact_plan_horizon must be at least one")
    self._num_contacts = len(site_ids)
    next_index = 0
    self._joint_target_slice: slice | None = None
    if cfg.joint_target_residual_scale > 0.0:
      self._joint_target_slice = slice(next_index, next_index + self._num_joints)
      next_index += self._num_joints
    self._contact_state_slice: slice | None = None
    if cfg.enable_contact_state_action:
      contact_plan_dim = self._num_contacts * cfg.contact_plan_horizon
      self._contact_state_slice = slice(next_index, next_index + contact_plan_dim)
      next_index += contact_plan_dim
    self._low_action_dim = next_index
    self._high_level_slice: slice | None = None
    if cfg.hierarchical_mpc_parameters:
      self._high_level_slice = slice(next_index, next_index + 16)
      next_index += 16
    self._action_dim = next_index
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._held_high_level_action = torch.zeros(self.num_envs, 16, device=self.device)
    self._contact_state = torch.full((self.num_envs, self._num_contacts), 0.5, device=self.device)
    self._contact_plan_raw = torch.zeros(
      self.num_envs, cfg.contact_plan_horizon, self._num_contacts, device=self.device
    )
    self._policy_step = 0
    self.last_tau_pd = torch.zeros(self.num_envs, self._num_joints, device=self.device)
    self.last_joint_target_residual = torch.zeros_like(self.last_tau_pd)
    self.last_q_des = torch.zeros_like(self.last_tau_pd)
    self._tracker_prev_action = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    self._tracker = None
    if cfg.tracker_policy_path:
      path = Path(cfg.tracker_policy_path)
      if not path.is_file():
        raise FileNotFoundError(f"Frozen tracker checkpoint not found: {path}")
      self._tracker = torch.jit.load(str(path), map_location=self.device).eval()
      for parameter in self._tracker.parameters():
        parameter.requires_grad_(False)
  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def low_level_raw_action(self) -> torch.Tensor:
    """Action components that directly alter the whole-body controller."""
    return self._raw_actions[:, :self._low_action_dim]

  @property
  def held_high_level_action(self) -> torch.Tensor:
    """Raw MPC-parameter action currently held across low-level steps."""
    return self._held_high_level_action

  @property
  def contact_state(self) -> torch.Tensor:
    """Continuous policy-estimated contact activation in configured foot order."""
    return self._contact_state

  @property
  def contact_plan_raw(self) -> torch.Tensor:
    """Raw full-horizon contact-plan residual, shape ``[B, H, feet]``."""
    return self._contact_plan_raw

  @property
  def q_des(self) -> torch.Tensor:
    """Last desired joint target sent to the explicit PD controller."""
    return self.last_q_des

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions.copy_(actions)
    if self._contact_state_slice is not None:
      mpc = self._env.command_manager.get_term(self.cfg.mpc_command_name)
      raw_plan = actions[:, self._contact_state_slice].reshape(
        self.num_envs, self.cfg.contact_plan_horizon, self._num_contacts
      )
      self._contact_plan_raw.copy_(raw_plan)
      if self.cfg.contact_plan_horizon == 1:
        self._contact_state.copy_(torch.sigmoid(raw_plan[:, 0]))
        if mpc is None or not hasattr(mpc, "set_policy_contact_state"):
          raise TypeError(
            f"{self.cfg.mpc_command_name} must expose set_policy_contact_state() "
            "when contact-state actions are enabled"
          )
        mpc.set_policy_contact_state(self._contact_state)
      else:
        if mpc is None or not hasattr(mpc, "set_policy_contact_plan"):
          raise TypeError(
            f"{self.cfg.mpc_command_name} must expose set_policy_contact_plan() "
            "when contact_plan_horizon is greater than one"
          )
        mpc.set_policy_contact_plan(self._contact_plan_raw)
    if self._high_level_slice is None:
      return
    if self._policy_step % self.cfg.high_level_decimation == 0:
      self._held_high_level_action.copy_(actions[:, self._high_level_slice])
    self._policy_step += 1
    mpc = self._env.command_manager.get_term(self.cfg.mpc_command_name)
    if mpc is None or not hasattr(mpc, "set_hierarchical_parameters"):
      raise TypeError(
        f"{self.cfg.mpc_command_name} must expose set_hierarchical_parameters() "
        "when hierarchical_mpc_parameters is enabled"
      )
    mpc.set_hierarchical_parameters(self._held_high_level_action)

  def _tracker_target(self, command: MotionReferenceCommand) -> torch.Tensor:
    if self._tracker is None:
      # Bootstrap mode is useful for validating the control split; production
      # runs should set tracker_policy_path to a frozen BeyondMimic export.
      return command.joint_pos
    # Ordering mirrors BeyondMimic TrackingEnvCfg.PolicyCfg: command,
    # anchor position/orientation in base frame, base velocity, q/dq, action.
    # That lets a TorchScript export of a matching BeyondMimic tracker be used
    # without an observation-layout adapter.
    anchor_ref_pos = command.body_pos_w[:, command.anchor_index]
    anchor_ref_quat = command.body_quat_w[:, command.anchor_index]
    anchor_pos = command.robot_body_pos_w[:, command.anchor_index]
    anchor_quat = command.robot_body_quat_w[:, command.anchor_index]
    anchor_pos_b = quat_apply_inverse(anchor_quat, anchor_ref_pos - anchor_pos)
    anchor_ori_b = matrix_from_quat(quat_mul(quat_inv(anchor_quat), anchor_ref_quat))[..., :2].reshape(self.num_envs, -1)
    root_quat = self._entity.data.root_link_quat_w
    base_lin_vel = quat_apply_inverse(root_quat, self._entity.data.root_link_lin_vel_w)
    base_ang_vel = quat_apply_inverse(root_quat, self._entity.data.root_link_ang_vel_w)
    q_rel = command.robot_joint_pos - self._entity.data.default_joint_pos[:, command.joint_ids]
    obs = torch.cat([
      command.command, anchor_pos_b, anchor_ori_b, base_lin_vel, base_ang_vel,
      q_rel, command.robot_joint_vel, self._tracker_prev_action,
    ], dim=-1)
    with torch.inference_mode():
      action = self._tracker(obs)
    if isinstance(action, (tuple, list)):
      action = action[0]
    if action.shape[-1] != self._num_joints:
      raise ValueError(f"Tracker output has {action.shape[-1]} actions; expected {self._num_joints}")
    self._tracker_prev_action.copy_(action)
    return command.joint_pos + cfg_scale(self.cfg.tracker_action_scale, action)

  def _expand_gain(self, gain: float | dict[str, float], joint_names: list[str]) -> torch.Tensor:
    if isinstance(gain, (float, int)):
      return torch.full((1, self._num_joints), float(gain), device=self.device)
    missing = [name for name in joint_names if name not in gain]
    if missing:
      raise ValueError(f"PD gain map is missing controlled joints: {missing}")
    return torch.tensor([[gain[name] for name in joint_names]], device=self.device, dtype=torch.float32)

  def randomize_pd_gains(
    self, env_ids: torch.Tensor | slice | None, *, kp_range: tuple[float, float], kd_range: tuple[float, float],
  ) -> None:
    """Scale explicit controller gains independently for each environment.

    This is the effort-actuator counterpart of mjlab's ``dr.pd_gains``.
    It deliberately acts on the controller that produces the commanded torque
    rather than on ``BuiltinMotorActuator``, which has no PD gains to mutate.
    """
    if kp_range[0] <= 0.0 or kd_range[0] <= 0.0 or kp_range[0] > kp_range[1] or kd_range[0] > kd_range[1]:
      raise ValueError("PD gain ranges must be positive ordered intervals")
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    count = self.num_envs if isinstance(ids, slice) else len(ids)
    kp_scale = torch.empty(count, self._num_joints, device=self.device).uniform_(*kp_range)
    kd_scale = torch.empty(count, self._num_joints, device=self.device).uniform_(*kd_range)
    self._kp[ids] = self._kp_nominal * kp_scale
    self._kd[ids] = self._kd_nominal * kd_scale

  def apply_actions(self) -> None:
    command = self._env.command_manager.get_term(self.cfg.motion_command_name)
    if not isinstance(command, MotionReferenceCommand):
      raise TypeError(f"{self.cfg.motion_command_name} must be MotionReferenceCommand")
    q_tracker = self._tracker_target(command)
    if self._joint_target_slice is None:
      self.last_joint_target_residual.zero_()
    else:
      self.last_joint_target_residual = (
        self._raw_actions[:, self._joint_target_slice] * self.cfg.joint_target_residual_scale
      )
      q_tracker = q_tracker + self.last_joint_target_residual
    self.last_q_des.copy_(q_tracker)
    q, dq = command.robot_joint_pos, command.robot_joint_vel
    self.last_tau_pd = self._kp * (q_tracker - q) + self._kd * (command.joint_vel - dq)

    # MPC outputs landmarks only. It never injects a force-to-torque
    # feed-forward term into the actuator command in this formulation.
    tau = self.last_tau_pd
    if self.cfg.torque_limit is not None:
      tau = tau.clamp(-self.cfg.torque_limit, self.cfg.torque_limit)
    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self.last_tau_pd[env_ids] = 0.0
    self.last_q_des[env_ids] = 0.0
    self._tracker_prev_action[env_ids] = 0.0
    self.last_joint_target_residual[env_ids] = 0.0
    self._held_high_level_action[env_ids] = 0.0
    self._contact_state[env_ids] = 0.5
    self._contact_plan_raw[env_ids] = 0.0
    if self._contact_state_slice is not None:
      mpc = self._env.command_manager.get_term(self.cfg.mpc_command_name)
      if mpc is not None and hasattr(mpc, "reset_policy_contact_state"):
        mpc.reset_policy_contact_state(env_ids)
      if mpc is not None and hasattr(mpc, "reset_policy_contact_plan"):
        mpc.reset_policy_contact_plan(env_ids)
    if self._high_level_slice is not None:
      mpc = self._env.command_manager.get_term(self.cfg.mpc_command_name)
      if mpc is not None and hasattr(mpc, "reset_hierarchical_parameters"):
        mpc.reset_hierarchical_parameters(env_ids)


def cfg_scale(scale: float, action: torch.Tensor) -> torch.Tensor:
  return action * scale


def randomize_hybrid_pd_gains(
  env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None,
  action_name: str = "hybrid_mimic", kp_range: tuple[float, float] = (1.0, 1.0),
  kd_range: tuple[float, float] = (1.0, 1.0),
) -> None:
  """MJLab startup event for the explicit HybridMimic PD controller."""
  action = env.action_manager.get_term(action_name)
  if not isinstance(action, HybridMimicAction):
    raise TypeError(f"{action_name!r} must be HybridMimicAction, got {type(action).__name__}")
  action.randomize_pd_gains(env_ids, kp_range=kp_range, kd_range=kd_range)


def hybrid_torque_l2(env: "ManagerBasedRlEnv", action_name: str = "hybrid_mimic") -> torch.Tensor:
  action = env.action_manager.get_term(action_name)
  if not isinstance(action, HybridMimicAction):
    raise TypeError(f"{action_name} must be HybridMimicAction")
  return action.last_tau_pd.square().mean(dim=-1)


def residual_action_l2(env: "ManagerBasedRlEnv", action_name: str = "hybrid_mimic") -> torch.Tensor:
  action = env.action_manager.get_term(action_name)
  if not isinstance(action, HybridMimicAction):
    raise TypeError(f"{action_name} must be HybridMimicAction")
  return action.low_level_raw_action.square().mean(dim=-1)
