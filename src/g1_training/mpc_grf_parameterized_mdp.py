"""Parameterized extension of G1 fixed-plan mimic MPC.

This module is the only G1 MPC path that may load an MPC-parameter network or
accept policy contact-plan/high-level parameter actions.  It subclasses the
parameter-free reference mimic command so the two paths can be tested
independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from g1_mpc.contact_schedule import make_reference_contact_schedule
from . import mpc_grf_mimic_mdp as mimic
from .mpc_parameter_net import MPCParameterBounds, MPCParameters, decode_parameters, nominal_parameters

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ParameterizedMimicLocoMPCCommandCfg(mimic.MimicLocoMPCCommandCfg):
  """Adds bounded parameterization to fixed-plan mimic MPC only."""

  parameter_network_path: str | None = None
  parameter_history_steps: int = 8
  parameter_bounds: MPCParameterBounds = field(default_factory=MPCParameterBounds)
  use_hierarchical_parameters: bool = False
  use_policy_contact_state: bool = False
  policy_contact_gain: float = 0.75
  use_policy_contact_plan: bool = False
  policy_contact_plan_residual_scale: float = 0.75
  preserve_nominal_support: bool = True

  def build(self, env: "ManagerBasedRlEnv") -> "ParameterizedMimicLocoMPCCommand":
    return ParameterizedMimicLocoMPCCommand(self, env)


def as_parameterized_mimic_loco_mpc_cfg(
  cfg: mimic.MimicLocoMPCCommandCfg,
) -> ParameterizedMimicLocoMPCCommandCfg:
  if isinstance(cfg, ParameterizedMimicLocoMPCCommandCfg):
    return cfg
  return ParameterizedMimicLocoMPCCommandCfg(
    **{item.name: getattr(cfg, item.name) for item in fields(cfg)}
  )


class ParameterizedMimicLocoMPCCommand(mimic.MimicLocoMPCCommand):
  """Reference mimic MPC with detached, frozen schedule parameterization."""

  cfg: ParameterizedMimicLocoMPCCommandCfg

  def __init__(self, cfg: ParameterizedMimicLocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)
    B, N = env.num_envs, cfg.mpc_horizon
    self._parameter_history = torch.zeros(B, cfg.parameter_history_steps, 29, device=env.device)
    self._mpc_parameters = nominal_parameters(B, device=env.device, dtype=torch.float32)
    self._hierarchical_parameter_raw = torch.zeros(B, 16, device=env.device)
    self._policy_contact_state = torch.full((B, 2), 0.5, device=env.device)
    self._policy_contact_plan_raw = torch.zeros(B, N, 2, device=env.device)
    self._parameter_net: torch.jit.ScriptModule | None = None
    if cfg.parameter_network_path is not None:
      self._parameter_net = torch.jit.load(cfg.parameter_network_path, map_location=env.device).eval()
      for parameter in self._parameter_net.parameters():
        parameter.requires_grad_(False)

  def set_hierarchical_parameters(self, raw: Tensor) -> None:
    if not self.cfg.use_hierarchical_parameters:
      raise RuntimeError("hierarchical MPC parameters are disabled")
    if raw.shape != self._hierarchical_parameter_raw.shape:
      raise ValueError(f"Expected {tuple(self._hierarchical_parameter_raw.shape)}, got {tuple(raw.shape)}")
    self._hierarchical_parameter_raw.copy_(raw.detach().to(self.device, dtype=torch.float32))

  def reset_hierarchical_parameters(self, env_ids: Tensor | slice | None = None) -> None:
    self._hierarchical_parameter_raw[slice(None) if env_ids is None else env_ids] = 0.0

  def set_policy_contact_state(self, contact_state: Tensor) -> None:
    if not self.cfg.use_policy_contact_state:
      raise RuntimeError("policy contact-state actions are disabled")
    if contact_state.shape != self._policy_contact_state.shape:
      raise ValueError(f"Expected {tuple(self._policy_contact_state.shape)}, got {tuple(contact_state.shape)}")
    self._policy_contact_state.copy_(contact_state.detach().to(self.device, dtype=torch.float32))

  def reset_policy_contact_state(self, env_ids: Tensor | slice | None = None) -> None:
    self._policy_contact_state[slice(None) if env_ids is None else env_ids] = 0.5

  def set_policy_contact_plan(self, raw: Tensor) -> None:
    if not self.cfg.use_policy_contact_plan:
      raise RuntimeError("policy contact-plan actions are disabled")
    if raw.shape != self._policy_contact_plan_raw.shape:
      raise ValueError(f"Expected {tuple(self._policy_contact_plan_raw.shape)}, got {tuple(raw.shape)}")
    self._policy_contact_plan_raw.copy_(raw.detach().to(self.device, dtype=torch.float32))

  def reset_policy_contact_plan(self, env_ids: Tensor | slice | None = None) -> None:
    self._policy_contact_plan_raw[slice(None) if env_ids is None else env_ids] = 0.0

  def _parameter_features(self, x0: Tensor, x_ref: Tensor) -> Tensor:
    contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=x0.dtype)
    try:
      sensor = self._env.scene[self.cfg.grf_sensor_name]
      if sensor.data.force is not None:
        contact = (sensor.data.force.reshape(self.num_envs, 2, 3).norm(dim=-1) > 20.0).to(x0.dtype)
    except KeyError:
      pass
    feature = torch.cat([x0, x_ref[:, 0], x_ref[:, 0] - x0, contact], dim=-1)
    self._parameter_history = torch.cat([self._parameter_history[:, 1:], feature.unsqueeze(1)], dim=1)
    return self._parameter_history

  def _infer_mpc_parameters(self, x0: Tensor, x_ref: Tensor) -> MPCParameters:
    history = self._parameter_features(x0, x_ref)
    if self.cfg.use_hierarchical_parameters:
      return decode_parameters(self._hierarchical_parameter_raw.to(dtype=x0.dtype), self.cfg.parameter_bounds)
    if self._parameter_net is None:
      return nominal_parameters(self.num_envs, device=self.device, dtype=x0.dtype)
    with torch.inference_mode():
      raw = self._parameter_net(history)
    if isinstance(raw, (tuple, list)):
      raw = raw[0]
    if not isinstance(raw, Tensor):
      raise TypeError("parameter network must return tensor [B, 16]")
    return decode_parameters(raw.to(device=self.device, dtype=x0.dtype), self.cfg.parameter_bounds)

  def _parameterize_reference(self, x0: Tensor, x_ref: Tensor) -> Tensor:
    parameters = self._infer_mpc_parameters(x0, x_ref)
    self._mpc_parameters = parameters
    ramp = torch.linspace(0.0, 1.0, x_ref.shape[1], device=self.device, dtype=x_ref.dtype)
    return x_ref + torch.cat([
      torch.zeros_like(x_ref[:, :, :3]),
      ramp.view(1, -1, 1) * parameters.momentum_residual.unsqueeze(1),
    ], dim=-1)

  def _make_reference_schedule(self, **kwargs):
    return make_reference_contact_schedule(
      B=kwargs["B"], N=kwargs["N"],
      reference_contact_state=kwargs["reference_contact_state"],
      reference_r_LF=kwargs["reference_contacts"][:, :, 0],
      reference_r_RF=kwargs["reference_contacts"][:, :, 1],
      R_LF_rot=kwargs["R_lf"], R_RF_rot=kwargs["R_rf"],
      policy_contact_plan_residual=(self._policy_contact_plan_raw if self.cfg.use_policy_contact_plan else None),
      policy_contact_plan_residual_scale=self.cfg.policy_contact_plan_residual_scale,
      preserve_nominal_support=self.cfg.preserve_nominal_support,
      device=self.device,
    )

  def _resample_command(self, env_ids: Tensor) -> None:
    super()._resample_command(env_ids)
    self._parameter_history[env_ids] = 0.0
    self.reset_hierarchical_parameters(env_ids)
    self.reset_policy_contact_state(env_ids)
    self.reset_policy_contact_plan(env_ids)


def mpc_hierarchical_parameter_state(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ParameterizedMimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 16, device=env.device)
  p = term._mpc_parameters
  return torch.cat([p.phase_rate_scale.unsqueeze(-1), p.duty_factor_offset.unsqueeze(-1),
    p.touchdown_mean_residual[:, :, :2].reshape(env.num_envs, 4),
    p.touchdown_std_xy.reshape(env.num_envs, 4), p.momentum_residual], dim=-1)


def hierarchical_mpc_parameter_l2(env: "ManagerBasedRlEnv", action_name: str = "hybrid_mimic") -> Tensor:
  action = env.action_manager.get_term(action_name)
  return torch.zeros(env.num_envs, device=env.device) if not hasattr(action, "held_high_level_action") else action.held_high_level_action.square().mean(dim=-1)


class FutureContactPlanTracking:
  """Delayed physical-contact supervision for the parameterized plan."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
    term = env.command_manager.get_term(cfg.params.get("command_name", "loco_mpc"))
    if not isinstance(term, ParameterizedMimicLocoMPCCommand):
      raise TypeError("FutureContactPlanTracking requires ParameterizedMimicLocoMPCCommand")
    H = term.cfg.mpc_horizon
    self._history = torch.zeros(H, env.num_envs, H, 2, device=env.device)
    self._valid = torch.zeros(H, env.num_envs, H, dtype=torch.bool, device=env.device)
    self._filtered = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    self._cursor = 0

  def __call__(self, env: "ManagerBasedRlEnv", command_name: str = "loco_mpc", sensor_name: str = "feet_ground_contact", std: float = 0.25, horizon_discount: float = 0.9, force_on_threshold: float = 20.0, force_off_threshold: float = 10.0) -> Tensor:
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, ParameterizedMimicLocoMPCCommand):
      return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene[sensor_name]
    if sensor.data.found is None:
      return torch.zeros(env.num_envs, device=env.device)
    found = sensor.data.found
    if found.ndim > 2:
      found = found.any(dim=-1)
    physical = found.to(torch.bool)
    if sensor.data.force is not None:
      force = sensor.data.force.reshape(env.num_envs, 2, 3).norm(dim=-1)
      threshold = torch.where(self._filtered, torch.full_like(force, force_off_threshold), torch.full_like(force, force_on_threshold))
      physical = physical & (force >= threshold)
    self._filtered.copy_(physical)
    self._history[self._cursor].copy_(term._contact_plan_mpc)
    self._valid[self._cursor].copy_(term._contact_plan_valid)
    score = torch.zeros(env.num_envs, device=env.device)
    weight_sum = torch.zeros_like(score)
    H = term.cfg.mpc_horizon
    for age in range(H):
      slot = (self._cursor - age) % H
      valid = self._valid[slot, :, age]
      prediction = self._history[slot, :, age]
      reward = torch.exp(-((prediction - physical.to(prediction.dtype)).square().mean(dim=-1)) / std**2)
      weight = horizon_discount ** age
      score += valid.to(reward.dtype) * weight * reward
      weight_sum += valid.to(reward.dtype) * weight
    self._cursor = (self._cursor + 1) % H
    return score / weight_sum.clamp_min(1.0)

  def reset(self, env_ids: Tensor) -> None:
    self._history[:, env_ids] = 0.0
    self._valid[:, env_ids] = False
    self._filtered[env_ids] = False
