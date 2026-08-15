"""Mimic-specific extension of the original centroidal MPC-GRF MDP.

The original :mod:`mpc_grf_mdp` is intentionally kept as the reproduction
baseline for the MPC-RL paper.  This module owns only the proposed
motion-reference extensions: exact articulated centroidal state, mass-
consistent reference targets, and the corresponding landmark reward.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from themis_training import mpc_grf_mdp as baseline
from themis_training.reference_centroidal import (
  CentroidalState,
  ReferenceCentroidalTrajectory,
  compute_centroidal_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


@dataclass(kw_only=True)
class MimicLocoMPCCommandCfg(baseline.LocoMPCCommandCfg):
  """Configuration that opts a task into the mimic-only MPC extension."""

  def build(self, env: "ManagerBasedRlEnv") -> "MimicLocoMPCCommand":
    return MimicLocoMPCCommand(self, env)


def as_mimic_loco_mpc_cfg(cfg: baseline.LocoMPCCommandCfg) -> MimicLocoMPCCommandCfg:
  """Copy a baseline MPC configuration into the explicit mimic extension.

  The conversion is deliberately shallow: nested configuration objects are
  already immutable-by-convention manager configuration data and should keep
  their identity.  This lets a robot task reuse the paper's baseline defaults
  while selecting a different command implementation only for mimic training.
  """
  if isinstance(cfg, MimicLocoMPCCommandCfg):
    return cfg
  return MimicLocoMPCCommandCfg(**{field.name: getattr(cfg, field.name) for field in fields(cfg)})


def apply_mimic_centroidal_reference_to_mpc_target(
  x_ref: Tensor, reference: ReferenceCentroidalTrajectory,
) -> tuple[Tensor, Tensor, Tensor]:
  """Write the exact reference state and control-stage contacts into MPC data."""
  if x_ref.ndim != 3 or x_ref.shape[-1] != 9:
    raise ValueError("x_ref must have shape [B, N + 1, 9]")
  expected = x_ref.shape[:2]
  for name, value in (
    ("com_pos_w", reference.com_pos_w),
    ("com_vel_w", reference.com_vel_w),
    ("linear_momentum_w", reference.linear_momentum_w),
    ("angular_momentum_w", reference.angular_momentum_w),
  ):
    if value.shape != (*expected, 3):
      raise ValueError(f"reference.{name} must have shape {(*expected, 3)}, got {tuple(value.shape)}")
  if reference.contact_pos_w.shape[:2] != expected:
    raise ValueError("reference.contact_pos_w must share x_ref batch/horizon dimensions")
  x_ref[:, :, :3] = reference.com_pos_w
  x_ref[:, :, 3:6] = reference.linear_momentum_w
  x_ref[:, :, 6:9] = reference.angular_momentum_w
  return x_ref, reference.com_vel_w, reference.contact_pos_w[:, :-1]


class MimicLocoMPCCommand(baseline.LocoMPCCommand):
  """CD-MPC command with motion-derived targets and exact online centroidal x0.

  It subclasses the paper baseline so the contact-plan action, QP construction,
  solver, rollout storage, and all non-mimic public APIs remain identical.
  """

  cfg: MimicLocoMPCCommandCfg

  def __init__(self, cfg: MimicLocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)
    body_ids = self._robot.indexing.body_ids.detach().cpu().numpy()
    if len(body_ids) != self._robot.data.body_link_pos_w.shape[1]:
      raise RuntimeError("Articulation body indexing and MJLab body kinematics disagree")
    model = env.sim.mj_model

    def model_field(name: str) -> Tensor:
      return torch.as_tensor(np.asarray(getattr(model, name))[body_ids], device=env.device, dtype=torch.float32)

    self._online_body_mass = model_field("body_mass")
    self._online_body_com_offset_b = model_field("body_ipos")
    self._online_body_inertia_diag = model_field("body_inertia")
    self._online_body_inertial_quat_b = model_field("body_iquat")
    self._linear_momentum_traj = torch.zeros_like(self._vel_traj)
    self._linear_momentum_mpc_target = torch.zeros_like(self._com_mpc_target)

  def current_centroidal_state(self) -> CentroidalState:
    """Read the exact whole-body centroidal state in simulator-world axes."""
    data = self._robot.data
    dtype = data.body_link_pos_w.dtype
    return compute_centroidal_state(
      body_link_pos_w=data.body_link_pos_w,
      body_link_quat_w=data.body_link_quat_w,
      body_com_lin_vel_w=data.body_com_lin_vel_w,
      body_link_ang_vel_w=data.body_link_ang_vel_w,
      body_mass=self._online_body_mass.to(dtype=dtype),
      body_com_offset_b=self._online_body_com_offset_b.to(dtype=dtype),
      body_inertia_diag=self._online_body_inertia_diag.to(dtype=dtype),
      body_inertial_quat_b=self._online_body_inertial_quat_b.to(dtype=dtype),
    )

  def _interpolate_traj_refs(self) -> None:
    super()._interpolate_traj_refs()
    N = self.cfg.mpc_horizon
    t_frac = self.cfg.tracking_lookahead_frac * (N - 1) + self._traj_step * self._env.step_dt / self.cfg.mpc_dt
    idx = min(max(int(t_frac), 0), N - 2)
    alpha = min(max(t_frac - idx, 0.0), 1.0)
    self._linear_momentum_mpc_target = (
      (1.0 - alpha) * self._linear_momentum_traj[:, idx]
      + alpha * self._linear_momentum_traj[:, idx + 1]
    )

  def _resample_command(self, env_ids: Tensor) -> None:
    super()._resample_command(env_ids)
    self._linear_momentum_traj[env_ids] = 0.0
    self._linear_momentum_mpc_target[env_ids] = 0.0

  def _update_command(self) -> None:
    """Solve the same QP as the baseline using exact mimic centroidal data."""
    self._step_count += 1
    self._traj_step += 1
    self._interpolate_traj_refs()
    if self._step_count % self.cfg.run_every_n_steps != 0:
      return

    cfg, B, N, dt, device, robot = (
      self.cfg, self.num_envs, self.cfg.mpc_horizon, self.cfg.mpc_dt, self.device, self._robot,
    )
    online = self.current_centroidal_state()
    c, linear_momentum, k = online.com_pos_w, online.linear_momentum_w, online.angular_momentum_w
    x0 = torch.cat([c, linear_momentum, k], dim=-1)

    site_pos = robot.data.site_pos_w[:, self._site_ids, :]
    r_lf = site_pos[:, self._lf_site_local_idx, :]
    r_rf = site_pos[:, self._rf_site_local_idx, :]
    vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
    if vel_cmd is None:
      vx_body = torch.zeros(B, device=device)
      vy_body = torch.zeros(B, device=device)
      wz = torch.zeros(B, device=device)
    else:
      vx_body, vy_body, wz = vel_cmd[:, 0], vel_cmd[:, 1], vel_cmd[:, 2]

    quat_w = robot.data.root_link_quat_w
    q_w, q_x, q_y, q_z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    yaw = torch.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
    cos_y, sin_y = yaw.cos(), yaw.sin()
    vx, vy = cos_y * vx_body - sin_y * vy_body, sin_y * vx_body + cos_y * vy_body
    site_quat = robot.data.site_quat_w[:, self._site_ids, :]
    R_lf = baseline._quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
    R_rf = baseline._quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

    k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
    yaw_k = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
    vx_w_k = yaw_k.cos() * vx_body.unsqueeze(1) - yaw_k.sin() * vy_body.unsqueeze(1)
    vy_w_k = yaw_k.sin() * vx_body.unsqueeze(1) + yaw_k.cos() * vy_body.unsqueeze(1)
    x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
    x_ref[:, 0, :2] = c[:, :2]
    x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
    x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
    x_ref[:, :, 2] = robot.data.default_root_state[:, 2:3]
    x_ref[:, :, 3] = vx_w_k * cfg.mass
    x_ref[:, :, 4] = vy_w_k * cfg.mass
    x_ref[:, :, 5] = 0.0
    x_ref[:, :, 6:8] = 0.0
    x_ref[:, :, 8] = (float(self._I_approx[2, 2]) * wz).unsqueeze(1)

    reference_contacts: Tensor | None = None
    reference_contact_state: Tensor | None = None
    reference_contact_valid: Tensor | None = None
    if cfg.motion_command_name is not None:
      motion = self._env.command_manager.get_term(cfg.motion_command_name)
      if motion is None or not hasattr(motion, "centroidal_horizon"):
        raise ValueError(f"motion_command_name={cfg.motion_command_name!r} must provide centroidal_horizon(steps, dt)")
      if hasattr(motion, "reference_centroidal_horizon"):
        reference = motion.reference_centroidal_horizon(N + 1, dt)
        x_ref, ref_vel, reference_contacts = apply_mimic_centroidal_reference_to_mpc_target(x_ref, reference)
        reference_contact_state = motion.reference_contact_horizon(N, dt)
        reference_contact_valid = motion.reference_horizon_valid(N, dt)
      else:
        ref_pos, ref_vel, ref_ang = motion.centroidal_horizon(N + 1, dt)
        x_ref[:, :, :3] = ref_pos
        x_ref[:, :, 3:6] = ref_vel * cfg.mass
        x_ref[:, :, 6:9] = ref_ang @ self._I_approx
      vx, vy = ref_vel[:, 0, 0], ref_vel[:, 0, 1]

    parameters = self._infer_mpc_parameters(x0, x_ref)
    self._mpc_parameters = parameters
    ramp = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=x_ref.dtype)
    x_ref[:, :, 3:9] += ramp.view(1, -1, 1) * parameters.momentum_residual.unsqueeze(1)
    if cfg.use_reference_contact_schedule:
      if reference_contacts is None or reference_contact_state is None:
        raise ValueError("use_reference_contact_schedule requires a MotionReferenceCommand with two contact bodies")
      if reference_contacts.shape[2] != 2 or reference_contact_state.shape[2] != 2:
        raise ValueError("The bipedal centroidal MPC requires exactly [left, right] reference contacts")
      schedule = baseline.make_reference_contact_schedule(
        B=B, N=N, reference_contact_state=reference_contact_state,
        reference_r_LF=reference_contacts[:, :, 0], reference_r_RF=reference_contacts[:, :, 1],
        R_LF_rot=R_lf, R_RF_rot=R_rf,
        policy_contact_state=(self._policy_contact_state if cfg.use_policy_contact_state else None),
        policy_contact_gain=cfg.policy_contact_gain, policy_contact_horizon_decay=cfg.policy_contact_horizon_decay,
        policy_contact_plan_residual=(self._policy_contact_plan_raw if cfg.use_policy_contact_plan else None),
        policy_contact_plan_residual_scale=cfg.policy_contact_plan_residual_scale,
        preserve_nominal_support=cfg.preserve_nominal_support, device=device,
      )
    else:
      phase = getattr(self._env, "_gait_phase", torch.zeros(B, device=device))
      v_cmd = torch.zeros(B, 3, device=device)
      v_cmd[:, 0], v_cmd[:, 1] = vx, vy
      schedule = baseline.make_walking_schedule(
        B=B, N=N, r_LF=r_lf, r_RF=r_rf, gait_phase=phase, period=cfg.gait_period, dt=dt,
        duty_factor=cfg.duty_factor, com_pos=c, v_cmd=v_cmd, yaw=yaw, yaw_rate=wz, hip_width=cfg.hip_width,
        R_LF_rot=R_lf, R_RF_rot=R_rf, phase_rate_scale=parameters.phase_rate_scale,
        duty_factor_offset=parameters.duty_factor_offset,
        touchdown_residual_LF=parameters.touchdown_mean_residual[:, 0],
        touchdown_residual_RF=parameters.touchdown_mean_residual[:, 1],
        touchdown_std_LF_xy=parameters.touchdown_std_xy[:, 0],
        touchdown_std_RF_xy=parameters.touchdown_std_xy[:, 1],
        reference_r_LF=(None if reference_contacts is None else reference_contacts[:, :, 0]),
        reference_r_RF=(None if reference_contacts is None else reference_contacts[:, :, 1]),
        policy_contact_state=(self._policy_contact_state if cfg.use_policy_contact_state else None),
        policy_contact_gain=cfg.policy_contact_gain, device=device,
      )

    mpc_in = baseline.MPCInput(
      x0=x0, schedule=schedule, x_ref=x_ref, u_ref=torch.zeros(B, N, 12, device=device),
      u_prev=self._u_prev, c_bar=None,
    )
    with torch.no_grad():
      mpc_out = self._mpc.solve(mpc_in)
    self._grf_ref = torch.cat([mpc_out.u_star[:, :3], mpc_out.u_star[:, 6:9]], dim=-1)
    self._u_prev = mpc_out.u_star.detach()
    x_pred = mpc_out.x_pred.detach()
    self._com_traj = x_pred[:, :, :3]
    self._linear_momentum_traj = x_pred[:, :, 3:6]
    self._vel_traj = self._linear_momentum_traj / cfg.mass
    k_pred = x_pred[:, :, 6:9]
    self._k_traj = k_pred
    self._kdot_traj = torch.zeros_like(k_pred)
    self._u_traj = mpc_out.u_pred.detach()
    self._sigma_traj = schedule.sigma[:, :, :2].detach()
    self._contact_plan_mpc.copy_(self._sigma_traj)
    if reference_contact_valid is None:
      self._contact_plan_valid.fill_(True)
    else:
      self._contact_plan_valid.copy_(reference_contact_valid)
    if N >= 2:
      self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / dt
      self._kdot_traj[:, -1] = self._kdot_traj[:, -2]
    self._traj_step = 0
    self._interpolate_traj_refs()
    self._vis_com, self._vis_ang_mom = self._com_traj, self._k_traj
    self._vis_r_lf, self._vis_r_rf = schedule.r_LF.detach(), schedule.r_RF.detach()
    self._vis_sigma_lf, self._vis_sigma_rf = schedule.sigma[:, :, 0].detach(), schedule.sigma[:, :, 1].detach()
    sigma_lf, sigma_rf = schedule.sigma[:, :, 0], schedule.sigma[:, :, 1]
    lf_td = (sigma_lf > 0.5) & (torch.cat([sigma_lf[:, :1], sigma_lf[:, :-1]], dim=1) < 0.5)
    rf_td = (sigma_rf > 0.5) & (torch.cat([sigma_rf[:, :1], sigma_rf[:, :-1]], dim=1) < 0.5)
    self._lf_landing_valid, self._rf_landing_valid = lf_td.any(dim=1), rf_td.any(dim=1)
    batch = torch.arange(B, device=device)
    self._lf_landing_target = torch.where(self._lf_landing_valid.unsqueeze(-1), schedule.r_LF[batch, lf_td.float().argmax(dim=1)], r_lf)
    self._rf_landing_target = torch.where(self._rf_landing_valid.unsqueeze(-1), schedule.r_RF[batch, rf_td.float().argmax(dim=1)], r_rf)


def mpc_com_ref(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  """Exact current-CoM error to the stored mimic MPC landmark."""
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, MimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 3, device=env.device)
  return term._com_mpc_target - term.current_centroidal_state().com_pos_w


def mpc_ang_mom_ref(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  """Exact current angular-momentum error for a mimic command observation."""
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, MimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 3, device=env.device)
  return term._k_mpc_target - term.current_centroidal_state().angular_momentum_w


class MpcExactCentroidalLandmarkTracking:
  """Reward the live articulated centroidal state for following MPC landmarks."""

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:
    self._env = env

  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str = "loco_mpc", w_com: float = 4.0,
    w_com_vel: float = 1.0, w_linear_momentum: float = 0.02, w_angular_momentum: float = 0.10,
  ) -> Tensor:
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, MimicLocoMPCCommand):
      return torch.zeros(env.num_envs, device=env.device)
    state = term.current_centroidal_state()
    loss = (
      w_com * (state.com_pos_w - term._com_mpc_target).square().sum(dim=-1)
      + w_com_vel * (state.com_vel_w - term._com_vel_mpc_target).square().sum(dim=-1)
      + w_linear_momentum * (state.linear_momentum_w - term._linear_momentum_mpc_target).square().sum(dim=-1)
      + w_angular_momentum * (state.angular_momentum_w - term._k_mpc_target).square().sum(dim=-1)
    )
    return torch.exp(-loss)


def __getattr__(name: str):
  """Expose unchanged baseline rewards/observations to mimic task configs."""
  return getattr(baseline, name)
