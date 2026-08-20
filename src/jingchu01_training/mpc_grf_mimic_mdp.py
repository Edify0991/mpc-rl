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

# The paper-compatible velocity command stays untouched.  This module adds
# only motion-derived centroidal references and their fixed contact schedule.
from . import mpc_grf_mdp as baseline
from jingchu01_mpc.contact_schedule import make_reference_contact_schedule
from training_common.reference_centroidal import (
  CentroidalState,
  ReferenceCentroidalTrajectory,
  compute_centroidal_state,
  compute_reference_centroidal,
  prealign_reference_kinematics_to_initial_anchor,
)
from .mimic_mdp import MotionReferenceCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


@dataclass(kw_only=True)
class MimicLocoMPCCommandCfg(baseline.LocoMPCCommandCfg):
  """Configuration for fixed-plan, reference-centroidal mimic MPC."""

  # Inherited for dataclass/config compatibility with LocoMPCCommandCfg, but
  # deliberately unused here: reference motion, rather than a 3-D twist,
  # owns every MPC target in this command.
  vel_cmd_name: str | None = None
  motion_command_name: str | None = None
  # These fields are MPC data, not local joint-tracker data.  They define the
  # model-matched bodies used to reconstruct c, l, k and the nominal contacts.
  centroidal_body_names: tuple[str, ...] = ()
  prealign_centroidal_to_initial_anchor: bool = False
  contact_body_names: tuple[str, ...] = ()
  contact_point_offsets_b: dict[str, tuple[float, float, float]] | None = None
  reference_contact_key: str | None = None
  # ``reference_contact_state`` may deliberately originate from the pre-
  # retargeting human BVH.  It is a fixed nominal contact *schedule*, while
  # c/l/k reference quantities remain reconstructed from the robot clip.
  reference_contact_height_threshold: float = 0.03
  reference_contact_speed_threshold: float = 0.35
  reference_contact_min_stance_frames: int = 3
  reference_contact_min_swing_frames: int = 2

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

  This is deliberately parameter-free: it consumes only the retargeted
  reference centroidal trajectory and its fixed contact plan.  Learned MPC
  parameters and policy contact-plan residuals live in
  :mod:`jingchu01_training.mpc_grf_parameterized_mdp`.
  """

  cfg: MimicLocoMPCCommandCfg

  def __init__(self, cfg: MimicLocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
    if cfg.vel_cmd_name is not None:
      raise ValueError(
        "MimicLocoMPCCommand must not consume a velocity command; "
        "set vel_cmd_name=None and provide motion_command_name."
      )
    super().__init__(cfg, env)
    body_ids = self._robot.indexing.body_ids.detach().cpu().numpy()
    if len(body_ids) != self._robot.data.body_link_pos_w.shape[1]:
      raise RuntimeError("Articulation body indexing and MJLab body kinematics disagree")
    self._online_body_mass = torch.empty(env.num_envs, len(body_ids), device=env.device)
    self._online_body_com_offset_b = torch.empty(env.num_envs, len(body_ids), 3, device=env.device)
    self._online_body_inertia_diag = torch.empty(env.num_envs, len(body_ids), 3, device=env.device)
    self._online_body_inertial_quat_b = torch.empty(env.num_envs, len(body_ids), 4, device=env.device)
    self._refresh_online_centroidal_model(torch.arange(env.num_envs, device=env.device))
    self._linear_momentum_traj = torch.zeros_like(self._vel_traj)
    B, N = env.num_envs, cfg.mpc_horizon
    self._u_traj = torch.zeros(B, N, 12, device=env.device)
    self._sigma_traj = torch.zeros(B, N, 2, device=env.device)
    self._u_mpc_target = torch.zeros(B, 12, device=env.device)
    self._force_mpc_target = torch.zeros(B, 2, 3, device=env.device)
    self._moment_mpc_target = torch.zeros(B, 2, 3, device=env.device)
    self._contact_mpc_target = torch.zeros(B, 2, device=env.device)
    self._contact_plan_mpc = torch.zeros(B, N, 2, device=env.device)
    self._contact_plan_valid = torch.ones(B, N, dtype=torch.bool, device=env.device)
    self._reference_motion: MotionReferenceCommand | None = None
    self._reference_centroidal: ReferenceCentroidalTrajectory | None = None
    self._reference_contact_state: Tensor | None = None
    # Reward and critic observations consume the same articulated state.  The
    # command manager invalidates this once per environment step, avoiding
    # duplicate O(num_bodies) centroidal reductions without carrying a stale
    # state across physics steps.
    self._centroidal_state_cache: CentroidalState | None = None

  def current_centroidal_state(self) -> CentroidalState:
    """Read the exact whole-body centroidal state in simulator-world axes."""
    if self._centroidal_state_cache is not None:
      return self._centroidal_state_cache
    data = self._robot.data
    dtype = data.body_link_pos_w.dtype
    self._centroidal_state_cache = compute_centroidal_state(
      body_link_pos_w=data.body_link_pos_w,
      body_link_quat_w=data.body_link_quat_w,
      body_com_lin_vel_w=data.body_com_lin_vel_w,
      body_link_ang_vel_w=data.body_link_ang_vel_w,
      body_mass=self._online_body_mass.to(dtype=dtype),
      body_com_offset_b=self._online_body_com_offset_b.to(dtype=dtype),
      body_inertia_diag=self._online_body_inertia_diag.to(dtype=dtype),
      body_inertial_quat_b=self._online_body_inertial_quat_b.to(dtype=dtype),
    )
    return self._centroidal_state_cache

  def _refresh_online_centroidal_model(self, env_ids: Tensor) -> None:
    """Refresh the articulated centroidal reduction from simulator DR arrays."""
    ids = self._robot_body_model_ids
    for target, name in (
      (self._online_body_mass, "body_mass"),
      (self._online_body_com_offset_b, "body_ipos"),
      (self._online_body_inertia_diag, "body_inertia"),
      (self._online_body_inertial_quat_b, "body_iquat"),
    ):
      value = self._model_tensor(name)
      if value is not None:
        target[env_ids] = value[env_ids][:, ids]

  def _ensure_reference_data(self, motion: MotionReferenceCommand) -> None:
    """Build immutable c/l/k/contact clip data once, under the MPC owner."""
    if self._reference_motion is motion and self._reference_centroidal is not None:
      return
    names = self.cfg.centroidal_body_names or tuple(motion._motion_body_names)
    local_ids, pos, quat, lin_vel, ang_vel, resolved_names = motion.reference_body_kinematics(names, "centroidal")
    if not resolved_names:
      raise ValueError("Mimic MPC needs at least one centroidal body")
    global_ids = motion.robot.indexing.body_ids[local_ids].detach().cpu().numpy()
    model = self._env.sim.mj_model
    dtype = pos.dtype

    def model_field(name: str) -> Tensor:
      return torch.as_tensor(np.asarray(getattr(model, name))[global_ids], device=self.device, dtype=dtype)

    name_to_index = {name: i for i, name in enumerate(resolved_names)}
    missing_contacts = [name for name in self.cfg.contact_body_names if name not in name_to_index]
    if missing_contacts:
      raise ValueError(
        "Every MPC contact_body_name must be included in centroidal_body_names: "
        f"{missing_contacts}"
      )
    contact_indices = torch.tensor(
      [name_to_index[name] for name in self.cfg.contact_body_names], device=self.device, dtype=torch.long,
    )
    offsets_cfg = self.cfg.contact_point_offsets_b or {}
    offsets = torch.tensor(
      [offsets_cfg.get(name, (0.0, 0.0, 0.0)) for name in self.cfg.contact_body_names],
      device=self.device, dtype=dtype,
    ) if self.cfg.contact_body_names else torch.empty(0, 3, device=self.device, dtype=dtype)
    if self.cfg.prealign_centroidal_to_initial_anchor and motion.cfg.reference_frame_alignment == "none":
      try:
        anchor_index = list(resolved_names).index(motion.cfg.anchor_body_name)
      except ValueError as exc:
        raise ValueError("centroidal prealignment requires the anchor in centroidal_body_names") from exc
      pos, quat, lin_vel, ang_vel = prealign_reference_kinematics_to_initial_anchor(
        body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lin_vel,
        body_ang_vel_w=ang_vel, anchor_body_index=anchor_index,
      )
    reference = compute_reference_centroidal(
      body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lin_vel, body_ang_vel_w=ang_vel,
      body_mass=model_field("body_mass"), body_com_offset_b=model_field("body_ipos"),
      body_inertia_diag=model_field("body_inertia"), body_inertial_quat_b=model_field("body_iquat"),
      contact_body_indices=contact_indices, contact_point_offset_b=offsets,
      body_linear_velocity_point=motion.motion.clip.body_linear_velocity_point,
    )
    self._reference_motion = motion
    self._reference_centroidal = reference
    self._reference_contact_state = self._load_or_infer_reference_contact_state(motion, reference)

  def _load_or_infer_reference_contact_state(
    self, motion: MotionReferenceCommand, reference: ReferenceCentroidalTrajectory,
  ) -> Tensor:
    contacts = reference.contact_pos_w
    n_contacts = contacts.shape[1]
    if self.cfg.reference_contact_key is not None:
      labels = motion.motion.optional_tensor(self.cfg.reference_contact_key)
      if labels is None:
        raise ValueError(f"reference_contact_key={self.cfg.reference_contact_key!r} is absent from {motion.motion.path}")
      if labels.shape != (motion.num_frames, n_contacts):
        raise ValueError(f"Reference contact labels must be [{motion.num_frames}, {n_contacts}], got {tuple(labels.shape)}")
      return labels.clamp(0.0, 1.0)
    if n_contacts == 0:
      return torch.empty(motion.num_frames, 0, device=self.device)
    height = contacts[..., 2]
    floor_height = height.amin(dim=0, keepdim=True)
    prev, next_ = torch.cat([contacts[:1], contacts[:-1]], 0), torch.cat([contacts[1:], contacts[-1:]], 0)
    speed = (next_ - prev).norm(dim=-1) * (0.5 * motion.fps)
    candidates = (height - floor_height <= self.cfg.reference_contact_height_threshold) & (speed <= self.cfg.reference_contact_speed_threshold)
    return self._smooth_reference_contact_state(candidates).to(torch.float32)

  def _smooth_reference_contact_state(self, candidates: Tensor) -> Tensor:
    min_stance, min_swing = self.cfg.reference_contact_min_stance_frames, self.cfg.reference_contact_min_swing_frames
    if min_stance < 1 or min_swing < 1:
      raise ValueError("reference contact minimum stance/swing duration must be at least one frame")
    state = candidates.clone()
    for contact_id in range(state.shape[1]):
      for target, maximum, replacement in ((False, min_swing - 1, True), (True, min_stance - 1, False)):
        start = 0
        while start < state.shape[0]:
          end = start + 1
          while end < state.shape[0] and bool(state[end, contact_id].item()) == bool(state[start, contact_id].item()):
            end += 1
          if bool(state[start, contact_id].item()) == target and end - start <= maximum:
            if start > 0 and end < state.shape[0] and bool(state[start - 1, contact_id].item()) == replacement and bool(state[end, contact_id].item()) == replacement:
              state[start:end, contact_id] = replacement
          start = end
    return state

  def _reference_horizon(
    self, motion: MotionReferenceCommand, steps: int, dt: float,
  ) -> tuple[ReferenceCentroidalTrajectory, Tensor, Tensor]:
    self._ensure_reference_data(motion)
    assert self._reference_centroidal is not None and self._reference_contact_state is not None
    s = motion.horizon_frame_progress(steps, dt)
    reference = self._reference_centroidal
    com_pos = motion.sample_reference_tensor(reference.com_pos_w, s)
    contact_pos = motion.sample_reference_tensor(reference.contact_pos_w, s)
    valid = motion.horizon_valid(steps, dt)
    zeros = torch.zeros_like(com_pos)
    com_vel = torch.where(valid.unsqueeze(-1), motion.sample_reference_tensor(reference.com_vel_w, s), zeros)
    linear = torch.where(valid.unsqueeze(-1), motion.sample_reference_tensor(reference.linear_momentum_w, s), zeros)
    angular = torch.where(valid.unsqueeze(-1), motion.sample_reference_tensor(reference.angular_momentum_w, s), zeros)
    origins = self._env.scene.env_origins[:, None, :]
    com_pos, contact_pos = com_pos + origins, contact_pos + origins.unsqueeze(-2)
    contact_state = motion.sample_reference_tensor(self._reference_contact_state, s)
    return ReferenceCentroidalTrajectory(
      com_pos_w=com_pos, com_vel_w=com_vel, linear_momentum_w=linear, angular_momentum_w=angular,
      contact_pos_w=contact_pos, contact_pos_rel_com_w=contact_pos - com_pos.unsqueeze(-2),
    ), contact_state, valid

  def reference_centroidal_observation(self) -> Tensor:
    if self.cfg.motion_command_name is None:
      return torch.zeros(self.num_envs, 18, device=self.device)
    motion = self._env.command_manager.get_term(self.cfg.motion_command_name)
    if not isinstance(motion, MotionReferenceCommand):
      raise TypeError("Mimic MPC motion command must be MotionReferenceCommand")
    reference, _, _ = self._reference_horizon(motion, 1, 0.0)
    return torch.cat((
      reference.com_pos_w[:, 0], reference.com_vel_w[:, 0], reference.linear_momentum_w[:, 0],
      reference.angular_momentum_w[:, 0], reference.contact_pos_rel_com_w[:, 0].flatten(1),
    ), dim=-1)

  def _interpolate_traj_refs(self) -> None:
    super()._interpolate_traj_refs()
    N = self.cfg.mpc_horizon
    t_frac = self.cfg.tracking_lookahead_frac * (N - 1) + self._traj_step * self._env.step_dt / self.cfg.mpc_dt
    idx = min(max(int(t_frac), 0), N - 2)
    alpha = min(max(t_frac - idx, 0.0), 1.0)
    self._u_mpc_target = (
      (1.0 - alpha) * self._u_traj[:, idx] + alpha * self._u_traj[:, idx + 1]
    )
    self._force_mpc_target = torch.stack(
      [self._u_mpc_target[:, 0:3], self._u_mpc_target[:, 6:9]], dim=1
    )
    self._moment_mpc_target = torch.stack(
      [self._u_mpc_target[:, 3:6], self._u_mpc_target[:, 9:12]], dim=1
    )
    self._contact_mpc_target = self._sigma_traj[:, idx]

  def _resample_command(self, env_ids: Tensor) -> None:
    super()._resample_command(env_ids)
    self._refresh_online_centroidal_model(env_ids)
    self._centroidal_state_cache = None
    self._linear_momentum_traj[env_ids] = 0.0
    self._u_traj[env_ids] = 0.0
    self._sigma_traj[env_ids] = 0.0
    self._u_mpc_target[env_ids] = 0.0
    self._force_mpc_target[env_ids] = 0.0
    self._moment_mpc_target[env_ids] = 0.0
    self._contact_mpc_target[env_ids] = 0.0
    self._contact_plan_mpc[env_ids] = 0.0
    self._contact_plan_valid[env_ids] = False

  def _parameterize_reference(self, x0: Tensor, x_ref: Tensor) -> Tensor:
    """Pure mimic hook: preserve the reference state exactly."""
    return x_ref

  def _make_reference_schedule(
    self, *, B: int, N: int, reference_contact_state: Tensor,
    reference_contacts: Tensor, R_lf: Tensor, R_rf: Tensor,
  ):
    return make_reference_contact_schedule(
      B=B, N=N, reference_contact_state=reference_contact_state,
      reference_r_LF=reference_contacts[:, :, 0],
      reference_r_RF=reference_contacts[:, :, 1],
      R_LF_rot=R_lf, R_RF_rot=R_rf, device=self.device,
    )

  def _update_command(self) -> None:
    """Solve the same QP as the baseline using exact mimic centroidal data."""
    self._centroidal_state_cache = None
    self._step_count += 1
    self._traj_step += 1
    self._interpolate_traj_refs()
    if self._step_count % self.cfg.run_every_n_steps != 0:
      return

    cfg, B, N, dt, device, robot = (
      self.cfg, self.num_envs, self.cfg.mpc_horizon, self.cfg.mpc_dt, self.device, self._robot,
    )
    self._sync_mpc_model_parameters(torch.arange(B, device=device))
    self._refresh_online_centroidal_model(torch.arange(B, device=device))
    online = self.current_centroidal_state()
    c, linear_momentum, k = online.com_pos_w, online.linear_momentum_w, online.angular_momentum_w
    x0 = torch.cat([c, linear_momentum, k], dim=-1)

    site_pos = robot.data.site_pos_w[:, self._site_ids, :]
    r_lf = site_pos[:, self._lf_site_local_idx, :]
    r_rf = site_pos[:, self._rf_site_local_idx, :]
    site_quat = robot.data.site_quat_w[:, self._site_ids, :]
    R_lf = baseline._quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
    R_rf = baseline._quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

    if cfg.motion_command_name is None:
      raise ValueError("MimicLocoMPCCommand requires motion_command_name")
    motion = self._env.command_manager.get_term(cfg.motion_command_name)
    if not isinstance(motion, MotionReferenceCommand):
      raise ValueError(f"motion_command_name={cfg.motion_command_name!r} must name a MotionReferenceCommand")
    # No velocity-command approximation is constructed in the Mimic path.
    # The QP target at every node is the retargeted clip's model-consistent
    # [c_ref, l_ref, k_ref], sampled at s_i=s(t)+i*f_ref*dt.
    reference, contact_horizon, valid_horizon = self._reference_horizon(motion, N + 1, dt)
    x_ref = torch.empty(B, N + 1, 9, device=device, dtype=x0.dtype)
    x_ref, _, reference_contacts = apply_mimic_centroidal_reference_to_mpc_target(x_ref, reference)
    reference_contact_state = contact_horizon[:, :-1]
    reference_contact_valid = valid_horizon[:, :-1]
    if reference_contacts.shape[2] != 2 or reference_contact_state.shape[2] != 2:
      raise ValueError("The bipedal centroidal MPC requires exactly [left, right] reference contacts")
    x_ref = self._parameterize_reference(x0, x_ref)
    schedule = self._make_reference_schedule(
      B=B, N=N, reference_contact_state=reference_contact_state,
      reference_contacts=reference_contacts, R_lf=R_lf, R_rf=R_rf,
    )

    mpc_in = baseline.MPCInput(
      x0=x0, schedule=schedule, x_ref=x_ref, u_ref=torch.zeros(B, N, 12, device=device),
      u_prev=self._u_prev, c_bar=None, model_parameters=self._mpc_model_parameters,
    )
    with torch.no_grad():
      mpc_out = self._mpc.solve(mpc_in)
    self._grf_ref = torch.cat([mpc_out.u_star[:, :3], mpc_out.u_star[:, 6:9]], dim=-1)
    self._u_prev = mpc_out.u_star.detach()
    x_pred = mpc_out.x_pred.detach()
    self._com_traj = x_pred[:, :, :3]
    self._linear_momentum_traj = x_pred[:, :, 3:6]
    self._vel_traj = self._linear_momentum_traj / self._mpc_model_parameters.mass[:, None, None]
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


def mpc_reference_centroidal(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  """MPC-owned reference ``[c, dc, l, k, r_contact-c]`` for the critic."""
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, MimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 18, device=env.device)
  return term.reference_centroidal_observation()


class MpcExactCentroidalLandmarkTracking:
  """Reward the live articulated centroidal state for following MPC landmarks."""

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:
    self._env = env

  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str = "loco_mpc", w_com: float = 4.0,
    w_com_vel: float = 1.0, w_angular_momentum: float = 0.10,
  ) -> Tensor:
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, MimicLocoMPCCommand):
      return torch.zeros(env.num_envs, device=env.device)
    state = term.current_centroidal_state()
    loss = (
      w_com * (state.com_pos_w - term._com_mpc_target).square().sum(dim=-1)
      + w_com_vel * (state.com_vel_w - term._com_vel_mpc_target).square().sum(dim=-1)
      + w_angular_momentum * (state.angular_momentum_w - term._k_mpc_target).square().sum(dim=-1)
    )
    return torch.exp(-loss)


def mpc_contact_force_ref(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, MimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 6, device=env.device)
  return term._force_mpc_target.reshape(env.num_envs, 6)


def mpc_contact_moment_ref(env: "ManagerBasedRlEnv", command_name: str = "loco_mpc") -> Tensor:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, MimicLocoMPCCommand):
    return torch.zeros(env.num_envs, 6, device=env.device)
  return term._moment_mpc_target.reshape(env.num_envs, 6)


def __getattr__(name: str):
  """Expose unchanged baseline rewards/observations to mimic task configs."""
  return getattr(baseline, name)
