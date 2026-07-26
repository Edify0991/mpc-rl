"""Contact schedule utilities for multi-contact centroidal MPC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

class ContactID(IntEnum):
    """Index ordering for the four contact end-effectors."""
    LF = 0
    RF = 1
    LH = 2
    RH = 3

@dataclass
class ContactSchedule:
    """Fixed contact schedule over the MPC horizon."""

    sigma: Tensor
    r_LF: Tensor
    r_RF: Tensor
    r_LH: Tensor
    r_RH: Tensor

    R_LF: Tensor | None = None
    R_RF: Tensor | None = None
    # Diagonal x/y touchdown uncertainty.  It is metadata for robust/candidate
    # selection; the nominal QP always uses r_LF/r_RF (the Gaussian mean).
    r_LF_std_xy: Tensor | None = None
    r_RF_std_xy: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.sigma.device

    @property
    def batch_size(self) -> int:
        return self.sigma.shape[0]

    @property
    def horizon(self) -> int:
        return self.sigma.shape[1]


def make_reference_contact_schedule(
    B: int,
    N: int,
    reference_contact_state: Tensor,
    reference_r_LF: Tensor,
    reference_r_RF: Tensor,
    R_LF_rot: Tensor | None = None,
    R_RF_rot: Tensor | None = None,
    policy_contact_state: Tensor | None = None,
    policy_contact_gain: float = 0.75,
    policy_contact_horizon_decay: float = 0.5,
    policy_contact_plan_residual: Tensor | None = None,
    policy_contact_plan_residual_scale: float = 0.75,
    preserve_nominal_support: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContactSchedule:
    """Build a non-periodic schedule directly from reference-motion contact.

    ``reference_contact_state`` is the nominal future contact sequence, not a
    phase-clock reconstruction. ``policy_contact_plan_residual`` is the
    Phase-1 full-horizon correction; all of its values, contact positions,
    and activations are frozen before the QP is assembled. The legacy
    two-dimensional action remains supported only for older tasks.
    """
    reference_contact_state = reference_contact_state.to(device=device, dtype=dtype)
    reference_r_LF = reference_r_LF.to(device=device, dtype=dtype)
    reference_r_RF = reference_r_RF.to(device=device, dtype=dtype)
    if reference_contact_state.shape != (B, N, 2):
        raise ValueError(
            "reference_contact_state must have shape [B, N, 2] in [left, right] order, "
            f"got {tuple(reference_contact_state.shape)}"
        )
    for name, contact_pos in (("reference_r_LF", reference_r_LF), ("reference_r_RF", reference_r_RF)):
        if contact_pos.shape != (B, N, 3):
            raise ValueError(f"{name} must have shape [B, N, 3], got {tuple(contact_pos.shape)}")
    if not 0.0 <= policy_contact_gain <= 1.0:
        raise ValueError("policy_contact_gain must lie in [0, 1]")
    if not 0.0 <= policy_contact_horizon_decay <= 1.0:
        raise ValueError("policy_contact_horizon_decay must lie in [0, 1]")

    sigma = torch.zeros(B, N, 4, device=device, dtype=dtype)
    sigma[:, :, :2] = reference_contact_state.clamp(0.0, 1.0)
    if policy_contact_plan_residual is not None:
        policy_contact_plan_residual = policy_contact_plan_residual.to(device=device, dtype=dtype)
        if policy_contact_plan_residual.shape != (B, N, 2):
            raise ValueError(
                "policy_contact_plan_residual must have shape [B, N, 2] in [left, right] order, "
                f"got {tuple(policy_contact_plan_residual.shape)}"
            )
        if not 0.0 <= policy_contact_plan_residual_scale <= 1.0:
            raise ValueError("policy_contact_plan_residual_scale must lie in [0, 1]")
        # A full-horizon policy plan is represented as a bounded residual
        # around the non-periodic reference schedule. It is fixed for this
        # QP call, so contact constraints remain linear in the wrench.
        sigma[:, :, :2] = (
            sigma[:, :, :2]
            + policy_contact_plan_residual_scale * torch.tanh(policy_contact_plan_residual)
        ).clamp(0.0, 1.0)
    elif policy_contact_state is not None:
        policy_contact_state = policy_contact_state.to(device=device, dtype=dtype)
        if policy_contact_state.shape != (B, 2):
            raise ValueError(
                "policy_contact_state must have shape [B, 2] in [left, right] order, "
                f"got {tuple(policy_contact_state.shape)}"
            )
        horizon_weights = policy_contact_horizon_decay ** torch.arange(N, device=device, dtype=dtype)
        delta = policy_contact_gain * (2.0 * policy_contact_state - 1.0)
        sigma[:, :, :2] = (
            sigma[:, :, :2] + horizon_weights.view(1, N, 1) * delta.unsqueeze(1)
        ).clamp(0.0, 1.0)

    if preserve_nominal_support:
        # Do not let an exploratory residual erase every *nominal* support at
        # a stage. A true reference flight phase is left untouched. This is a
        # pre-QP feasibility guard, not an optimization constraint: all
        # selected values stay fixed parameters of the resulting QP.
        nominal_support = reference_contact_state.amax(dim=-1) >= 0.5
        has_support = sigma[:, :, :2].amax(dim=-1) >= 0.5
        restore = nominal_support & ~has_support
        support_index = reference_contact_state.argmax(dim=-1, keepdim=True)
        restored = sigma[:, :, :2].scatter(-1, support_index, 0.5)
        sigma[:, :, :2] = torch.where(restore.unsqueeze(-1), restored, sigma[:, :, :2])

    def _expand_rot(R: Tensor | None) -> Tensor | None:
        if R is None:
            return None
        R = R.to(device=device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        return R.contiguous()

    return ContactSchedule(
        sigma=sigma,
        r_LF=reference_r_LF,
        r_RF=reference_r_RF,
        r_LH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        r_RH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        R_LF=_expand_rot(R_LF_rot),
        R_RF=_expand_rot(R_RF_rot),
    )

def make_double_support_schedule(
    B: int,
    N: int,
    r_LF: Tensor,
    r_RF: Tensor,
    r_LH: Tensor | None = None,
    r_RH: Tensor | None = None,
    R_LF_rot: Tensor | None = None,
    R_RF_rot: Tensor | None = None,
    hands_active: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContactSchedule:
    """Create a simple double-support (both feet) schedule."""
    sigma = torch.zeros(B, N, 4, device=device, dtype=dtype)
    sigma[:, :, ContactID.LF] = 1.0
    sigma[:, :, ContactID.RF] = 1.0
    if hands_active:
        sigma[:, :, ContactID.LH] = 1.0
        sigma[:, :, ContactID.RH] = 1.0

    def _expand(t: Tensor | None, default_val: float = 0.0) -> Tensor:
        if t is None:
            return torch.full((B, N, 3), default_val, device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)
        return t.unsqueeze(1).expand(B, N, 3)

    def _expand_rot(R: Tensor | None) -> Tensor | None:
        if R is None:
            return None
        R = R.to(device=device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        return R.contiguous()

    return ContactSchedule(
        sigma=sigma,
        r_LF=_expand(r_LF),
        r_RF=_expand(r_RF),
        r_LH=_expand(r_LH),
        r_RH=_expand(r_RH),
        R_LF=_expand_rot(R_LF_rot),
        R_RF=_expand_rot(R_RF_rot),
    )

def make_walking_schedule(
    B: int,
    N: int,
    r_LF: Tensor,
    r_RF: Tensor,
    gait_phase: Tensor,
    period: float = 0.7,
    dt: float = 0.05,
    duty_factor: float = 0.5,
    com_pos: "Tensor | None" = None,
    v_cmd: "Tensor | None" = None,
    yaw: "Tensor | None" = None,
    yaw_rate: "Tensor | None" = None,
    hip_width: float = 0.1,
    R_LF_rot: "Tensor | None" = None,
    R_RF_rot: "Tensor | None" = None,
    phase_rate_scale: "Tensor | None" = None,
    duty_factor_offset: "Tensor | None" = None,
    touchdown_residual_LF: "Tensor | None" = None,
    touchdown_residual_RF: "Tensor | None" = None,
    touchdown_std_LF_xy: "Tensor | None" = None,
    touchdown_std_RF_xy: "Tensor | None" = None,
    reference_r_LF: "Tensor | None" = None,
    reference_r_RF: "Tensor | None" = None,
    policy_contact_state: "Tensor | None" = None,
    policy_contact_gain: float = 0.75,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContactSchedule:
    """Create a walking schedule with predictive Raibert-style foot placement."""
    import math

    r_LF = r_LF.to(device=device, dtype=dtype)
    r_RF = r_RF.to(device=device, dtype=dtype)

    if phase_rate_scale is None:
        phase_rate_scale = torch.ones(B, device=device, dtype=dtype)
    else:
        phase_rate_scale = phase_rate_scale.to(device=device, dtype=dtype).reshape(B)
    if duty_factor_offset is None:
        duty_factor_offset = torch.zeros(B, device=device, dtype=dtype)
    else:
        duty_factor_offset = duty_factor_offset.to(device=device, dtype=dtype).reshape(B)
    phase_rate = (2.0 * math.pi / period) * phase_rate_scale
    k_steps = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    phase_traj = gait_phase.unsqueeze(1) + phase_rate.unsqueeze(1) * dt * k_steps

    duty = (duty_factor + duty_factor_offset).clamp(0.35, 0.75)
    threshold = torch.cos(torch.pi * duty).unsqueeze(1)
    rf_stance = phase_traj.sin() < threshold
    lf_stance = (phase_traj + math.pi).sin() < threshold

    sigma = torch.zeros(B, N, 4, device=device, dtype=dtype)
    sigma[:, :, ContactID.LF] = lf_stance.to(dtype)
    sigma[:, :, ContactID.RF] = rf_stance.to(dtype)
    if policy_contact_state is not None:
        policy_contact_state = policy_contact_state.to(device=device, dtype=dtype)
        if policy_contact_state.shape != (B, 2):
            raise ValueError(
                "policy_contact_state must have shape [B, 2] in [left, right] order, "
                f"got {tuple(policy_contact_state.shape)}"
            )
        if not 0.0 <= policy_contact_gain <= 1.0:
            raise ValueError("policy_contact_gain must lie in [0, 1]")
        # Raw policy zero maps to w=0.5, which leaves the nominal phase
        # schedule unchanged.  w<0.5 relaxes a nominal support and w>0.5 can
        # activate an early support.  sigma is frozen before the QP, so the
        # MPC remains conditionally convex despite learned contact timing.
        delta = policy_contact_gain * (2.0 * policy_contact_state.unsqueeze(1) - 1.0)
        sigma[:, :, :2] = (sigma[:, :, :2] + delta).clamp(0.0, 1.0)

    if com_pos is not None and v_cmd is not None:
        v_cmd = v_cmd.to(device=device, dtype=dtype)
        com_pos = com_pos.to(device=device, dtype=dtype)

        stride_time = period / (2.0 * phase_rate_scale)

        wz_t = (yaw_rate.to(device=device, dtype=dtype)
                if yaw_rate is not None
                else torch.zeros(B, device=device, dtype=dtype))
        yaw_t = (yaw.to(device=device, dtype=dtype)
                 if yaw is not None
                 else torch.zeros(B, device=device, dtype=dtype))

        cos_y0 = yaw_t.cos(); sin_y0 = yaw_t.sin()
        vx_body = ( cos_y0 * v_cmd[:, 0] + sin_y0 * v_cmd[:, 1])
        vy_body = (-sin_y0 * v_cmd[:, 0] + cos_y0 * v_cmd[:, 1])

        k_idx   = torch.arange(N, device=device, dtype=dtype)
        yaw_k   = yaw_t.unsqueeze(1) + wz_t.unsqueeze(1) * k_idx * dt
        cos_k   = yaw_k.cos()
        sin_k   = yaw_k.sin()

        vx_k = (cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1))
        vy_k = (sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1))

        com_arc = torch.zeros(B, N, 3, device=device, dtype=dtype)
        com_arc[:, :, 0] = com_pos[:, 0:1] + torch.cumsum(vx_k * dt, dim=1)
        com_arc[:, :, 1] = com_pos[:, 1:2] + torch.cumsum(vy_k * dt, dim=1)
        com_arc[:, :, 2] = com_pos[:, 2:3].expand(B, N)

        def _foot_traj(
            r_f0: Tensor,
            stance_mask: Tensor,
            sign_y: float,
            touchdown_residual: "Tensor | None",
            reference_touchdown: "Tensor | None",
        ) -> Tensor:
            """Propagate foot position; apply Raibert heuristic at touchdowns."""
            traj = torch.zeros(B, N, 3, device=device, dtype=dtype)
            r_cur = r_f0.clone()
            r_f0_z = r_f0[:, 2].clamp(min=0.0)

            for k in range(N):
                is_stance = stance_mask[:, k]
                new_stance = (
                    is_stance & (~stance_mask[:, k - 1])
                    if k > 0
                    else torch.zeros(B, device=device, dtype=torch.bool)
                )

                if new_stance.any():
                    hip_x = sign_y * (-sin_k[:, k]) * hip_width
                    hip_y = sign_y * ( cos_k[:, k]) * hip_width

                    p_new_x = (com_arc[:, k, 0]
                                + vx_k[:, k] * 0.5 * stride_time
                                + hip_x)
                    p_new_y = (com_arc[:, k, 1]
                                + vy_k[:, k] * 0.5 * stride_time
                                + hip_y)
                    p_new = torch.stack([p_new_x, p_new_y, r_f0_z], dim=-1)
                    if reference_touchdown is not None:
                        reference_touchdown = reference_touchdown.to(device=device, dtype=dtype)
                        if reference_touchdown.shape != (B, N, 3):
                            raise ValueError(
                                "reference touchdown trajectory must have shape [B, N, 3], "
                                f"got {tuple(reference_touchdown.shape)}"
                            )
                        # The motion reference is the Gaussian mean; the
                        # learned residual is applied only when a new stance
                        # begins, never to an already-loaded support foot.
                        p_new = reference_touchdown[:, k]
                    if touchdown_residual is not None:
                        residual = touchdown_residual.to(device=device, dtype=dtype)
                        if residual.shape != (B, 3):
                            raise ValueError(
                                "touchdown residual must have shape [B, 3], "
                                f"got {tuple(residual.shape)}"
                            )
                        # z is kept at the terrain/nominal contact height.  A
                        # terrain model should supply height corrections.
                        p_new = p_new + torch.cat([
                            residual[:, :2], torch.zeros(B, 1, device=device, dtype=dtype)
                        ], dim=-1)
                    r_cur = torch.where(new_stance.unsqueeze(-1), p_new, r_cur)

                traj[:, k, :] = r_cur

            return traj

        r_LF_traj = _foot_traj(r_LF, lf_stance, sign_y=+1.0, touchdown_residual=touchdown_residual_LF, reference_touchdown=reference_r_LF)
        r_RF_traj = _foot_traj(r_RF, rf_stance, sign_y=-1.0, touchdown_residual=touchdown_residual_RF, reference_touchdown=reference_r_RF)
    else:
        r_LF_traj = r_LF.unsqueeze(1).expand(B, N, 3).contiguous()
        r_RF_traj = r_RF.unsqueeze(1).expand(B, N, 3).contiguous()

    def _expand_rot(R: "Tensor | None") -> "Tensor | None":
        if R is None:
            return None
        R = R.to(device=device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        return R.contiguous()

    return ContactSchedule(
        sigma=sigma,
        r_LF=r_LF_traj,
        r_RF=r_RF_traj,
        r_LH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        r_RH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        R_LF=_expand_rot(R_LF_rot),
        R_RF=_expand_rot(R_RF_rot),
        r_LF_std_xy=(None if touchdown_std_LF_xy is None else touchdown_std_LF_xy.to(device=device, dtype=dtype)),
        r_RF_std_xy=(None if touchdown_std_RF_xy is None else touchdown_std_RF_xy.to(device=device, dtype=dtype)),
    )
