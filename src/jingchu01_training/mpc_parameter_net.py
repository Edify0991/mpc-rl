"""Bounded neural parameterization for contact-scheduled centroidal MPC.

The network is deliberately an *MPC parameter adaptor*, not another direct
joint-control policy.  It outputs a small, bounded set of quantities that are
frozen before each QP solve: contact timing, touchdown residual statistics,
and a centroidal-momentum reference residual.  Consequently the centroidal
MPC remains a convex QP conditional on its network output.

Sampling touchdown locations is provided for offline candidate selection or
Monte-Carlo evaluation.  Do not sample independently inside every QP solve:
that creates a noisy control target and gives no chance-constraint guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MPCParameterBounds:
    """Physical output limits for :class:`MPCParameterNet`."""

    phase_rate_min: float = 0.80
    phase_rate_max: float = 1.20
    duty_offset_max: float = 0.10
    touchdown_xy_max: float = 0.18
    touchdown_std_min: float = 0.005
    touchdown_std_max: float = 0.100
    linear_momentum_residual_max: float = 12.0
    angular_momentum_residual_max: float = 4.0


@dataclass(frozen=True)
class MPCParameters:
    """Decoded, bounded MPC parameters for a batch of environments.

    ``phase_rate_scale`` changes the contact-clock rate while the numerical
    CD-MPC integration period remains fixed.  This is intentional: changing
    the solver ``dt`` per environment would require rebuilding all dynamics
    matrices and introduces timing/force/foothold multilinearities.
    """

    phase_rate_scale: Tensor              # [B]
    duty_factor_offset: Tensor             # [B]
    touchdown_mean_residual: Tensor        # [B, 2, 3], world-frame; z = 0
    touchdown_std_xy: Tensor               # [B, 2, 2]
    momentum_residual: Tensor              # [B, 6], [linear, angular]


class MPCParameterNet(nn.Module):
    """GRU adaptor emitting raw parameters from a short state/reference history.

    Input features are project-specific and intentionally supplied as a single
    tensor ``[batch, history, input_dim]``.  The runtime command currently
    uses 29 features: current/reference centroidal state, their error, and
    binary foot-contact observations.  Keeping this interface compact makes
    offline imitation targets and TorchScript export unambiguous.
    """

    raw_dim: int = 16

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.raw_dim)
        )
        # A newly-created adaptor must be exactly nominal until it is trained.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history: Tensor) -> Tensor:
        """Return raw parameters ``[B, 16]`` suitable for :func:`decode_parameters`."""
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected history [B, H, {self.input_dim}]"
            )
        _, hidden = self.gru(history)
        return self.head(hidden[-1])


def nominal_parameters(batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> MPCParameters:
    """Return the no-correction parameter set used without a checkpoint."""
    z = torch.zeros(batch_size, device=device, dtype=dtype)
    return MPCParameters(
        phase_rate_scale=torch.ones_like(z),
        duty_factor_offset=z,
        touchdown_mean_residual=torch.zeros(batch_size, 2, 3, device=device, dtype=dtype),
        touchdown_std_xy=torch.zeros(batch_size, 2, 2, device=device, dtype=dtype),
        momentum_residual=torch.zeros(batch_size, 6, device=device, dtype=dtype),
    )


def decode_parameters(raw: Tensor, bounds: MPCParameterBounds = MPCParameterBounds()) -> MPCParameters:
    """Map unconstrained network outputs to physically bounded MPC parameters."""
    if raw.ndim != 2 or raw.shape[-1] != MPCParameterNet.raw_dim:
        raise ValueError(f"Expected raw MPC parameters [B, 16], got {tuple(raw.shape)}")
    B = raw.shape[0]
    phase_rate = bounds.phase_rate_min + (bounds.phase_rate_max - bounds.phase_rate_min) * torch.sigmoid(raw[:, 0])
    duty_offset = bounds.duty_offset_max * torch.tanh(raw[:, 1])
    xy = bounds.touchdown_xy_max * torch.tanh(raw[:, 2:6]).view(B, 2, 2)
    touchdown_mean = torch.zeros(B, 2, 3, device=raw.device, dtype=raw.dtype)
    touchdown_mean[:, :, :2] = xy
    std_span = bounds.touchdown_std_max - bounds.touchdown_std_min
    touchdown_std = bounds.touchdown_std_min + std_span * torch.sigmoid(raw[:, 6:10]).view(B, 2, 2)
    momentum = torch.cat([
        bounds.linear_momentum_residual_max * torch.tanh(raw[:, 10:13]),
        bounds.angular_momentum_residual_max * torch.tanh(raw[:, 13:16]),
    ], dim=-1)
    return MPCParameters(phase_rate, duty_offset, touchdown_mean, touchdown_std, momentum)


def sample_touchdown_candidates(
    parameters: MPCParameters,
    num_candidates: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``[B, K, 2, 3]`` touchdown residual candidates.

    Samples are mirrored in pairs so their batch mean remains close to the
    learned Gaussian mean.  They are intended for low-rate candidate scoring,
    not for injecting independent noise into the MPC at every control step.
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    mean = parameters.touchdown_mean_residual
    std = parameters.touchdown_std_xy
    B = mean.shape[0]
    half = (num_candidates + 1) // 2
    eps = torch.randn(B, half, 2, 2, device=mean.device, dtype=mean.dtype, generator=generator)
    eps = torch.cat([eps, -eps], dim=1)[:, :num_candidates]
    samples = mean.unsqueeze(1).expand(-1, num_candidates, -1, -1).clone()
    samples[..., :2] += eps * std.unsqueeze(1)
    return samples
