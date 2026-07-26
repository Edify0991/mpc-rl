"""Mass-consistent centroidal quantities for a kinematic reference motion.

Reference NPZ files provide link-frame pose/velocity trajectories, not CoM or
momentum.  This module reconstructs those quantities using the matched MuJoCo
model's mass, inertial offset, inertia and inertial-frame orientation.  Contact
points are defined on named bodies and remain available independently of the
policy's contact intention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_mul


@dataclass(frozen=True)
class ReferenceCentroidalTrajectory:
  """Per-reference-frame quantities in world coordinates."""

  com_pos_w: torch.Tensor              # [T, 3]
  com_vel_w: torch.Tensor              # [T, 3]
  linear_momentum_w: torch.Tensor      # [T, 3]
  angular_momentum_w: torch.Tensor     # [T, 3], about the system CoM
  contact_pos_w: torch.Tensor          # [T, C, 3]
  contact_pos_rel_com_w: torch.Tensor  # [T, C, 3]

  def select(self, frame_ids: torch.Tensor) -> "ReferenceCentroidalTrajectory":
    """Gather a batch/horizon of frame ids while preserving its leading shape."""
    return ReferenceCentroidalTrajectory(
      com_pos_w=self.com_pos_w[frame_ids],
      com_vel_w=self.com_vel_w[frame_ids],
      linear_momentum_w=self.linear_momentum_w[frame_ids],
      angular_momentum_w=self.angular_momentum_w[frame_ids],
      contact_pos_w=self.contact_pos_w[frame_ids],
      contact_pos_rel_com_w=self.contact_pos_rel_com_w[frame_ids],
    )


def _rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
  """Quaternion [w, x, y, z] to rotation matrix for arbitrary leading dims."""
  w, x, y, z = quat.unbind(dim=-1)
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ), dim=-1).reshape(*quat.shape[:-1], 3, 3)


def compute_reference_centroidal(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  body_mass: torch.Tensor,
  body_com_offset_b: torch.Tensor,
  body_inertia_diag: torch.Tensor,
  body_inertial_quat_b: torch.Tensor,
  contact_body_indices: torch.Tensor | None = None,
  contact_point_offset_b: torch.Tensor | None = None,
) -> ReferenceCentroidalTrajectory:
  """Compute CoM, momentum and contact vectors for every reference frame.

  Args:
    body_*: named-reference body trajectories with shape ``[T, B, ...]``.
    body_com_offset_b/body_inertial_quat_b: MuJoCo inertial-frame parameters
      for exactly the same ordered bodies.
    contact_body_indices: indexes into ``B`` for candidate contact bodies.
      Their point offset defaults to each body origin.
  """
  if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape [T, B, 3]")
  t, b, _ = body_pos_w.shape
  expected = (b,)
  for name, value, tail in (
    ("body_mass", body_mass, expected),
    ("body_com_offset_b", body_com_offset_b, (b, 3)),
    ("body_inertia_diag", body_inertia_diag, (b, 3)),
    ("body_inertial_quat_b", body_inertial_quat_b, (b, 4)),
  ):
    if tuple(value.shape) != tail:
      raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tail}")
  if body_mass.sum() <= 0:
    raise ValueError("Total selected body mass must be positive")

  mass = body_mass.view(1, b, 1)
  com_offset_w = quat_apply(body_quat_w.reshape(-1, 4), body_com_offset_b.expand(t, -1, -1).reshape(-1, 3)).reshape(t, b, 3)
  body_com_w = body_pos_w + com_offset_w
  body_com_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, com_offset_w, dim=-1)
  total_mass = body_mass.sum()
  com_pos_w = (mass * body_com_w).sum(dim=1) / total_mass
  com_vel_w = (mass * body_com_vel_w).sum(dim=1) / total_mass
  linear_momentum_w = total_mass * com_vel_w

  inertial_quat = body_inertial_quat_b.expand(t, -1, -1)
  inertia_rot = _rotation_matrix(quat_mul(body_quat_w, inertial_quat))
  inertia_b = torch.diag_embed(body_inertia_diag).expand(t, -1, -1, -1)
  inertia_w = inertia_rot @ inertia_b @ inertia_rot.transpose(-1, -2)
  spin = torch.matmul(inertia_w, body_ang_vel_w.unsqueeze(-1)).squeeze(-1)
  orbital = torch.cross(body_com_w - com_pos_w[:, None, :], mass * body_com_vel_w, dim=-1)
  angular_momentum_w = (spin + orbital).sum(dim=1)

  if contact_body_indices is None:
    contact_body_indices = torch.empty(0, device=body_pos_w.device, dtype=torch.long)
  if contact_body_indices.ndim != 1:
    raise ValueError("contact_body_indices must be one-dimensional")
  c = len(contact_body_indices)
  if contact_point_offset_b is None:
    contact_point_offset_b = torch.zeros(c, 3, device=body_pos_w.device, dtype=body_pos_w.dtype)
  if tuple(contact_point_offset_b.shape) != (c, 3):
    raise ValueError(f"contact_point_offset_b must have shape {(c, 3)}")
  contact_pos_w = body_pos_w[:, contact_body_indices]
  if c:
    contact_quat = body_quat_w[:, contact_body_indices]
    point_offset_w = quat_apply(contact_quat.reshape(-1, 4), contact_point_offset_b.expand(t, -1, -1).reshape(-1, 3)).reshape(t, c, 3)
    contact_pos_w = contact_pos_w + point_offset_w
  return ReferenceCentroidalTrajectory(
    com_pos_w=com_pos_w,
    com_vel_w=com_vel_w,
    linear_momentum_w=linear_momentum_w,
    angular_momentum_w=angular_momentum_w,
    contact_pos_w=contact_pos_w,
    contact_pos_rel_com_w=contact_pos_w - com_pos_w[:, None, :],
  )
