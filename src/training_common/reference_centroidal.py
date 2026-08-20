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
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat


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


@dataclass(frozen=True)
class CentroidalState:
  """Whole-robot centroidal state for a batch of simulator environments."""

  com_pos_w: torch.Tensor
  com_vel_w: torch.Tensor
  linear_momentum_w: torch.Tensor
  angular_momentum_w: torch.Tensor


def prealign_reference_kinematics_to_initial_anchor(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  anchor_body_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Canonicalize a reference clip before centroidal reconstruction.

  The convention follows BeyondMimic's anchor-relative tracking treatment:
  remove only the arbitrary initial anchor ``x/y`` and initial anchor yaw,
  while retaining all heights, roll/pitch, and subsequent global translation.
  The result is a clip in a canonical reference world, ready to be placed at
  an IsaacLab environment origin.  It is deliberately a *fixed*, clip-level
  preprocessing transform, not the runtime robot-error alignment used by a
  body-pose tracking observation.

  All input/output quaternions use ``[w, x, y, z]`` and every vector is in a
  world frame.  A proper yaw rotation maps both polar and axial vectors by
  the same matrix, so linear and angular velocities are transformed equally.
  """
  if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape [T, B, 3]")
  if body_quat_w.shape != (*body_pos_w.shape[:2], 4):
    raise ValueError("body_quat_w must have shape [T, B, 4]")
  if body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape:
    raise ValueError("body linear/angular velocity must have shape [T, B, 3]")
  _, bodies, _ = body_pos_w.shape
  if not 0 <= anchor_body_index < bodies:
    raise ValueError(f"anchor_body_index={anchor_body_index} is outside [0, {bodies})")

  anchor_pos_0 = body_pos_w[0, anchor_body_index]
  # This is exactly the yaw-only part used by BeyondMimic's anchor-relative
  # target construction, frozen at the start of the reference clip.
  q_align = quat_inv(yaw_quat(body_quat_w[0, anchor_body_index].unsqueeze(0))).squeeze(0)
  frames = body_pos_w.shape[0]
  q = q_align.view(1, 1, 4).expand(frames, bodies, 4)

  xy_origin = anchor_pos_0.clone()
  xy_origin[2] = 0.0

  def _rotate(vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(q.reshape(-1, 4), vector.reshape(-1, 3)).reshape_as(vector)

  return (
    _rotate(body_pos_w - xy_origin.view(1, 1, 3)),
    quat_mul(q.reshape(-1, 4), body_quat_w.reshape(-1, 4)).reshape_as(body_quat_w),
    _rotate(body_lin_vel_w),
    _rotate(body_ang_vel_w),
  )


def _rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
  """Quaternion [w, x, y, z] to rotation matrix for arbitrary leading dims."""
  w, x, y, z = quat.unbind(dim=-1)
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ), dim=-1).reshape(*quat.shape[:-1], 3, 3)


def compute_centroidal_state(
  *,
  body_link_pos_w: torch.Tensor,
  body_link_quat_w: torch.Tensor,
  body_com_lin_vel_w: torch.Tensor,
  body_link_ang_vel_w: torch.Tensor,
  body_mass: torch.Tensor,
  body_com_offset_b: torch.Tensor,
  body_inertia_diag: torch.Tensor,
  body_inertial_quat_b: torch.Tensor,
) -> CentroidalState:
  """Compute exact whole-articulation centroidal state from mjlab tensors.

  ``mjlab`` supplies link-origin poses and inertial-CoM velocities for every
  articulation body in the simulator world.  This routine shifts only the
  *position* from link origin to CoM, then sums spin and orbital momentum, so
  no root-inertia approximation enters.  Shapes are ``[E, B, ...]`` for E
  parallel environments and B robot bodies.
  """
  if body_link_pos_w.ndim != 3 or body_link_pos_w.shape[-1] != 3:
    raise ValueError("body_link_pos_w must have shape [E, B, 3]")
  environments, bodies, _ = body_link_pos_w.shape
  for name, value, expected in (
    ("body_link_quat_w", body_link_quat_w, (environments, bodies, 4)),
    ("body_com_lin_vel_w", body_com_lin_vel_w, (environments, bodies, 3)),
    ("body_link_ang_vel_w", body_link_ang_vel_w, (environments, bodies, 3)),
  ):
    if tuple(value.shape) != expected:
      raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
  def _per_env(value: torch.Tensor, tail: tuple[int, ...], name: str) -> torch.Tensor:
    if tuple(value.shape) == (bodies, *tail):
      return value.unsqueeze(0).expand(environments, -1, *([-1] * len(tail)))
    if tuple(value.shape) == (environments, bodies, *tail):
      return value
    raise ValueError(
      f"{name} has shape {tuple(value.shape)}, expected [B,{','.join(map(str, tail))}] "
      f"or [E,B,{','.join(map(str, tail))}]"
    )

  body_mass = _per_env(body_mass, (), "body_mass")
  body_com_offset_b = _per_env(body_com_offset_b, (3,), "body_com_offset_b")
  body_inertia_diag = _per_env(body_inertia_diag, (3,), "body_inertia_diag")
  body_inertial_quat_b = _per_env(body_inertial_quat_b, (4,), "body_inertial_quat_b")
  total_mass = body_mass.sum(dim=1)
  if torch.any(total_mass <= 0):
    raise ValueError("Total body mass must be positive")

  mass = body_mass.unsqueeze(-1)
  com_offset_w = quat_apply(
    body_link_quat_w.reshape(-1, 4),
    body_com_offset_b.reshape(-1, 3),
  ).reshape(environments, bodies, 3)
  body_com_w = body_link_pos_w + com_offset_w
  # In MJLab, ``body_com_lin_vel_w`` is already the velocity of the rigid
  # body's inertial CoM.  Adding omega×d here would double-shift it.
  body_com_vel_w = body_com_lin_vel_w
  com_pos_w = (mass * body_com_w).sum(dim=1) / total_mass.unsqueeze(-1)
  com_vel_w = (mass * body_com_vel_w).sum(dim=1) / total_mass.unsqueeze(-1)
  linear_momentum_w = total_mass.unsqueeze(-1) * com_vel_w

  inertial_quat = body_inertial_quat_b
  inertia_rot = _rotation_matrix(quat_mul(body_link_quat_w, inertial_quat))
  inertia_b = torch.diag_embed(body_inertia_diag)
  inertia_w = inertia_rot @ inertia_b @ inertia_rot.transpose(-1, -2)
  spin = torch.matmul(inertia_w, body_link_ang_vel_w.unsqueeze(-1)).squeeze(-1)
  orbital = torch.cross(
    body_com_w - com_pos_w[:, None, :], mass * body_com_vel_w, dim=-1,
  )
  return CentroidalState(
    com_pos_w=com_pos_w,
    com_vel_w=com_vel_w,
    linear_momentum_w=linear_momentum_w,
    angular_momentum_w=(spin + orbital).sum(dim=1),
  )


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
  body_linear_velocity_point: str = "inertial_com",
) -> ReferenceCentroidalTrajectory:
  """Compute CoM, momentum and contact vectors for every reference frame.

  Args:
    body_*: named-reference body trajectories with shape ``[T, B, ...]``.
      ``body_pos_w`` is at link origin. BeyondMimic/MJLab's
      ``body_lin_vel_w`` is at inertial CoM by default; use
      ``body_linear_velocity_point='link_origin'`` only for origin-Jacobian
      data.
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
  if body_linear_velocity_point == "inertial_com":
    body_com_vel_w = body_lin_vel_w
  elif body_linear_velocity_point == "link_origin":
    body_com_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, com_offset_w, dim=-1)
  else:
    raise ValueError(
      "body_linear_velocity_point must be 'inertial_com' or 'link_origin', "
      f"got {body_linear_velocity_point!r}"
    )
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
