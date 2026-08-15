"""Reusable DAgger storage and losses for MPC-RL teacher-to-student transfer.

The Phase-1 teacher has privileged full-horizon MPC/contact information while
the Phase-2 student is causal and emits only ``q_des``.  This module is kept
independent of rsl_rl internals: a rollout collector queries the frozen teacher
on student-visited states, stores the matched action-coordinate targets here, and interleaves
``update_student`` calls with normal PPO updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DaggerLossCfg:
  """Weights for causal student imitation of a privileged teacher."""

  q_des_weight: float = 1.0
  landmark_weight: float = 0.1


class DaggerReplayBuffer:
  """Ring buffer of student observations and teacher targets.

  ``teacher_landmark`` is optional and can hold the concatenated MPC CoM,
  momentum, force, and contact-plan landmarks. It trains an auxiliary student
  head only; it is never required at deployment.
  """

  def __init__(
    self,
    capacity: int,
    observation_dim: int,
    action_dim: int,
    landmark_dim: int = 0,
    device: torch.device | str = "cpu",
  ) -> None:
    if capacity < 1 or observation_dim < 1 or action_dim < 1 or landmark_dim < 0:
      raise ValueError("invalid DAgger replay-buffer dimensions")
    self.capacity = capacity
    self.device = torch.device(device)
    self.observations = torch.zeros(capacity, observation_dim, device=self.device)
    self.teacher_joint_action = torch.zeros(capacity, action_dim, device=self.device)
    self.teacher_landmark = torch.zeros(capacity, landmark_dim, device=self.device)
    self._size = 0
    self._cursor = 0

  def __len__(self) -> int:
    return self._size

  @torch.no_grad()
  def add(
    self,
    observations: Tensor,
    teacher_joint_action: Tensor,
    teacher_landmark: Tensor | None = None,
  ) -> None:
    """Append a teacher query in the student's normalized joint-action space.

    The environment maps this target to ``q_des`` as
    ``q_ref + action_scale * teacher_joint_action``.  This avoids comparing a
    raw Gaussian policy output directly to an absolute joint-angle target.
    """
    if observations.ndim != 2 or teacher_joint_action.ndim != 2:
      raise ValueError("observations and teacher_joint_action must be rank-two")
    batch = observations.shape[0]
    if batch != teacher_joint_action.shape[0]:
      raise ValueError("observation/teacher batch sizes must agree")
    if observations.shape[1] != self.observations.shape[1] or teacher_joint_action.shape[1] != self.teacher_joint_action.shape[1]:
      raise ValueError("DAgger sample dimensions do not match replay buffer")
    if self.teacher_landmark.shape[1] == 0:
      if teacher_landmark is not None and teacher_landmark.numel() != 0:
        raise ValueError("buffer was created without landmark storage")
    else:
      if teacher_landmark is None or teacher_landmark.shape != (batch, self.teacher_landmark.shape[1]):
        raise ValueError("teacher_landmark has an invalid shape")
    # Keep the most recent capacity samples when a vectorized rollout is
    # larger than the buffer.
    if batch > self.capacity:
      observations = observations[-self.capacity:]
      teacher_joint_action = teacher_joint_action[-self.capacity:]
      teacher_landmark = None if teacher_landmark is None else teacher_landmark[-self.capacity:]
      batch = self.capacity
    indices = (torch.arange(batch, device=self.device) + self._cursor) % self.capacity
    self.observations[indices] = observations.to(self.device)
    self.teacher_joint_action[indices] = teacher_joint_action.to(self.device)
    if teacher_landmark is not None:
      self.teacher_landmark[indices] = teacher_landmark.to(self.device)
    self._cursor = (self._cursor + batch) % self.capacity
    self._size = min(self.capacity, self._size + batch)

  def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor | None]:
    if self._size == 0:
      raise RuntimeError("cannot sample an empty DAgger replay buffer")
    if batch_size < 1:
      raise ValueError("batch_size must be positive")
    indices = torch.randint(self._size, (batch_size,), device=self.device)
    landmarks = self.teacher_landmark[indices] if self.teacher_landmark.shape[1] else None
    return self.observations[indices], self.teacher_joint_action[indices], landmarks


def teacher_joint_action_from_full_action(teacher_action: Tensor, num_joints: int = 29) -> Tensor:
  """Extract the deployable joint-target component of the Phase-1 action.

  Phase-1 appends its ``2H`` contact-plan residual after the joint component;
  Phase-2 deliberately has no such output.  The returned tensor is detached
  because teacher queries are fixed supervised targets.
  """
  if teacher_action.ndim != 2 or teacher_action.shape[1] < num_joints:
    raise ValueError("teacher_action must have shape [B, at least num_joints]")
  return teacher_action[:, :num_joints].detach()


def dagger_distillation_loss(
  student_joint_action: Tensor,
  teacher_joint_action: Tensor,
  *,
  student_landmark: Tensor | None = None,
  teacher_landmark: Tensor | None = None,
  cfg: DaggerLossCfg = DaggerLossCfg(),
) -> Tensor:
  """Return MSE loss for joint targets and optional auxiliary landmarks.

  Both joint tensors use the normalized policy-action coordinates.  They are
  equivalent to matching ``q_des`` because teacher and student share the same
  reference target and action scale in the paired rollout.
  """
  if student_joint_action.shape != teacher_joint_action.shape:
    raise ValueError("student_joint_action and teacher_joint_action must have the same shape")
  loss = cfg.q_des_weight * (student_joint_action - teacher_joint_action).square().mean()
  if (student_landmark is None) != (teacher_landmark is None):
    raise ValueError("student/teacher landmarks must either both be supplied or both be absent")
  if student_landmark is not None:
    if student_landmark.shape != teacher_landmark.shape:
      raise ValueError("student and teacher landmarks must have the same shape")
    loss = loss + cfg.landmark_weight * (student_landmark - teacher_landmark).square().mean()
  return loss


def update_student(
  *,
  student: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  observations: Tensor,
  teacher_joint_action: Tensor,
  student_landmark_head: torch.nn.Module | None = None,
  teacher_landmark: Tensor | None = None,
  cfg: DaggerLossCfg = DaggerLossCfg(),
) -> Tensor:
  """One DAgger supervised update; caller controls teacher-query collection."""
  student_joint_action = student(observations)
  if isinstance(student_joint_action, (tuple, list)):
    student_joint_action = student_joint_action[0]
  student_landmark = None if student_landmark_head is None else student_landmark_head(observations)
  loss = dagger_distillation_loss(
    student_joint_action,
    teacher_joint_action,
    student_landmark=student_landmark,
    teacher_landmark=teacher_landmark,
    cfg=cfg,
  )
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  return loss.detach()
