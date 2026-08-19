"""Motion-reference command, observations, and rewards for hybrid imitation.

This is the mjlab port of the *data contract* used by BeyondMimic's G1
``MotionCommand``: a reference ``.npz`` carries named joint trajectories plus
world-frame body pose and velocity trajectories.  Keeping the names in the
file is intentional: silently relying on a joint order is unsafe for humanoid
imitation, especially when a G1 clip is used with a different articulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from training_common.reference_centroidal import prealign_reference_kinematics_to_initial_anchor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionReferenceCommandCfg(CommandTermCfg):
  """Named BeyondMimic reference-motion stream.

  ``motion_file`` deliberately defaults to ``None``.  The repository ships a
  THEMIS model, not a G1 MJCF, so making a G1 clip the implicit default would
  create an invalid task.  A G1 task must provide both its G1 entity and a G1
  clip with matching ``joint_names``/``body_names`` metadata.
  """

  asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))
  motion_file: str | None = None
  joint_names: tuple[str, ...] = ()
  body_names: tuple[str, ...] = ()
  anchor_body_name: str = ""
  # Static clip transform applied exactly once at load time. ``initial_anchor``
  # removes the initial anchor x/y and yaw but preserves height, roll/pitch,
  # joint motion and subsequent root translation. It is deliberately distinct
  # from the local joint-tracker reward: world-frame body poses are retained
  # only to construct reference centroidal/contact quantities for MPC.
  reference_frame_alignment: str = "initial_anchor"  # "none" | "initial_anchor"
  loop: bool = True
  random_start: bool = True
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: "ManagerBasedRlEnv") -> "MotionReferenceCommand":
    return MotionReferenceCommand(self, env)


@dataclass(frozen=True)
class MJLabMotionClip:
  """Named, immutable reference-trajectory tensors on the training device."""

  fps: float
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  source_motion_file: str
  # BeyondMimic/MJLab conversion writes CoM velocity.  The explicit field
  # also keeps CPU-trimmed (origin-Jacobian) clips physically unambiguous.
  body_linear_velocity_point: str

  @property
  def num_frames(self) -> int:
    return int(self.joint_pos.shape[0])


class MJLabMotionLoader:
  """Load and validate the BeyondMimic NPZ contract for MJLab training.

  Offline conversion owns temporal resampling and velocity differentiation;
  this loader deliberately does not resample again, because re-interpolating
  body poses independently at training time would break kinematic and
  centroidal consistency.  It only validates metadata and transfers the
  immutable clip to the simulation device.
  """

  _REQUIRED = {
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names",
  }

  def __init__(self, path: Path, device: torch.device | str):
    if not path.is_file():
      raise FileNotFoundError(f"Motion reference does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
      missing = self._REQUIRED.difference(data.files)
      if missing:
        raise ValueError(f"{path} is not a BeyondMimic reference NPZ; missing {sorted(missing)}")
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      if fps <= 0.0:
        raise ValueError(f"Motion fps must be positive, got {fps}")
      tensors = {
        name: torch.as_tensor(data[name], device=device, dtype=torch.float32)
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
      }
      joint_names = tuple(str(name) for name in data["joint_names"].tolist())
      body_names = tuple(str(name) for name in data["body_names"].tolist())
      source = str(np.asarray(data.get("source_motion_file", "")).reshape(-1)[0])
      velocity_point = str(np.asarray(data.get("body_linear_velocity_point", "inertial_com")).reshape(-1)[0])
      if velocity_point not in {"inertial_com", "link_origin"}:
        raise ValueError(
          "body_linear_velocity_point must be 'inertial_com' or 'link_origin', "
          f"got {velocity_point!r}"
        )
    self.path = path
    self.device = device
    self.clip = MJLabMotionClip(
      fps=fps, joint_names=joint_names, body_names=body_names,
      source_motion_file=source, body_linear_velocity_point=velocity_point, **tensors,
    )
    self._validate()

  def optional_tensor(self, key: str) -> torch.Tensor | None:
    """Read one optional, clip-level label without weakening the core contract."""
    with np.load(self.path, allow_pickle=False) as data:
      if key not in data.files:
        return None
      return torch.as_tensor(data[key], device=self.device, dtype=torch.float32)

  def _validate(self) -> None:
    clip = self.clip
    frames = clip.num_frames
    if frames < 2:
      raise ValueError("Motion reference needs at least two frames")
    if clip.joint_vel.shape != clip.joint_pos.shape:
      raise ValueError("joint_vel must have exactly the same [T,J] shape as joint_pos")
    if len(clip.joint_names) != clip.joint_pos.shape[1] or len(set(clip.joint_names)) != len(clip.joint_names):
      raise ValueError("joint_names must be unique and match joint_pos's second dimension")
    body_shape = (frames, len(clip.body_names))
    expected = {
      "body_pos_w": (*body_shape, 3), "body_quat_w": (*body_shape, 4),
      "body_lin_vel_w": (*body_shape, 3), "body_ang_vel_w": (*body_shape, 3),
    }
    for name, shape in expected.items():
      if tuple(getattr(clip, name).shape) != shape:
        raise ValueError(f"{name} has shape {tuple(getattr(clip, name).shape)}, expected {shape}")
    if len(set(clip.body_names)) != len(clip.body_names):
      raise ValueError("body_names must be unique")
    finite_fields = ("joint_pos", "joint_vel", *expected)
    if not all(bool(torch.isfinite(getattr(clip, name)).all()) for name in finite_fields):
      raise ValueError("Motion trajectory contains NaN or Inf")
    quat_norm = clip.body_quat_w.norm(dim=-1)
    if bool((quat_norm < 1.0e-6).any()):
      raise ValueError("body_quat_w contains a zero-norm quaternion")

  def joint_indices(self, names: tuple[str, ...]) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(self.clip.joint_names)}
    missing = [name for name in names if name not in lookup]
    if missing:
      raise ValueError(f"Reference motion has no requested joints: {missing}")
    return torch.tensor([lookup[name] for name in names], device=self.clip.joint_pos.device, dtype=torch.long)

  def body_indices(self, names: tuple[str, ...]) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(self.clip.body_names)}
    missing = [name for name in names if name not in lookup]
    if missing:
      raise ValueError(f"Reference motion has no requested bodies: {missing}")
    return torch.tensor([lookup[name] for name in names], device=self.clip.body_pos_w.device, dtype=torch.long)


class MotionReferenceCommand(CommandTerm):
  """MJLab counterpart of BeyondMimic ``MotionCommand``.

  It advances a named :class:`MJLabMotionClip` at simulation time, maps its
  q/dq and body channels to the active articulation, applies the static
  reference-frame transform once, and exposes causal, continuously sampled
  tracker references.  Reference centroidal reconstruction and contact-plan
  generation deliberately live in ``mpc_grf_mimic_mdp``.
  """

  cfg: MotionReferenceCommandCfg

  def __init__(self, cfg: MotionReferenceCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    if not cfg.motion_file:
      raise ValueError(
        "commands.motion.motion_file is required. Provide a retargeted clip "
        "whose joint_names and body_names match the simulated robot."
      )
    self.motion = MJLabMotionLoader(Path(cfg.motion_file), self.device)
    clip = self.motion.clip
    self.fps = clip.fps
    self._q = clip.joint_pos
    self._dq = clip.joint_vel
    self._body_pos = clip.body_pos_w
    self._body_quat = clip.body_quat_w
    self._body_lin_vel = clip.body_lin_vel_w
    self._body_ang_vel = clip.body_ang_vel_w
    self._motion_joint_names = clip.joint_names
    self._motion_body_names = clip.body_names
    self.num_frames = clip.num_frames

    self._asset_cfg = cfg.asset_cfg
    self._asset_cfg.resolve(env.scene)
    self.robot = env.scene[self._asset_cfg.name]
    requested_joints = cfg.joint_names or tuple(self.robot.joint_names)
    joint_ids, resolved_joint_names = self.robot.find_joints_by_actuator_names(requested_joints)
    if len(joint_ids) != len(requested_joints):
      raise ValueError(f"Unresolved motion joint names: requested={requested_joints}, resolved={resolved_joint_names}")
    self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self.motion_joint_ids = self.motion.joint_indices(tuple(resolved_joint_names))

    requested_bodies = cfg.body_names or (cfg.anchor_body_name,)
    if not cfg.anchor_body_name:
      raise ValueError("commands.motion.anchor_body_name is required")
    body_ids, resolved_body_names = self.robot.find_bodies(requested_bodies)
    if len(body_ids) != len(requested_bodies):
      raise ValueError(f"Unresolved motion body names: requested={requested_bodies}, resolved={resolved_body_names}")
    self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
    self.motion_body_ids = self.motion.body_indices(tuple(resolved_body_names))
    try:
      self.anchor_index = list(resolved_body_names).index(cfg.anchor_body_name)
    except ValueError as exc:
      raise ValueError("anchor_body_name must be included in body_names") from exc

    if cfg.reference_frame_alignment not in {"none", "initial_anchor"}:
      raise ValueError("reference_frame_alignment must be 'none' or 'initial_anchor'")
    self._reference_frame_alignment = cfg.reference_frame_alignment
    if self._reference_frame_alignment == "initial_anchor":
      # Keep the source NPZ raw/provenance-preserving.  The deterministic
      # canonicalization belongs at command load, where its meaning is shared
      # by pose rewards, reference centroidal computation and MPC contacts.
      anchor_motion_id = int(self.motion_body_ids[self.anchor_index])
      self._body_pos, self._body_quat, self._body_lin_vel, self._body_ang_vel = (
        prealign_reference_kinematics_to_initial_anchor(
          body_pos_w=self._body_pos,
          body_quat_w=self._body_quat,
          body_lin_vel_w=self._body_lin_vel,
          body_ang_vel_w=self._body_ang_vel,
          anchor_body_index=anchor_motion_id,
        )
      )

    self._frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._frame_progress = torch.zeros(self.num_envs, device=self.device)
    self.metrics["joint_pos_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.sample_reference_tensor(self._q, self._frame_progress)[:, self.motion_joint_ids]

  @property
  def joint_vel(self) -> torch.Tensor:
    velocity = self.sample_reference_tensor(self._dq, self._frame_progress)[:, self.motion_joint_ids]
    # A non-looping clip has a static terminal reference. Retaining the last
    # finite-difference velocity would request motion beyond the data.
    if not self.cfg.loop:
      velocity = torch.where(
        (self._frame_progress < self.num_frames - 1).unsqueeze(-1), velocity, torch.zeros_like(velocity)
      )
    return velocity

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self.joint_ids]

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self.joint_ids]

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    """Live body positions for the frozen tracker input, never a reward target."""
    return self.robot.data.body_link_pos_w[:, self.body_ids]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    """Live body orientations for the frozen tracker input, never a reward target."""
    return self.robot.data.body_link_quat_w[:, self.body_ids]

  @property
  def body_pos_w(self) -> torch.Tensor:
    """Continuously sampled reference body positions in simulator-world axes."""
    return self.sample_reference_tensor(self._body_pos, self._frame_progress)[:, self.motion_body_ids] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    """Continuously sampled reference body orientations for a frozen tracker."""
    return self.sample_reference_tensor(self._body_quat, self._frame_progress, quaternion=True)[:, self.motion_body_ids]

  def joint_reference_frame_offset(self, frame_offset: int = 1) -> torch.Tensor:
    """One causal reference preview ``[q_ref, dq_ref]`` at a clip-frame offset."""
    if frame_offset < 0:
      raise ValueError("frame_offset must be non-negative")
    frames = self._normalize_frame_progress(self._frame_progress + float(frame_offset))
    q = self.sample_reference_tensor(self._q, frames)[:, self.motion_joint_ids]
    dq = self.sample_reference_tensor(self._dq, frames)[:, self.motion_joint_ids]
    if not self.cfg.loop:
      valid = (self._frame_progress + frame_offset) < self.num_frames - 1
      dq = torch.where(valid.unsqueeze(-1), dq, torch.zeros_like(dq))
    return torch.cat([q, dq], dim=-1)

  def reference_body_kinematics(
    self, names: tuple[str, ...], label: str,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Return canonical clip channels and matching robot-body ids for MPC use."""
    body_ids, resolved_names = self.robot.find_bodies(names)
    if len(body_ids) != len(names):
      raise ValueError(f"Unresolved {label} body names: requested={names}, resolved={resolved_names}")
    motion_ids = self.motion.body_indices(tuple(resolved_names))
    local_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
    return (
      local_ids, self._body_pos[:, motion_ids], self._body_quat[:, motion_ids],
      self._body_lin_vel[:, motion_ids], self._body_ang_vel[:, motion_ids], tuple(resolved_names),
    )

  def horizon_frame_progress(self, steps: int, dt: float) -> torch.Tensor:
    """Continuous clip coordinates for MPC nodes: ``s_i=s_0+i f_{ref}dt``."""
    if steps < 1:
      raise ValueError("steps must be positive")
    offsets = torch.arange(steps, device=self.device, dtype=self._frame_progress.dtype) * (self.fps * dt)
    return self._normalize_frame_progress(self._frame_progress[:, None] + offsets[None, :])

  def horizon_valid(self, steps: int, dt: float) -> torch.Tensor:
    if self.cfg.loop:
      return torch.ones(self.num_envs, steps, dtype=torch.bool, device=self.device)
    offsets = torch.arange(steps, device=self.device, dtype=self._frame_progress.dtype) * (self.fps * dt)
    return self._frame_progress[:, None] + offsets[None, :] < self.num_frames - 1

  def sample_reference_tensor(
    self, values: torch.Tensor, frame_progress: torch.Tensor, *, quaternion: bool = False,
  ) -> torch.Tensor:
    """Linearly sample a clip tensor at scalar or batched continuous coordinates."""
    if values.shape[0] != self.num_frames:
      raise ValueError("Reference tensor must have the clip frame dimension first")
    progress = self._normalize_frame_progress(frame_progress)
    i0 = progress.floor().long()
    if self.cfg.loop:
      i1 = (i0 + 1).remainder(self.num_frames)
    else:
      i1 = (i0 + 1).clamp(max=self.num_frames - 1)
    alpha = (progress - i0.to(progress.dtype))
    v0, v1 = values[i0], values[i1]
    if quaternion:
      same_hemisphere = (v0 * v1).sum(dim=-1, keepdim=True) >= 0.0
      v1 = torch.where(same_hemisphere, v1, -v1)
      sampled = v0 + alpha.reshape(*alpha.shape, *([1] * (v0.ndim - alpha.ndim))) * (v1 - v0)
      return sampled / sampled.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return v0 + alpha.reshape(*alpha.shape, *([1] * (v0.ndim - alpha.ndim))) * (v1 - v0)

  def _normalize_frame_progress(self, progress: torch.Tensor) -> torch.Tensor:
    if self.cfg.loop:
      return torch.remainder(progress, self.num_frames)
    return progress.clamp(min=0.0, max=float(self.num_frames - 1))

  def _update_metrics(self) -> None:
    self.metrics["joint_pos_error"] += (self.joint_pos - self.robot_joint_pos).norm(dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.random_start:
      self._frame_progress[env_ids] = torch.randint(self.num_frames, (len(env_ids),), device=self.device).float()
    else:
      self._frame_progress[env_ids] = 0.0
    self._frame[env_ids] = self._frame_progress[env_ids].long()

  def _update_command(self) -> None:
    self._frame_progress += self.fps * self._env.step_dt
    if self.cfg.loop:
      self._frame_progress.remainder_(self.num_frames)
    else:
      self._frame_progress.clamp_(max=self.num_frames - 1)
    self._frame.copy_(self._frame_progress.long())


def _motion(env: "ManagerBasedRlEnv", command_name: str) -> MotionReferenceCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MotionReferenceCommand):
    raise TypeError(f"'{command_name}' must be a MotionReferenceCommand")
  return command


def motion_reference(env: "ManagerBasedRlEnv", command_name: str = "motion") -> torch.Tensor:
  """Deployable tracker input: named q/dq reference at the current phase."""
  return _motion(env, command_name).command


def motion_reference_preview(
  env: "ManagerBasedRlEnv", command_name: str = "motion", frame_offset: int = 1
) -> torch.Tensor:
  """Causal one-frame motion-reference preview for the policy actor."""
  return _motion(env, command_name).joint_reference_frame_offset(frame_offset)


def motion_clip_complete(env: "ManagerBasedRlEnv", command_name: str = "motion") -> torch.Tensor:
  """Terminate a non-looping imitation episode at the clip's terminal pose."""
  command = _motion(env, command_name)
  if command.cfg.loop:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return command._frame >= command.num_frames - 1


def motion_joint_error_exp(env: "ManagerBasedRlEnv", command_name: str = "motion", std: float = 0.4) -> torch.Tensor:
  command = _motion(env, command_name)
  return torch.exp(-(command.joint_pos - command.robot_joint_pos).square().mean(dim=-1) / std**2)


def motion_joint_vel_error_exp(env: "ManagerBasedRlEnv", command_name: str = "motion", std: float = 2.0) -> torch.Tensor:
  command = _motion(env, command_name)
  return torch.exp(-(command.joint_vel - command.robot_joint_vel).square().mean(dim=-1) / std**2)
