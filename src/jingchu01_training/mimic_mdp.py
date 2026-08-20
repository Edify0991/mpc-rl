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
import torch.nn.functional as F
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

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
  # BeyondMimic-style failure-biased clip sampling. Bins are one policy
  # second wide by default; the uniform component keeps all frames reachable.
  adaptive_sampling: bool = True
  adaptive_bin_length_s: float = 1.0
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
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
    # Jingchu01 deliberately uses Robotbase as both tracking anchor and
    # floating-base root. Store its MuJoCo model id explicitly so reset can
    # convert the clip's CoM velocity into generalized root velocity.
    self._anchor_body_model_id = int(self.robot.indexing.body_ids[self.body_ids[self.anchor_index]])

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
    if cfg.adaptive_bin_length_s <= 0.0:
      raise ValueError("adaptive_bin_length_s must be positive")
    if cfg.adaptive_kernel_size < 1:
      raise ValueError("adaptive_kernel_size must be at least one")
    if not 0.0 <= cfg.adaptive_uniform_ratio <= 1.0:
      raise ValueError("adaptive_uniform_ratio must lie in [0, 1]")
    if not 0.0 <= cfg.adaptive_alpha <= 1.0:
      raise ValueError("adaptive_alpha must lie in [0, 1]")
    if not 0.0 <= cfg.adaptive_lambda <= 1.0:
      raise ValueError("adaptive_lambda must lie in [0, 1]")
    frames_per_bin = max(1, int(round(cfg.adaptive_bin_length_s * self.fps)))
    self._adaptive_bin_count = max(1, (self.num_frames + frames_per_bin - 1) // frames_per_bin)
    self._adaptive_failure = torch.zeros(self._adaptive_bin_count, device=self.device)
    self._adaptive_kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device,
    )
    self._adaptive_kernel /= self._adaptive_kernel.sum()
    self.metrics["joint_pos_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

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

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    """Reference body linear velocity at the clip's declared point."""
    return self.sample_reference_tensor(self._body_lin_vel, self._frame_progress)[:, self.motion_body_ids]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    """Reference body angular velocity in world axes."""
    return self.sample_reference_tensor(self._body_ang_vel, self._frame_progress)[:, self.motion_body_ids]

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

  def sample_reference_tensor_zoh(self, values: torch.Tensor, frame_progress: torch.Tensor) -> torch.Tensor:
    """Sample a reference tensor with left-continuous zero-order hold.

    This is intentionally separate from :meth:`sample_reference_tensor`.
    Kinematic and centroidal quantities are continuous landmarks and use
    linear interpolation; hybrid quantities such as a binary contact mode
    must retain their source frame exactly.  At an integer frame boundary the
    value is the new (post-event) frame value.
    """
    if values.shape[0] != self.num_frames:
      raise ValueError("Reference tensor must have the clip frame dimension first")
    progress = self._normalize_frame_progress(frame_progress)
    return values[progress.floor().long()]

  def _normalize_frame_progress(self, progress: torch.Tensor) -> torch.Tensor:
    if self.cfg.loop:
      return torch.remainder(progress, self.num_frames)
    return progress.clamp(min=0.0, max=float(self.num_frames - 1))

  def _update_metrics(self) -> None:
    self.metrics["joint_pos_error"] += (self.joint_pos - self.robot_joint_pos).norm(dim=-1)

  def _model_field_per_env(self, name: str) -> torch.Tensor:
    """Read a MuJoCo model field in a normalized ``[E, ...]`` layout."""
    value = getattr(self._env.sim.model, name, None)
    if value is None:
      raise RuntimeError(f"Simulator model does not expose required field {name!r}")
    tensor = torch.as_tensor(value, device=self.device, dtype=self._body_pos.dtype)
    if tensor.ndim == 0:
      return tensor.expand(self.num_envs).clone()
    if tensor.shape[0] != self.num_envs:
      return tensor.unsqueeze(0).expand(self.num_envs, *tensor.shape)
    return tensor

  def _anchor_root_linear_velocity_w(
    self, anchor_quat_w: torch.Tensor, anchor_lin_vel_w: torch.Tensor, anchor_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    """Convert a reference anchor CoM velocity to root-link-origin velocity.

    ``body_pos_w`` is at link origins while the normal BeyondMimic/MJLab
    ``body_lin_vel_w`` contract is inertial-CoM velocity. MuJoCo's free-joint
    translational velocity is at the root-link origin, requiring the
    ``omega x r_com`` shift before the reset write.
    """
    if self.motion.clip.body_linear_velocity_point == "link_origin":
      return anchor_lin_vel_w
    offset_b = self._model_field_per_env("body_ipos")[:, self._anchor_body_model_id]
    offset_w = quat_apply(anchor_quat_w, offset_b)
    return anchor_lin_vel_w - torch.cross(anchor_ang_vel_w, offset_w, dim=-1)

  def _adaptive_start_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample reference frames using BeyondMimic's failure-biased bins."""
    if not self.cfg.adaptive_sampling:
      return torch.randint(self.num_frames, (len(env_ids),), device=self.device)

    terminated = self._env.termination_manager.terminated[env_ids]
    if bool(terminated.any()):
      failed_frames = self._frame_progress[env_ids][terminated]
      failed_bins = (failed_frames * self._adaptive_bin_count / self.num_frames).long()
      failed_bins.clamp_(0, self._adaptive_bin_count - 1)
      failures = torch.bincount(failed_bins, minlength=self._adaptive_bin_count).to(self._adaptive_failure.dtype)
      self._adaptive_failure.mul_(1.0 - self.cfg.adaptive_alpha).add_(failures, alpha=self.cfg.adaptive_alpha)

    probabilities = self._adaptive_failure + self.cfg.adaptive_uniform_ratio / self._adaptive_bin_count
    probabilities = F.pad(
      probabilities.view(1, 1, -1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate",
    )
    probabilities = F.conv1d(probabilities, self._adaptive_kernel.view(1, 1, -1)).view(-1)
    probabilities /= probabilities.sum().clamp_min(1.0e-12)
    sampled_bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
    frames = (
      (sampled_bins.to(torch.float32) + torch.rand(len(env_ids), device=self.device))
      / self._adaptive_bin_count * (self.num_frames - 1)
    ).long()

    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    normalized_entropy = entropy / float(np.log(self._adaptive_bin_count)) if self._adaptive_bin_count > 1 else 1.0
    top_prob, top_bin = probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = normalized_entropy
    self.metrics["sampling_top1_prob"][:] = top_prob
    self.metrics["sampling_top1_bin"][:] = top_bin.to(torch.float32) / self._adaptive_bin_count
    return frames

  def _write_reference_reset_state(self, env_ids: torch.Tensor) -> None:
    """Write root pose/velocity and joint state from the sampled reference."""
    anchor_pos_w = self.body_pos_w[:, self.anchor_index]
    anchor_quat_w = self.body_quat_w[:, self.anchor_index]
    anchor_lin_vel_w = self.body_lin_vel_w[:, self.anchor_index]
    anchor_ang_vel_w = self.body_ang_vel_w[:, self.anchor_index]
    root_lin_vel_w = self._anchor_root_linear_velocity_w(
      anchor_quat_w, anchor_lin_vel_w, anchor_ang_vel_w,
    )
    root_state = self.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = anchor_pos_w[env_ids]
    root_state[:, 3:7] = anchor_quat_w[env_ids]
    root_state[:, 7:10] = root_lin_vel_w[env_ids]
    root_state[:, 10:13] = anchor_ang_vel_w[env_ids]

    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
    reference_q, reference_dq = self.joint_pos[env_ids], self.joint_vel[env_ids]
    limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
    if limits is not None:
      selected_limits = limits[env_ids][:, self.joint_ids]
      reference_q = reference_q.clamp(selected_limits[..., 0], selected_limits[..., 1])
    joint_pos[:, self.joint_ids] = reference_q
    joint_vel[:, self.joint_ids] = reference_dq
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    if self.cfg.random_start:
      self._frame_progress[env_ids] = self._adaptive_start_frames(env_ids).to(self._frame_progress.dtype)
    else:
      self._frame_progress[env_ids] = 0.0
    self._frame[env_ids] = self._frame_progress[env_ids].long()
    # Match BeyondMimic's reset contract. The first MimicLocoMPC solve thus
    # reads x0 from this sampled root/joint state, not a standing reset.
    self._write_reference_reset_state(env_ids)

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
