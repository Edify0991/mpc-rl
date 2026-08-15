"""Shared BeyondMimic-compatible motion I/O for the MuJoCo/MJLab stack.

It converts the same retargeted CSV/PKL convention used by BeyondMimic into a
state sequence, then captures the resulting kinematics through the MJLab
articulation runtime used by training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

_NUMPY_PICKLE_MODULE_REMAP = {
  "numpy._core": "numpy.core",
  "numpy._core.multiarray": "numpy.core.multiarray",
  "numpy._core.numeric": "numpy.core.numeric",
  "numpy._core.umath": "numpy.core.umath",
  "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
}


class _NumpyCompatUnpickler(pickle.Unpickler):
  def find_class(self, module: str, name: str):  # type: ignore[no-untyped-def]
    return super().find_class(_NUMPY_PICKLE_MODULE_REMAP.get(module, module), name)


def load_pickle_compat(path: Path) -> object:
  """Load a PKL, including archives created with NumPy's old private name."""
  with path.open("rb") as file:
    try:
      return pickle.load(file)
    except ModuleNotFoundError as error:
      if "numpy._core" not in str(error):
        raise
  with path.open("rb") as file:
    return _NumpyCompatUnpickler(file).load()


@dataclass(frozen=True)
class RetargetedMotion:
  root_pos_w: np.ndarray       # [T, 3]
  root_quat_w: np.ndarray      # [T, 4], wxyz
  joint_pos: np.ndarray        # [T, J]
  fps: float
  source_format: str
  pkl_root_rot_order: str | None = None


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float64)
  norm = np.linalg.norm(quat, axis=-1, keepdims=True)
  return quat / np.maximum(norm, 1e-12)


def _upright_tilt_deg_wxyz(quat: np.ndarray) -> float:
  x, y = quat[:, 1], quat[:, 2]
  return float(np.median(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))))


def _slice_frames(frame_range: tuple[int, int] | None, *arrays: np.ndarray) -> list[np.ndarray]:
  if frame_range is None:
    return list(arrays)
  start, end = frame_range
  if start < 1 or end < start:
    raise ValueError("--frame-range expects 1-indexed inclusive START END with 1 <= START <= END")
  out = [array[start - 1:end] for array in arrays]
  if not out or out[0].shape[0] == 0:
    raise ValueError(f"Frame range {frame_range} selects no frames")
  return out


def load_retargeted_motion(
  path: Path,
  *,
  input_format: str = "auto",
  input_fps: float = 30.0,
  frame_range: tuple[int, int] | None = None,
  pkl_root_rot_order: str = "auto",
) -> RetargetedMotion:
  """Load BeyondMimic-style CSV or PKL retargeting output.

  CSV columns are ``root_pos(3), root_quat_xyzw(4), joint_pos(J)``.  PKL
  requires ``root_pos``, ``root_rot``, ``dof_pos`` and may provide ``fps``.
  """
  if input_format == "auto":
    input_format = "pkl" if path.suffix.lower() in {".pkl", ".pickle"} else "csv"
  if input_format not in {"csv", "pkl"}:
    raise ValueError(f"Unsupported input format {input_format!r}")

  if input_format == "csv":
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[1] < 8:
      raise ValueError("CSV must contain root xyz, root quaternion xyzw, and at least one joint")
    root_pos, quat_xyzw, joint_pos = _slice_frames(frame_range, values[:, :3], values[:, 3:7], values[:, 7:])
    if input_fps <= 0.0:
      raise ValueError("--input-fps must be positive")
    return RetargetedMotion(
      root_pos_w=np.asarray(root_pos, dtype=np.float64),
      root_quat_w=_normalize_quat(np.asarray(quat_xyzw, dtype=np.float64)[:, [3, 0, 1, 2]]),
      joint_pos=np.asarray(joint_pos, dtype=np.float64), fps=float(input_fps), source_format="csv",
    )

  payload = load_pickle_compat(path)
  if not isinstance(payload, dict):
    raise TypeError(f"PKL must contain a dictionary, got {type(payload)!r}")
  missing = {"root_pos", "root_rot", "dof_pos"}.difference(payload)
  if missing:
    raise KeyError(f"PKL is missing required keys: {sorted(missing)}")
  root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
  raw_quat = np.asarray(payload["root_rot"], dtype=np.float64)
  joint_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
  if root_pos.ndim != 2 or root_pos.shape[1] != 3 or raw_quat.shape != (root_pos.shape[0], 4):
    raise ValueError("PKL root_pos/root_rot must have shapes [T,3] and [T,4]")
  if joint_pos.ndim != 2 or joint_pos.shape[0] != root_pos.shape[0]:
    raise ValueError("PKL dof_pos must have shape [T,J] with the same T")
  if pkl_root_rot_order == "wxyz":
    quat, resolved_order = _normalize_quat(raw_quat), "wxyz"
  elif pkl_root_rot_order == "xyzw":
    quat, resolved_order = _normalize_quat(raw_quat[:, [3, 0, 1, 2]]), "xyzw"
  elif pkl_root_rot_order == "auto":
    as_wxyz = _normalize_quat(raw_quat)
    as_xyzw = _normalize_quat(raw_quat[:, [3, 0, 1, 2]])
    if _upright_tilt_deg_wxyz(as_xyzw) + 1e-3 < _upright_tilt_deg_wxyz(as_wxyz):
      quat, resolved_order = as_xyzw, "xyzw"
    else:
      quat, resolved_order = as_wxyz, "wxyz"
  else:
    raise ValueError("--pkl-root-rot-order must be auto, wxyz, or xyzw")
  root_pos, quat, joint_pos = _slice_frames(frame_range, root_pos, quat, joint_pos)
  fps = float(payload.get("fps", input_fps))
  if fps <= 0.0:
    raise ValueError("PKL fps must be positive")
  return RetargetedMotion(root_pos, quat, joint_pos, fps, "pkl", resolved_order)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:] *= -1.0
  return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack((aw * bw - ax * bx - ay * by - az * bz,
                   aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw), axis=-1)


def _slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
  dot = np.sum(a * b, axis=-1, keepdims=True)
  b = np.where(dot < 0.0, -b, b)
  dot = np.abs(dot)
  theta = np.arccos(np.clip(dot, -1.0, 1.0))
  sin_theta = np.sin(theta)
  linear = _normalize_quat((1.0 - blend[:, None]) * a + blend[:, None] * b)
  spherical = (np.sin((1.0 - blend[:, None]) * theta) / np.maximum(sin_theta, 1e-12)) * a + (
    np.sin(blend[:, None] * theta) / np.maximum(sin_theta, 1e-12)
  ) * b
  return np.where(dot > 0.9995, linear, spherical)


def resample_motion(motion: RetargetedMotion, output_fps: float) -> RetargetedMotion:
  """Lerp translations/joints and SLERP root pose at the requested rate."""
  if motion.root_pos_w.shape[0] < 2:
    raise ValueError("At least two motion frames are required")
  if output_fps <= 0.0:
    raise ValueError("--output-fps must be positive")
  duration = (motion.root_pos_w.shape[0] - 1) / motion.fps
  # Match BeyondMimic ``csv_to_npz.py``: its torch.arange excludes the
  # terminal sample, preventing one duplicate endpoint when clips loop.
  times = np.arange(0.0, duration, 1.0 / output_fps)
  if len(times) < 2:
    raise ValueError("Resampling produced fewer than two frames; use a longer clip or higher output FPS")
  phase = np.clip(times / duration, 0.0, 1.0) * (motion.root_pos_w.shape[0] - 1)
  index0 = np.floor(phase).astype(np.int64)
  index1 = np.minimum(index0 + 1, motion.root_pos_w.shape[0] - 1)
  blend = phase - index0
  root_pos = (1.0 - blend[:, None]) * motion.root_pos_w[index0] + blend[:, None] * motion.root_pos_w[index1]
  joint_pos = (1.0 - blend[:, None]) * motion.joint_pos[index0] + blend[:, None] * motion.joint_pos[index1]
  return RetargetedMotion(root_pos, _slerp(motion.root_quat_w[index0], motion.root_quat_w[index1], blend), joint_pos,
                          float(output_fps), motion.source_format, motion.pkl_root_rot_order)


def generalized_velocities(motion: RetargetedMotion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Finite-difference root linear/angular and joint velocities in world axes.

  The SO(3) rule deliberately matches BeyondMimic's ``_so3_derivative``:
  central relative rotation for interior frames and duplicated nearest interior
  value at the two endpoints.
  """
  dt = 1.0 / motion.fps
  root_lin = np.gradient(motion.root_pos_w, dt, axis=0, edge_order=1)
  joint_vel = np.gradient(motion.joint_pos, dt, axis=0, edge_order=1)
  quat = motion.root_quat_w
  if len(quat) == 2:
    q_rel = _quat_mul(quat[1:], _quat_conjugate(quat[:-1]))
    q_rel = np.where(q_rel[:, :1] < 0.0, -q_rel, q_rel)
    xyz_norm = np.linalg.norm(q_rel[:, 1:], axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(q_rel[:, 0], -1.0, 1.0))
    angular = q_rel[:, 1:] * (angle / np.maximum(xyz_norm, 1e-12) / dt)[:, None]
    return root_lin, np.repeat(angular, 2, axis=0), joint_vel

  q_rel = _quat_mul(quat[2:], _quat_conjugate(quat[:-2]))
  q_rel = np.where(q_rel[:, :1] < 0.0, -q_rel, q_rel)
  xyz_norm = np.linalg.norm(q_rel[:, 1:], axis=-1)
  angle = 2.0 * np.arctan2(xyz_norm, np.clip(q_rel[:, 0], -1.0, 1.0))
  angular_inner = q_rel[:, 1:] * (angle / np.maximum(xyz_norm, 1e-12) / (2.0 * dt))[:, None]
  return root_lin, np.concatenate((angular_inner[:1], angular_inner, angular_inner[-1:]), axis=0), joint_vel


def robot_profile(robot: str) -> tuple[mujoco.MjModel, tuple[str, ...]]:
  """Compile a committed MJLab robot profile and return its joint order."""
  if robot == "themis":
    from themis_training.themis.themis_constants import JOINT_NAMES_EXPR, get_spec
    joint_names = tuple(JOINT_NAMES_EXPR)
  elif robot == "g1":
    from g1_training.model import actuated_joint_names, compile_model
    model = compile_model()
    return model, actuated_joint_names(model)
  elif robot == "jingchu01":
    from jingchu01_training.model import actuated_joint_names, compile_model
    model = compile_model()
    return model, actuated_joint_names(model)
  else:
    raise ValueError("--robot must be themis, g1, or jingchu01")
  return get_spec().compile(), joint_names


def robot_entity_cfg(robot: str):
  """Return the MJLab entity configuration used to capture a reference clip."""
  if robot == "themis":
    from themis_training.themis.themis_constants import get_themis_effort_robot_cfg
    return get_themis_effort_robot_cfg()
  if robot == "g1":
    from g1_training.g1.g1_constants import get_g1_effort_robot_cfg
    return get_g1_effort_robot_cfg()
  if robot == "jingchu01":
    from jingchu01_training.jingchu01.jingchu01_constants import get_jingchu01_effort_robot_cfg
    return get_jingchu01_effort_robot_cfg()
  raise ValueError("--robot must be themis, g1, or jingchu01")


def capture_motion_through_mjlab_simulator(
  motion: RetargetedMotion,
  *,
  robot: Literal["themis", "g1", "jingchu01"],
  joint_names: tuple[str, ...],
  sim_dt: float,
  device: str,
) -> dict[str, np.ndarray]:
  """Replay a clip through MJLab and read the resulting simulator state.

  This intentionally mirrors BeyondMimic's conversion path: write a root and
  joint state into the articulation, call a no-integration simulator forward
  pass, then log the articulation data.  Consequently body pose/velocity
  conventions come from the same MJLab runtime used at training, rather than
  from a parallel hand-written MuJoCo kinematics implementation.
  """
  if sim_dt <= 0.0:
    raise ValueError("--sim-dt must be positive")
  if motion.joint_pos.shape[1] != len(joint_names):
    raise ValueError(f"Motion has {motion.joint_pos.shape[1]} joints, but profile expects {len(joint_names)}")

  import torch
  from mjlab.entity import Entity
  from mjlab.sim import Simulation, SimulationCfg

  entity = Entity(robot_entity_cfg(robot))
  simulation_cfg = SimulationCfg()
  simulation_cfg.mujoco.timestep = sim_dt
  simulation = Simulation(1, simulation_cfg, entity.compile(), device)
  entity.initialize(simulation.mj_model, simulation.model, simulation.data, device)
  simulation.reset()

  joint_ids, resolved_joint_names = entity.find_joints(joint_names, preserve_order=True)
  if tuple(resolved_joint_names) != joint_names:
    raise ValueError(
      f"MJLab entity could not resolve requested joint ordering: "
      f"expected={joint_names}, resolved={tuple(resolved_joint_names)}"
    )
  joint_ids_t = torch.tensor(joint_ids, device=device, dtype=torch.long)
  T, B = motion.root_pos_w.shape[0], entity.num_bodies
  log = {
    "joint_pos": np.empty((T, entity.num_joints), np.float32),
    "joint_vel": np.empty((T, entity.num_joints), np.float32),
    "root_pos_w": motion.root_pos_w.astype(np.float32),
    "root_quat_w": motion.root_quat_w.astype(np.float32),
    # These are the root-link generalized velocities that were written into
    # MuJoCo.  They remain useful for replay; the body channels below are
    # read directly from MJLab EntityData.
    "root_lin_vel_w": np.empty((T, 3), np.float32),
    "root_ang_vel_w": np.empty((T, 3), np.float32),
    "body_pos_w": np.empty((T, B, 3), np.float32),
    "body_quat_w": np.empty((T, B, 4), np.float32),
    "body_lin_vel_w": np.empty((T, B, 3), np.float32),
    "body_ang_vel_w": np.empty((T, B, 3), np.float32),
    # Log the entity's native orders, rather than assuming that a retargeter
    # input order happens to match MJLab's articulation order.
    "joint_names": np.asarray(entity.joint_names),
    "body_names": np.asarray(entity.body_names),
  }
  root_lin, root_ang, joint_vel = generalized_velocities(motion)
  for frame in range(T):
    root_state = torch.as_tensor(
      np.concatenate((motion.root_pos_w[frame], motion.root_quat_w[frame], root_lin[frame], root_ang[frame])),
      device=device, dtype=torch.float32,
    ).unsqueeze(0)
    joint_pos = entity.data.default_joint_pos.clone()
    joint_velocity = entity.data.default_joint_vel.clone()
    joint_pos[:, joint_ids_t] = torch.as_tensor(motion.joint_pos[frame], device=device, dtype=torch.float32)
    joint_velocity[:, joint_ids_t] = torch.as_tensor(joint_vel[frame], device=device, dtype=torch.float32)
    entity.write_root_state_to_sim(root_state)
    entity.write_joint_state_to_sim(joint_pos, joint_velocity)
    # This is MJLab's no-integration equivalent of BeyondMimic's render/update
    # capture step.  We must not call ``step``: this is a kinematic reference,
    # not a physically simulated rollout.
    simulation.forward()
    entity.update(sim_dt)
    log["joint_pos"][frame] = entity.data.joint_pos[0].detach().cpu().numpy()
    log["joint_vel"][frame] = entity.data.joint_vel[0].detach().cpu().numpy()
    log["root_lin_vel_w"][frame] = entity.data.root_link_lin_vel_w[0].detach().cpu().numpy()
    log["root_ang_vel_w"][frame] = entity.data.root_link_ang_vel_w[0].detach().cpu().numpy()
    log["body_pos_w"][frame] = entity.data.body_link_pos_w[0].detach().cpu().numpy()
    log["body_quat_w"][frame] = entity.data.body_link_quat_w[0].detach().cpu().numpy()
    # Same semantic contract as BeyondMimic/IsaacLab's body_lin_vel_w:
    # position is link origin, while linear velocity is at inertial CoM.
    log["body_lin_vel_w"][frame] = entity.data.body_com_lin_vel_w[0].detach().cpu().numpy()
    log["body_ang_vel_w"][frame] = entity.data.body_link_ang_vel_w[0].detach().cpu().numpy()
  return log


def reconstruct_body_kinematics(
  model: mujoco.MjModel,
  motion: RetargetedMotion,
  joint_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
  """Legacy CPU diagnostic reconstruction using direct MuJoCo Jacobians.

  Unlike :func:`capture_motion_through_mjlab_simulator`, this routine reports
  ``body_lin_vel_w`` at each *link origin*.  It is retained for headless
  trimming/diagnostics only; any centroidal consumer must mark its output as
  ``body_linear_velocity_point='link_origin'``.
  """
  if motion.joint_pos.shape[1] != len(joint_names):
    raise ValueError(f"Motion has {motion.joint_pos.shape[1]} joints, but profile expects {len(joint_names)}")
  joint_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
  if np.any(joint_ids < 0):
    missing = [name for name, jid in zip(joint_names, joint_ids, strict=True) if jid < 0]
    raise ValueError(f"Profile joints missing from compiled MJCF: {missing}")
  free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if len(free_ids) != 1:
    raise ValueError(f"Expected exactly one robot free joint, got {len(free_ids)}")
  root_qpos = int(model.jnt_qposadr[free_ids[0]])
  root_dof = int(model.jnt_dofadr[free_ids[0]])
  joint_qpos, joint_dof = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
  body_ids = np.arange(1, model.nbody, dtype=np.int32)  # omit MuJoCo world body
  body_names = np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids])
  T, B = motion.root_pos_w.shape[0], len(body_ids)
  root_lin, root_ang, joint_vel = generalized_velocities(motion)
  out = {
    "joint_pos": motion.joint_pos.astype(np.float32), "joint_vel": joint_vel.astype(np.float32),
    "root_pos_w": motion.root_pos_w.astype(np.float32), "root_quat_w": motion.root_quat_w.astype(np.float32),
    "root_lin_vel_w": root_lin.astype(np.float32), "root_ang_vel_w": root_ang.astype(np.float32),
    "body_pos_w": np.empty((T, B, 3), np.float32), "body_quat_w": np.empty((T, B, 4), np.float32),
    "body_lin_vel_w": np.empty((T, B, 3), np.float32), "body_ang_vel_w": np.empty((T, B, 3), np.float32),
    "body_names": body_names,
  }
  data = mujoco.MjData(model)
  jac_pos, jac_rot = np.empty((3, model.nv)), np.empty((3, model.nv))
  for frame in range(T):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[root_qpos:root_qpos + 3] = motion.root_pos_w[frame]
    data.qpos[root_qpos + 3:root_qpos + 7] = motion.root_quat_w[frame]
    data.qpos[joint_qpos] = motion.joint_pos[frame]
    data.qvel[root_dof:root_dof + 3] = root_lin[frame]
    data.qvel[root_dof + 3:root_dof + 6] = root_ang[frame]
    data.qvel[joint_dof] = joint_vel[frame]
    mujoco.mj_forward(model, data)
    out["body_pos_w"][frame] = data.xpos[body_ids]
    out["body_quat_w"][frame] = data.xquat[body_ids]
    for target, body_id in enumerate(body_ids):
      mujoco.mj_jac(model, data, jac_pos, jac_rot, data.xpos[body_id], int(body_id))
      out["body_lin_vel_w"][frame, target] = jac_pos @ data.qvel
      out["body_ang_vel_w"][frame, target] = jac_rot @ data.qvel
  return out
