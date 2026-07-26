"""Reconstruct JC01 centroidal quantities from BeyondMimic sim2sim MCAP logs.

Robot root/joint states in each policy-running frame are replayed through the
same MuJoCo model used for the reference clip.  MuJoCo forward kinematics and
spatial velocities then provide every link pose and velocity; this makes the
mass, inertia and momentum convention identical to the reference processor.
MCAP foot-contact labels are retained directly, rather than inferred from
kinematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .mcap_policy_frames import iter_policy_running_frames
from .process_reference_centroidal import _parse_body_map, _parse_contact
from .reference_centroidal import compute_reference_centroidal


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
  return quat_xyzw[[3, 0, 1, 2]]


def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
  w, x, y, z = quat_wxyz
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_rotation(yaw: float) -> np.ndarray:
  cosine, sine = np.cos(yaw), np.sin(yaw)
  return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _read_frames(mcap: Path, topic: str, expected_joints: int) -> dict[str, np.ndarray]:
  log_time_ns: list[int] = []
  frame_index: list[int] = []
  joint_q: list[np.ndarray] = []
  joint_dq: list[np.ndarray] = []
  root_pos_w: list[np.ndarray] = []
  root_quat_wxyz: list[np.ndarray] = []
  root_lin_vel_b: list[np.ndarray] = []
  root_ang_vel_b: list[np.ndarray] = []
  contact_state: list[np.ndarray] = []
  contact_force_norm: list[np.ndarray] = []
  phase_t_global: list[float] = []
  for timestamp_ns, frame in iter_policy_running_frames(mcap, topic):
    tick = frame.get("tick", {})
    # Runtime source snapshots and contact labels live beside ``tick`` in the
    # processed policy-frame message; only policy bookkeeping/joint arrays are
    # nested in ``tick``.
    source = frame.get("sources", {}).get("base_state", {}).get("sample", {}).get("values", {})
    contacts = frame.get("foot_contact", {})
    q = np.asarray(tick.get("joint_q"), dtype=np.float64)
    dq = np.asarray(tick.get("joint_dq"), dtype=np.float64)
    if q.shape != (expected_joints,) or dq.shape != (expected_joints,):
      raise ValueError(
        f"Frame {frame.get('frame_index')} has joint shapes q={q.shape}, dq={dq.shape}; "
        f"expected ({expected_joints},)"
      )
    required_base = ("pos_w", "quat_xyzw", "lin_vel", "ang_vel")
    if any(key not in source for key in required_base):
      raise ValueError(f"Frame {frame.get('frame_index')} is missing base-state fields")
    left = contacts.get("left")
    right = contacts.get("right")
    if left is None or right is None:
      raise ValueError(f"Frame {frame.get('frame_index')} is missing left/right foot-contact labels")
    log_time_ns.append(int(timestamp_ns))
    frame_index.append(int(frame.get("frame_index", len(frame_index))))
    joint_q.append(q)
    joint_dq.append(dq)
    root_pos_w.append(np.asarray(source["pos_w"], dtype=np.float64))
    root_quat_wxyz.append(_xyzw_to_wxyz(np.asarray(source["quat_xyzw"], dtype=np.float64)))
    root_lin_vel_b.append(np.asarray(source["lin_vel"], dtype=np.float64))
    root_ang_vel_b.append(np.asarray(source["ang_vel"], dtype=np.float64))
    contact_state.append(np.asarray((bool(left["in_contact"]), bool(right["in_contact"]))))
    contact_force_norm.append(np.asarray((left["force_norm_n"], right["force_norm_n"]), dtype=np.float64))
    phase_t_global.append(float(tick.get("phase_t_global", np.nan)))
  if len(log_time_ns) < 2:
    raise ValueError("At least two policy-running frames are required")
  time_s = (np.asarray(log_time_ns, dtype=np.int64) - log_time_ns[0]) * 1.0e-9
  if not np.all(np.diff(time_s) > 0.0):
    raise ValueError("MCAP policy-running timestamps must be strictly increasing")
  return {
    "log_time_ns": np.asarray(log_time_ns, dtype=np.int64),
    "time_s": time_s,
    "frame_index": np.asarray(frame_index, dtype=np.int64),
    "joint_q": np.stack(joint_q),
    "joint_dq": np.stack(joint_dq),
    "root_pos_w": np.stack(root_pos_w),
    "root_quat_wxyz": np.stack(root_quat_wxyz),
    "root_lin_vel_b": np.stack(root_lin_vel_b),
    "root_ang_vel_b": np.stack(root_ang_vel_b),
    "contact_state": np.stack(contact_state),
    "contact_force_norm_n": np.stack(contact_force_norm),
    "phase_t_global": np.asarray(phase_t_global),
  }


def _reconstruct_bodies(
  model: mujoco.MjModel,
  model_ids: list[int],
  joint_names: list[str],
  frames: dict[str, np.ndarray],
  velocity_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Replay state records through MuJoCo and collect selected body kinematics."""
  joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
  if any(joint_id < 0 for joint_id in joint_ids):
    missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
    raise ValueError(f"MCAP/reference joint names are absent from XML: {missing}")
  root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_freejoint")
  if root_joint_id < 0 or model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("XML must contain a free joint named 'root_freejoint'")
  root_qpos = model.jnt_qposadr[root_joint_id]
  root_dof = model.jnt_dofadr[root_joint_id]
  joint_qpos = model.jnt_qposadr[joint_ids]
  joint_dof = model.jnt_dofadr[joint_ids]
  frames_count = frames["joint_q"].shape[0]
  bodies_count = len(model_ids)
  pos = np.empty((frames_count, bodies_count, 3), dtype=np.float32)
  quat = np.empty((frames_count, bodies_count, 4), dtype=np.float32)
  lin_vel = np.empty((frames_count, bodies_count, 3), dtype=np.float32)
  ang_vel = np.empty((frames_count, bodies_count, 3), dtype=np.float32)
  data = mujoco.MjData(model)
  jac_pos = np.empty((3, model.nv), dtype=np.float64)
  jac_rot = np.empty((3, model.nv), dtype=np.float64)
  for frame_id in range(frames_count):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    root_quat = frames["root_quat_wxyz"][frame_id]
    data.qpos[root_qpos:root_qpos + 3] = frames["root_pos_w"][frame_id]
    data.qpos[root_qpos + 3:root_qpos + 7] = root_quat
    data.qpos[joint_qpos] = frames["joint_q"][frame_id]
    if velocity_frame == "root":
      root_rotation = np.empty((9,), dtype=np.float64)
      mujoco.mju_quat2Mat(root_rotation, root_quat)
      root_rotation = root_rotation.reshape(3, 3)
      root_linear = root_rotation @ frames["root_lin_vel_b"][frame_id]
      root_angular = root_rotation @ frames["root_ang_vel_b"][frame_id]
    else:
      root_linear = frames["root_lin_vel_b"][frame_id]
      root_angular = frames["root_ang_vel_b"][frame_id]
    data.qvel[root_dof:root_dof + 3] = root_linear
    data.qvel[root_dof + 3:root_dof + 6] = root_angular
    data.qvel[joint_dof] = frames["joint_dq"][frame_id]
    mujoco.mj_forward(model, data)
    pos[frame_id] = data.xpos[model_ids]
    quat[frame_id] = data.xquat[model_ids]
    # ``data.cvel`` is centered on MuJoCo's body inertial frame.  The
    # centroidal helper below instead takes velocity at ``xpos`` (the body
    # frame origin), because it subsequently shifts that velocity to the
    # inertial CoM using ``omega x R*ipos``.  Jacobians evaluated at xpos give
    # exactly that convention and retain the recorded generalized velocity.
    for output_id, body_id in enumerate(model_ids):
      mujoco.mj_jac(model, data, jac_pos, jac_rot, data.xpos[body_id], body_id)
      lin_vel[frame_id, output_id] = jac_pos @ data.qvel
      ang_vel[frame_id, output_id] = jac_rot @ data.qvel
  return pos, quat, lin_vel, ang_vel


def _align_to_reference(
  *,
  reference_motion: Path,
  reference_root_name: str,
  raw_root_pos_w: np.ndarray,
  raw_root_quat_wxyz: np.ndarray,
  trajectory: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float | list[float]]]:
  reference = np.load(reference_motion, allow_pickle=False)
  names = [str(name) for name in reference["body_names"].tolist()]
  if reference_root_name not in names:
    raise ValueError(f"Reference root body {reference_root_name!r} is absent")
  root_id = names.index(reference_root_name)
  reference_root_pos = np.asarray(reference["body_pos_w"][0, root_id], dtype=np.float64)
  reference_root_quat = np.asarray(reference["body_quat_w"][0, root_id], dtype=np.float64)
  yaw_reference = _yaw_from_wxyz(reference_root_quat)
  yaw_simulation = _yaw_from_wxyz(raw_root_quat_wxyz[0])
  yaw_offset = yaw_reference - yaw_simulation
  rotation = _yaw_rotation(yaw_offset)
  aligned: dict[str, np.ndarray] = {}
  for key in ("com_pos_w", "contact_pos_w"):
    values = trajectory[key]
    aligned[key] = np.einsum("ij,...j->...i", rotation, values - raw_root_pos_w[0]) + reference_root_pos
  for key in ("com_vel_w", "linear_momentum_w", "angular_momentum_w"):
    aligned[key] = np.einsum("ij,...j->...i", rotation, trajectory[key])
  aligned["contact_pos_rel_com_w"] = aligned["contact_pos_w"] - aligned["com_pos_w"][:, None, :]
  metadata: dict[str, float | list[float]] = {
    "yaw_offset_rad": float(yaw_offset),
    "reference_root_initial_pos_w_m": reference_root_pos.tolist(),
    "simulation_root_initial_pos_w_m": raw_root_pos_w[0].tolist(),
  }
  return aligned, metadata


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mcap", type=Path, help="BeyondMimic policy-running-frame MCAP")
  parser.add_argument("reference_motion", type=Path, help="Matched BeyondMimic reference NPZ")
  parser.add_argument("xml", type=Path, help="Matched MuJoCo XML scene")
  parser.add_argument("output", type=Path, help="Output processed sim2sim NPZ")
  parser.add_argument("--urdf", type=Path, default=None, help="Optional provenance-only URDF")
  parser.add_argument(
    "--topic", default="runtime/processed/policy_running_frame",
    help="MCAP topic containing policy-running JSON frames",
  )
  parser.add_argument("--body-map", action="append", default=[], metavar="REF=MUJOCO")
  parser.add_argument(
    "--contact", action="append", default=[], metavar="LABEL=REF_BODY:DX,DY,DZ",
    help="Contact label and point used for kinematic position reporting. Order must be left, then right.",
  )
  parser.add_argument(
    "--base-velocity-frame", choices=("root", "world"), default="root",
    help="Frame of MCAP base lin_vel/ang_vel. The recorded JC01 source tags them as root-local.",
  )
  parser.add_argument(
    "--contact-force-threshold-n", type=float, default=100.0,
    help="Recompute sim contact as foot-force norm >= this value [N], rather than using logged in_contact.",
  )
  parser.add_argument("--reference-root-body", default="Robotbase")
  args = parser.parse_args()
  if not args.mcap.is_file() or not args.reference_motion.is_file() or not args.xml.is_file():
    raise FileNotFoundError("MCAP, reference NPZ and XML must all exist")
  if args.urdf is not None and not args.urdf.is_file():
    raise FileNotFoundError(args.urdf)
  if args.contact_force_threshold_n < 0.0:
    raise ValueError("--contact-force-threshold-n must be non-negative")

  reference = np.load(args.reference_motion, allow_pickle=False)
  required = {"body_names", "joint_names", "source_joint_names"}
  if missing := required.difference(reference.files):
    raise ValueError(f"Reference motion is missing fields: {sorted(missing)}")
  reference_body_names = [str(name) for name in reference["body_names"].tolist()]
  # MCAP ``joint_q``/``joint_dq`` are robot-state arrays in the deployment
  # (MuJoCo/source) ordering, not the reordered policy-reference layout.  The
  # reference records both layouts precisely for this purpose.
  joint_names = [str(name) for name in reference["source_joint_names"].tolist()]
  model = mujoco.MjModel.from_xml_path(str(args.xml))
  body_map = _parse_body_map(args.body_map)
  model_body_names = [body_map.get(name, name) for name in reference_body_names]
  model_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in model_body_names]
  if any(body_id < 0 for body_id in model_ids):
    missing = [name for name, body_id in zip(model_body_names, model_ids) if body_id < 0]
    raise ValueError(f"Reference bodies absent from XML: {missing}")

  contact_specs = [_parse_contact(value) for value in args.contact]
  contact_labels = [item[0] for item in contact_specs]
  contact_names = [item[1] for item in contact_specs]
  if contact_labels != ["left_sole_center", "right_sole_center"]:
    raise ValueError("This MCAP logger has left/right labels; contacts must be ordered left_sole_center, right_sole_center")
  body_index = {name: index for index, name in enumerate(reference_body_names)}
  if any(name not in body_index for name in contact_names):
    raise ValueError("A contact body is absent from reference body_names")
  contact_indices = torch.tensor([body_index[name] for name in contact_names], dtype=torch.long)
  contact_offsets = torch.tensor([item[2] for item in contact_specs], dtype=torch.float32)

  frames = _read_frames(args.mcap, args.topic, len(joint_names))
  # The logger's boolean in_contact was generated with its own low (20 N)
  # threshold.  Keep it for provenance, but use a configurable force threshold
  # for the analysis contact state so the comparison is not tied to that logger
  # implementation detail.
  force_threshold_contact = frames["contact_force_norm_n"] >= args.contact_force_threshold_n
  body_pos, body_quat, body_lin_vel, body_ang_vel = _reconstruct_bodies(
    model, model_ids, joint_names, frames, args.base_velocity_frame,
  )
  centroidal = compute_reference_centroidal(
    body_pos_w=torch.as_tensor(body_pos),
    body_quat_w=torch.as_tensor(body_quat),
    body_lin_vel_w=torch.as_tensor(body_lin_vel),
    body_ang_vel_w=torch.as_tensor(body_ang_vel),
    body_mass=torch.as_tensor(model.body_mass[model_ids], dtype=torch.float32),
    body_com_offset_b=torch.as_tensor(model.body_ipos[model_ids], dtype=torch.float32),
    body_inertia_diag=torch.as_tensor(model.body_inertia[model_ids], dtype=torch.float32),
    body_inertial_quat_b=torch.as_tensor(model.body_iquat[model_ids], dtype=torch.float32),
    contact_body_indices=contact_indices,
    contact_point_offset_b=contact_offsets,
  )
  raw = {
    "com_pos_w": centroidal.com_pos_w.numpy(),
    "com_vel_w": centroidal.com_vel_w.numpy(),
    "linear_momentum_w": centroidal.linear_momentum_w.numpy(),
    "angular_momentum_w": centroidal.angular_momentum_w.numpy(),
    "contact_pos_w": centroidal.contact_pos_w.numpy(),
    "contact_pos_rel_com_w": centroidal.contact_pos_rel_com_w.numpy(),
  }
  aligned, alignment = _align_to_reference(
    reference_motion=args.reference_motion,
    reference_root_name=args.reference_root_body,
    raw_root_pos_w=frames["root_pos_w"],
    raw_root_quat_wxyz=frames["root_quat_wxyz"],
    trajectory=raw,
  )
  duration_s = float(frames["time_s"][-1])
  inferred_fps = float((len(frames["time_s"]) - 1) / duration_s)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    fps=np.asarray(inferred_fps, dtype=np.float32),
    time_s=frames["time_s"].astype(np.float64),
    motion_progress=(frames["time_s"] / duration_s).astype(np.float64),
    frame_index=frames["frame_index"],
    mcap_log_time_ns=frames["log_time_ns"],
    phase_t_global=frames["phase_t_global"],
    com_pos_w=aligned["com_pos_w"].astype(np.float32),
    com_vel_w=aligned["com_vel_w"].astype(np.float32),
    linear_momentum_w=aligned["linear_momentum_w"].astype(np.float32),
    angular_momentum_w=aligned["angular_momentum_w"].astype(np.float32),
    contact_pos_w=aligned["contact_pos_w"].astype(np.float32),
    contact_pos_rel_com_w=aligned["contact_pos_rel_com_w"].astype(np.float32),
    raw_com_pos_w=raw["com_pos_w"].astype(np.float32),
    raw_contact_pos_w=raw["contact_pos_w"].astype(np.float32),
    contact_state=force_threshold_contact.astype(np.uint8),
    logged_contact_state=frames["contact_state"].astype(np.uint8),
    contact_force_norm_n=frames["contact_force_norm_n"].astype(np.float32),
    contact_force_threshold_n=np.asarray(args.contact_force_threshold_n, dtype=np.float32),
    contact_state_source=np.asarray("foot_force_norm_threshold"),
    contact_labels=np.asarray(contact_labels),
    centroidal_reference_body_names=np.asarray(reference_body_names),
    centroidal_model_body_names=np.asarray(model_body_names),
    contact_reference_body_names=np.asarray(contact_names),
    contact_point_offsets_b=contact_offsets.numpy(),
    total_mass=np.asarray(float(model.body_mass[model_ids].sum()), dtype=np.float32),
    source_mcap=np.asarray(str(args.mcap.resolve())),
    source_reference_motion=np.asarray(str(args.reference_motion.resolve())),
    source_xml=np.asarray(str(args.xml.resolve())),
    source_urdf=np.asarray("" if args.urdf is None else str(args.urdf.resolve())),
    mcap_topic=np.asarray(args.topic),
    base_velocity_frame=np.asarray(args.base_velocity_frame),
    yaw_alignment_rad=np.asarray(alignment["yaw_offset_rad"], dtype=np.float64),
  )
  summary = {
    "mcap": str(args.mcap.resolve()), "reference_motion": str(args.reference_motion.resolve()),
    "xml": str(args.xml.resolve()), "output": str(args.output.resolve()),
    "frames": int(len(frames["time_s"])), "duration_s": duration_s, "effective_fps": inferred_fps,
    "total_mass_kg": float(model.body_mass[model_ids].sum()), "mcap_topic": args.topic,
    "base_velocity_frame": args.base_velocity_frame, "alignment": alignment,
    "contact_state_source": "foot_force_norm_threshold",
    "contact_force_threshold_n": args.contact_force_threshold_n,
    "contacts": [{"label": label, "reference_body": body, "offset_b_m": list(offset)} for label, body, offset in contact_specs],
  }
  args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
