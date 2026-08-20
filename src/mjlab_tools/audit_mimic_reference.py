"""Audit a retargeted mimic clip against its source BVH at MPC cadence.

The tool deliberately keeps the 50-Hz retargeted NPZ as the tracker input.
It infers left/right nominal contacts, writes them into a *new* enriched NPZ,
and resamples both the retargeted and source-BVH contact schedules on the
continuous MPC grid.  The runtime command then samples the 50-Hz labels at
``mpc_dt``; it must not replace the mimic input by a 14.29-Hz motion clip.

This tool has no MuJoCo dependency.  Its CoM/linear-momentum figure is clearly
labelled as a uniform-mass diagnostic proxy.  Run
``process-reference-centroidal`` with the matched MuJoCo XML before claiming
exact centroidal quantities or angular momentum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


@dataclass
class BvhNode:
  name: str
  parent: int | None
  offset: np.ndarray
  channels: tuple[str, ...]
  channel_start: int


def _smooth_contact_state(candidates: np.ndarray, min_stance: int, min_swing: int) -> np.ndarray:
  """Match the runtime height/speed contact clean-up rule."""
  state = candidates.astype(bool, copy=True)
  for contact_id in range(state.shape[1]):
    for target, maximum, replacement in ((False, min_swing - 1, True), (True, min_stance - 1, False)):
      start = 0
      while start < state.shape[0]:
        end = start + 1
        while end < state.shape[0] and state[end, contact_id] == state[start, contact_id]:
          end += 1
        if state[start, contact_id] == target and end - start <= maximum:
          if start > 0 and end < state.shape[0] and state[start - 1, contact_id] == replacement and state[end, contact_id] == replacement:
            state[start:end, contact_id] = replacement
        start = end
  return state


def _infer_contacts(
  points: np.ndarray, fps: float, vertical_axis: int, height_threshold: float,
  speed_threshold: float, min_stance: int, min_swing: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  height = points[..., vertical_axis]
  relative_height = height - height.min(axis=0, keepdims=True)
  previous = np.concatenate((points[:1], points[:-1]), axis=0)
  following = np.concatenate((points[1:], points[-1:]), axis=0)
  speed = np.linalg.norm(following - previous, axis=-1) * (0.5 * fps)
  candidate = (relative_height <= height_threshold) & (speed <= speed_threshold)
  return _smooth_contact_state(candidate, min_stance, min_swing), relative_height, speed


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
  """Rotate one body-frame vector by [w,x,y,z] quaternions, vectorized in time."""
  q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1.0e-12)
  q_vec = q[..., 1:]
  first = 2.0 * np.cross(q_vec, vector)
  return vector + q[..., :1] * first + np.cross(q_vec, first)


def _rotation(axis: str, angle_rad: np.ndarray) -> np.ndarray:
  cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
  result = np.zeros((len(angle_rad), 3, 3), dtype=np.float64)
  if axis == "X":
    result[:, 0, 0] = 1.0
    result[:, 1, 1], result[:, 1, 2] = cosine, -sine
    result[:, 2, 1], result[:, 2, 2] = sine, cosine
  elif axis == "Y":
    result[:, 1, 1] = 1.0
    result[:, 0, 0], result[:, 0, 2] = cosine, sine
    result[:, 2, 0], result[:, 2, 2] = -sine, cosine
  elif axis == "Z":
    result[:, 2, 2] = 1.0
    result[:, 0, 0], result[:, 0, 1] = cosine, -sine
    result[:, 1, 0], result[:, 1, 1] = sine, cosine
  else:
    raise ValueError(f"Unsupported BVH rotation axis {axis!r}")
  return result


def _read_bvh(path: Path) -> tuple[list[BvhNode], np.ndarray, float]:
  """Read a standard BVH hierarchy and motion table without third-party parsers."""
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  try:
    motion_line = next(index for index, line in enumerate(lines) if line.strip().upper() == "MOTION")
  except StopIteration as exc:
    raise ValueError(f"BVH has no MOTION section: {path}") from exc
  nodes: list[BvhNode] = []
  stack: list[int] = []
  pending_name: str | None = None
  pending_end_site = False
  end_site_depth = 0
  channel_cursor = 0
  index = 0
  while index < motion_line:
    words = lines[index].strip().split()
    index += 1
    if not words:
      continue
    if words[0] in {"ROOT", "JOINT"}:
      pending_name = words[1]
      continue
    if words[0] == "End":
      pending_name = None
      pending_end_site = True
      continue
    if words[0] == "{":
      if pending_end_site:
        pending_end_site = False
        end_site_depth = 1
        continue
      if end_site_depth:
        end_site_depth += 1
        continue
      if pending_name is not None:
        nodes.append(BvhNode(pending_name, stack[-1] if stack else None, np.zeros(3), (), channel_cursor))
        stack.append(len(nodes) - 1)
        pending_name = None
      continue
    if words[0] == "}":
      if end_site_depth:
        end_site_depth -= 1
        continue
      if stack:
        stack.pop()
      continue
    if end_site_depth:
      continue
    if words[0] == "OFFSET" and stack:
      nodes[stack[-1]].offset = np.asarray(words[1:4], dtype=np.float64)
      continue
    if words[0] == "CHANNELS" and stack:
      count = int(words[1])
      channels = tuple(words[2:2 + count])
      nodes[stack[-1]].channels = channels
      nodes[stack[-1]].channel_start = channel_cursor
      channel_cursor += count
  if not nodes or channel_cursor == 0:
    raise ValueError(f"Could not parse BVH hierarchy/channels from {path}")
  frames = int(lines[motion_line + 1].split(":", 1)[1].strip())
  frame_time = float(lines[motion_line + 2].split(":", 1)[1].strip())
  values = np.loadtxt(lines[motion_line + 3:], dtype=np.float64)
  values = np.atleast_2d(values)
  if values.shape != (frames, channel_cursor):
    raise ValueError(f"BVH motion shape {values.shape}, expected {(frames, channel_cursor)}")
  return nodes, values, 1.0 / frame_time


def _bvh_joint_positions(nodes: list[BvhNode], values: np.ndarray, names: tuple[str, str], scale: float) -> np.ndarray:
  lookup = {node.name: index for index, node in enumerate(nodes)}
  missing = [name for name in names if name not in lookup]
  if missing:
    raise ValueError(f"BVH does not contain requested foot joints: {missing}; available={list(lookup)}")
  frames = len(values)
  positions = np.zeros((frames, len(nodes), 3), dtype=np.float64)
  rotations = np.zeros((frames, len(nodes), 3, 3), dtype=np.float64)
  identity = np.eye(3, dtype=np.float64)[None].repeat(frames, axis=0)
  for index, node in enumerate(nodes):
    local = identity.copy()
    translation = np.zeros((frames, 3), dtype=np.float64)
    channels = values[:, node.channel_start:node.channel_start + len(node.channels)]
    for channel_index, channel in enumerate(node.channels):
      if channel.endswith("position"):
        axis = "XYZ".index(channel[0].upper())
        translation[:, axis] = channels[:, channel_index]
      elif channel.endswith("rotation"):
        local = local @ _rotation(channel[0].upper(), np.deg2rad(channels[:, channel_index]))
    if node.parent is None:
      positions[:, index] = (node.offset + translation) * scale
      rotations[:, index] = local
    else:
      parent = node.parent
      positions[:, index] = positions[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], node.offset * scale)
      rotations[:, index] = rotations[:, parent] @ local
  return positions[:, [lookup[name] for name in names]]


def _sample_linear(values: np.ndarray, source_fps: float, times_s: np.ndarray) -> np.ndarray:
  progress = np.clip(times_s * source_fps, 0.0, values.shape[0] - 1)
  first = np.floor(progress).astype(np.int64)
  second = np.minimum(first + 1, values.shape[0] - 1)
  alpha = (progress - first).reshape((-1,) + (1,) * (values.ndim - 1))
  return values[first] + alpha * (values[second] - values[first])


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
  fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
  fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
  plt.close(fig)


def _plot_contacts(
  output_dir: Path, retarget_time: np.ndarray, retarget_height: np.ndarray, retarget_speed: np.ndarray,
  source_time: np.ndarray, source_height: np.ndarray, source_speed: np.ndarray,
  mpc_time: np.ndarray, retarget_mpc: np.ndarray, source_mpc: np.ndarray,
) -> None:
  fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False, layout="constrained")
  for foot, color in enumerate(("#0072B2", "#D55E00")):
    label = ("left", "right")[foot]
    axes[0].plot(retarget_time, retarget_height[:, foot], color=color, linewidth=0.8, label=f"JC01 {label}")
    axes[0].plot(source_time, source_height[:, foot], color=color, linewidth=0.8, linestyle="--", alpha=0.75, label=f"BVH {label}")
    axes[1].plot(retarget_time, retarget_speed[:, foot], color=color, linewidth=0.8, label=f"JC01 {label}")
    axes[1].plot(source_time, source_speed[:, foot], color=color, linewidth=0.8, linestyle="--", alpha=0.75, label=f"BVH {label}")
  axes[0].set_title("Contact inference inputs: retargeted JC01 (solid) vs source BVH (dashed)")
  axes[0].set_ylabel("Height above each-foot minimum [m]")
  axes[1].set_ylabel("Foot-point speed [m/s]")
  for axis in axes[:2]:
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    axis.set_xlabel("Time [s]")
  schedule = np.vstack((retarget_mpc[:, 0], retarget_mpc[:, 1], source_mpc[:, 0], source_mpc[:, 1]))
  axes[2].imshow(
    schedule, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
    cmap=ListedColormap(("#F2F2F2", "#0072B2")),
    extent=(mpc_time[0], mpc_time[-1], 3.5, -0.5),
  )
  axes[2].set_yticks(range(4), ("JC01 left", "JC01 right", "BVH left", "BVH right"))
  axes[2].set_xlabel("Time [s] at MPC grid (Δt = 0.07 s)")
  axes[2].set_title("Nominal stance schedule: blue = stance")
  _save(fig, output_dir, "contact_sequence_comparison")


def _plot_centroidal_proxy(output_dir: Path, time_s: np.ndarray, com: np.ndarray, linear_momentum: np.ndarray) -> None:
  fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
  for axis, values, ylabel in zip(axes, (com, linear_momentum), ("Uniform-body CoM proxy [m]", "Uniform-mass linear-momentum proxy [kg m/s]")):
    for component, color, label in zip(range(3), ("#0072B2", "#D55E00", "#009E73"), ("x", "y", "z")):
      axis.plot(time_s, values[:, component], color=color, linewidth=0.8, label=label)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=8, frameon=False)
  axes[0].set_title("Diagnostic only: uniform-mass proxy; exact centroidal quantities require the matched MuJoCo XML")
  axes[-1].set_xlabel("Time [s]")
  _save(fig, output_dir, "retargeted_centroidal_proxy_not_for_mpc")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("retargeted_npz", type=Path)
  parser.add_argument("source_bvh", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--mpc-dt", type=float, default=0.07)
  parser.add_argument(
    "--contact-source", choices=("source-bvh", "retargeted"), default="source-bvh",
    help="Contact labels exported as reference_contact_state; source-bvh is the recommended semantic schedule.",
  )
  parser.add_argument(
    "--source-time-offset-s", type=float, default=0.0,
    help="Source-BVH time added before resampling. Use only after an explicit phase-alignment audit.",
  )
  parser.add_argument("--left-body", default="left_ankle_roll")
  parser.add_argument("--right-body", default="right_ankle_roll")
  parser.add_argument("--sole-offset-z", type=float, default=-0.04)
  parser.add_argument("--source-left-foot", default="LeftFoot")
  parser.add_argument("--source-right-foot", default="RightFoot")
  parser.add_argument("--bvh-unit-scale", type=float, default=0.01)
  parser.add_argument("--height-threshold", type=float, default=0.03)
  parser.add_argument("--speed-threshold", type=float, default=0.35)
  parser.add_argument("--min-stance-frames", type=int, default=3)
  parser.add_argument("--min-swing-frames", type=int, default=2)
  parser.add_argument("--nominal-total-mass", type=float, default=57.00294)
  args = parser.parse_args()
  if args.mpc_dt <= 0.0 or args.bvh_unit_scale <= 0.0 or args.nominal_total_mass <= 0.0:
    raise ValueError("mpc dt, BVH scale and nominal mass must be positive")
  if not args.retargeted_npz.is_file() or not args.source_bvh.is_file():
    raise FileNotFoundError("retargeted NPZ and source BVH must exist")
  args.output_dir.mkdir(parents=True, exist_ok=True)

  with np.load(args.retargeted_npz, allow_pickle=False) as source:
    arrays = {key: source[key] for key in source.files}
  required = {"fps", "body_names", "body_pos_w", "body_quat_w", "body_lin_vel_w"}
  missing = required.difference(arrays)
  if missing:
    raise ValueError(f"Retargeted clip is missing {sorted(missing)}")
  retarget_fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
  body_names = [str(name) for name in arrays["body_names"].tolist()]
  lookup = {name: index for index, name in enumerate(body_names)}
  requested = (args.left_body, args.right_body)
  missing = [name for name in requested if name not in lookup]
  if missing:
    raise ValueError(f"Retargeted clip lacks foot bodies {missing}; available={body_names}")
  foot_ids = [lookup[name] for name in requested]
  contact_pos = arrays["body_pos_w"][:, foot_ids].astype(np.float64)
  offset = np.zeros_like(contact_pos)
  offset[..., 2] = args.sole_offset_z
  contact_pos += _quat_rotate_wxyz(arrays["body_quat_w"][:, foot_ids].astype(np.float64), offset)
  retarget_state, retarget_height, retarget_speed = _infer_contacts(
    contact_pos, retarget_fps, 2, args.height_threshold, args.speed_threshold,
    args.min_stance_frames, args.min_swing_frames,
  )

  bvh_nodes, bvh_values, bvh_fps = _read_bvh(args.source_bvh)
  bvh_pos = _bvh_joint_positions(
    bvh_nodes, bvh_values, (args.source_left_foot, args.source_right_foot), args.bvh_unit_scale,
  )
  source_state, source_height, source_speed = _infer_contacts(
    bvh_pos, bvh_fps, 1, args.height_threshold, args.speed_threshold,
    args.min_stance_frames, args.min_swing_frames,
  )
  source_start = args.source_time_offset_s
  source_duration = (len(bvh_pos) - 1) / bvh_fps - source_start
  duration = min((len(contact_pos) - 1) / retarget_fps, source_duration)
  if duration <= 0.0:
    raise ValueError("source-time-offset-s leaves no overlapping source-BVH trajectory")
  mpc_time = np.arange(0.0, duration + 1.0e-9, args.mpc_dt)
  retarget_mpc = _sample_linear(retarget_state.astype(np.float64), retarget_fps, mpc_time) >= 0.5
  source_mpc = _sample_linear(source_state.astype(np.float64), bvh_fps, mpc_time + source_start) >= 0.5
  # Preserve both 50-Hz sources. The selected array is exported explicitly as
  # the MPC nominal schedule; centroidal quantities remain retargeted-only.
  source_at_retarget_fps = _sample_linear(
    source_state.astype(np.float64), bvh_fps,
    np.arange(len(contact_pos)) / retarget_fps + source_start,
  ) >= 0.5

  retarget_time = np.arange(len(contact_pos)) / retarget_fps
  source_time = np.arange(len(bvh_pos)) / bvh_fps
  _plot_contacts(
    args.output_dir, retarget_time, retarget_height, retarget_speed, source_time, source_height,
    source_speed, mpc_time, retarget_mpc, source_mpc,
  )
  # This proxy is diagnostic only; exactly calculated c/l/k are produced by
  # process_reference_centroidal once the XML asset is supplied.
  proxy_com = arrays["body_pos_w"].astype(np.float64).mean(axis=1)
  proxy_linear_momentum = args.nominal_total_mass * arrays["body_lin_vel_w"].astype(np.float64).mean(axis=1)
  _plot_centroidal_proxy(args.output_dir, retarget_time, proxy_com, proxy_linear_momentum)

  arrays["retargeted_contact_state"] = retarget_state.astype(np.uint8)
  arrays["source_bvh_contact_state"] = source_at_retarget_fps.astype(np.uint8)
  arrays["reference_contact_state"] = (
    source_at_retarget_fps if args.contact_source == "source-bvh" else retarget_state
  ).astype(np.uint8)
  arrays["reference_contact_source"] = np.asarray(args.contact_source)
  arrays["source_bvh_time_offset_s"] = np.asarray(args.source_time_offset_s, dtype=np.float32)
  arrays["reference_contact_labels"] = np.asarray(("left_sole_center", "right_sole_center"))
  arrays["reference_contact_body_names"] = np.asarray(requested)
  arrays["reference_contact_point_offsets_b"] = np.asarray(((0.0, 0.0, args.sole_offset_z),) * 2, dtype=np.float32)
  arrays["reference_contact_height_threshold"] = np.asarray(args.height_threshold, dtype=np.float32)
  arrays["reference_contact_speed_threshold"] = np.asarray(args.speed_threshold, dtype=np.float32)
  arrays["reference_contact_min_stance_frames"] = np.asarray(args.min_stance_frames, dtype=np.int32)
  arrays["reference_contact_min_swing_frames"] = np.asarray(args.min_swing_frames, dtype=np.int32)
  arrays["mpc_dt"] = np.asarray(args.mpc_dt, dtype=np.float32)
  arrays["mpc_time_s"] = mpc_time.astype(np.float32)
  arrays["mpc_retargeted_contact_state"] = retarget_mpc.astype(np.uint8)
  arrays["mpc_source_bvh_contact_state"] = source_mpc.astype(np.uint8)
  arrays["mpc_reference_frame_progress"] = (mpc_time * retarget_fps).astype(np.float32)
  arrays["contact_audit_source_bvh"] = np.asarray(str(args.source_bvh.resolve()))
  enriched = args.output_dir / f"{args.retargeted_npz.stem}_with_reference_contacts.npz"
  np.savez_compressed(enriched, **arrays)
  agreement = (retarget_mpc == source_mpc)
  summary = {
    "retargeted_npz": str(args.retargeted_npz.resolve()),
    "source_bvh": str(args.source_bvh.resolve()),
    "enriched_mimic_npz": str(enriched.resolve()),
    "retarget_fps": retarget_fps,
    "source_bvh_fps": bvh_fps,
    "reference_contact_source": args.contact_source,
    "source_bvh_time_offset_s": args.source_time_offset_s,
    "mpc_dt_s": args.mpc_dt,
    "mpc_fps": 1.0 / args.mpc_dt,
    "duration_compared_s": duration,
    "mpc_nodes": int(len(mpc_time)),
    "retarget_stance_fraction": retarget_mpc.mean(axis=0).tolist(),
    "source_stance_fraction": source_mpc.mean(axis=0).tolist(),
    "per_foot_schedule_agreement": agreement.mean(axis=0).tolist(),
    "both_feet_schedule_agreement": float(agreement.all(axis=1).mean()),
    "contact_thresholds": {
      "height_m": args.height_threshold, "speed_mps": args.speed_threshold,
      "min_stance_frames_at_50hz": args.min_stance_frames,
      "min_swing_frames_at_50hz": args.min_swing_frames,
    },
    "centroidal_warning": "The plotted CoM/linear-momentum values are uniform-mass diagnostic proxies. Exact c/l/k require process-reference-centroidal with the matched MuJoCo XML.",
    "training_decision": (
      "reference_contact_state is the selected nominal MPC schedule. It is source-BVH semantic contact "
      "when contact-source=source-bvh; validate transition alignment and report its disagreement with retargeted foot kinematics."
    ),
  }
  (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
