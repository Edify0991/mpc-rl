"""Offline centroidal/contact preprocessing for a named-reference NPZ clip.

The script uses the MuJoCo XML as the numerical source of truth for body mass,
inertial offsets, inertia and inertial-frame orientation.  An optional URDF is
recorded as provenance only: parsing its inertias independently can disagree
with the compiled XML due to conversion transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .reference_centroidal import compute_reference_centroidal


def _parse_body_map(values: list[str]) -> dict[str, str]:
  mapping: dict[str, str] = {}
  for value in values:
    if "=" not in value:
      raise ValueError(f"Invalid --body-map {value!r}; expected REFERENCE_BODY=MUJOCO_BODY")
    reference_name, model_name = value.split("=", 1)
    if not reference_name or not model_name or reference_name in mapping:
      raise ValueError(f"Invalid or duplicate body map {value!r}")
    mapping[reference_name] = model_name
  return mapping


def _parse_contact(value: str) -> tuple[str, str, tuple[float, float, float]]:
  """Parse ``LABEL=REFERENCE_BODY:dx,dy,dz``."""
  if "=" not in value or ":" not in value:
    raise ValueError(
      f"Invalid --contact {value!r}; expected LABEL=REFERENCE_BODY:dx,dy,dz"
    )
  label, body_and_offset = value.split("=", 1)
  body_name, offset_text = body_and_offset.split(":", 1)
  try:
    offset = tuple(float(x) for x in offset_text.split(","))
  except ValueError as exc:
    raise ValueError(f"Invalid contact offset in {value!r}") from exc
  if not label or not body_name or len(offset) != 3:
    raise ValueError(f"Invalid --contact {value!r}")
  return label, body_name, offset  # type: ignore[return-value]


def _smooth_contact_state(candidates: np.ndarray, min_stance: int, min_swing: int) -> np.ndarray:
  """Fill short swing gaps, then reject short isolated stance runs."""
  if min_stance < 1 or min_swing < 1:
    raise ValueError("minimum stance/swing durations must be positive")
  state = candidates.astype(bool, copy=True)
  frames, contacts = state.shape
  for contact_id in range(contacts):
    for target, max_length, replacement in ((False, min_swing - 1, True), (True, min_stance - 1, False)):
      if max_length <= 0:
        continue
      start = 0
      while start < frames:
        value = bool(state[start, contact_id])
        end = start + 1
        while end < frames and bool(state[end, contact_id]) == value:
          end += 1
        if value == target and end - start <= max_length:
          left_matches = start > 0 and bool(state[start - 1, contact_id]) == replacement
          right_matches = end < frames and bool(state[end, contact_id]) == replacement
          if left_matches and right_matches:
            state[start:end, contact_id] = replacement
        start = end
  return state


def _infer_contact_state(
  contact_pos_w: np.ndarray,
  fps: float,
  height_threshold: float,
  speed_threshold: float,
  min_stance: int,
  min_swing: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return smoothed contact state, absolute height and central-difference speed."""
  height = contact_pos_w[..., 2]
  floor_height = height.min(axis=0, keepdims=True)
  previous = np.concatenate((contact_pos_w[:1], contact_pos_w[:-1]), axis=0)
  following = np.concatenate((contact_pos_w[1:], contact_pos_w[-1:]), axis=0)
  speed = np.linalg.norm(following - previous, axis=-1) * (0.5 * fps)
  candidate = ((height - floor_height) <= height_threshold) & (speed <= speed_threshold)
  return _smooth_contact_state(candidate, min_stance, min_swing), height, speed


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion", type=Path, help="BeyondMimic-format reference NPZ")
  parser.add_argument("xml", type=Path, help="Matched MuJoCo XML scene")
  parser.add_argument("output", type=Path, help="Output processed NPZ")
  parser.add_argument("--urdf", type=Path, default=None, help="Optional source URDF, saved as provenance")
  parser.add_argument("--body-map", action="append", default=[], metavar="REF=MUJOCO")
  parser.add_argument(
    "--contact", action="append", default=[], metavar="LABEL=REF_BODY:DX,DY,DZ",
    help="Candidate contact point. Repeat to add multiple points.",
  )
  parser.add_argument("--height-threshold", type=float, default=0.03)
  parser.add_argument("--speed-threshold", type=float, default=0.35)
  parser.add_argument("--min-stance-frames", type=int, default=3)
  parser.add_argument("--min-swing-frames", type=int, default=2)
  args = parser.parse_args()

  if args.urdf is not None and not args.urdf.is_file():
    raise FileNotFoundError(f"URDF does not exist: {args.urdf}")
  if not args.motion.is_file() or not args.xml.is_file():
    raise FileNotFoundError("motion NPZ and XML must both exist")
  if args.height_threshold < 0.0 or args.speed_threshold < 0.0:
    raise ValueError("contact thresholds must be non-negative")

  data = np.load(args.motion, allow_pickle=False)
  required = {"fps", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_names"}
  missing = required.difference(data.files)
  if missing:
    raise ValueError(f"Reference is missing fields: {sorted(missing)}")
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  if fps <= 0.0:
    raise ValueError(f"Reference fps must be positive, got {fps}")
  velocity_point = str(np.asarray(data.get("body_linear_velocity_point", "inertial_com")).reshape(-1)[0])
  if velocity_point not in {"inertial_com", "link_origin"}:
    raise ValueError(f"Unsupported body_linear_velocity_point={velocity_point!r}")

  reference_names = [str(name) for name in data["body_names"].tolist()]
  if len(set(reference_names)) != len(reference_names):
    raise ValueError("Reference contains duplicate body names")
  body_map = _parse_body_map(args.body_map)
  model = mujoco.MjModel.from_xml_path(str(args.xml))
  model_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(model.nbody)
  ]
  resolved_model_names = [body_map.get(name, name) for name in reference_names]
  unresolved = [
    reference_name for reference_name, model_name in zip(reference_names, resolved_model_names, strict=True)
    if model_name not in model_names
  ]
  if unresolved:
    raise ValueError(f"Reference bodies absent from XML after body mapping: {unresolved}")
  model_ids = [model_names.index(name) for name in resolved_model_names]

  contact_specs = [_parse_contact(value) for value in args.contact]
  labels = [item[0] for item in contact_specs]
  contact_reference_names = [item[1] for item in contact_specs]
  if len(set(labels)) != len(labels):
    raise ValueError("Contact labels must be unique")
  reference_index = {name: index for index, name in enumerate(reference_names)}
  missing_contact = [name for name in contact_reference_names if name not in reference_index]
  if missing_contact:
    raise ValueError(f"Contact reference bodies are absent: {missing_contact}")
  contact_indices = torch.tensor([reference_index[name] for name in contact_reference_names], dtype=torch.long)
  contact_offsets = torch.tensor([item[2] for item in contact_specs], dtype=torch.float32)
  if not contact_specs:
    contact_offsets = torch.empty(0, 3, dtype=torch.float32)

  trajectory = compute_reference_centroidal(
    body_pos_w=torch.as_tensor(data["body_pos_w"], dtype=torch.float32),
    body_quat_w=torch.as_tensor(data["body_quat_w"], dtype=torch.float32),
    body_lin_vel_w=torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32),
    body_ang_vel_w=torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32),
    body_mass=torch.as_tensor(model.body_mass[model_ids], dtype=torch.float32),
    body_com_offset_b=torch.as_tensor(model.body_ipos[model_ids], dtype=torch.float32),
    body_inertia_diag=torch.as_tensor(model.body_inertia[model_ids], dtype=torch.float32),
    body_inertial_quat_b=torch.as_tensor(model.body_iquat[model_ids], dtype=torch.float32),
    contact_body_indices=contact_indices,
    contact_point_offset_b=contact_offsets,
    body_linear_velocity_point=velocity_point,
  )
  contact_pos_w = trajectory.contact_pos_w.numpy()
  if len(contact_specs):
    contact_state, contact_height, contact_speed = _infer_contact_state(
      contact_pos_w, fps, args.height_threshold, args.speed_threshold,
      args.min_stance_frames, args.min_swing_frames,
    )
  else:
    contact_state = np.empty((contact_pos_w.shape[0], 0), dtype=bool)
    contact_height = np.empty_like(contact_state, dtype=np.float32)
    contact_speed = np.empty_like(contact_state, dtype=np.float32)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    fps=np.asarray(fps, dtype=np.float32),
    com_pos_w=trajectory.com_pos_w.numpy(),
    com_vel_w=trajectory.com_vel_w.numpy(),
    linear_momentum_w=trajectory.linear_momentum_w.numpy(),
    angular_momentum_w=trajectory.angular_momentum_w.numpy(),
    contact_pos_w=contact_pos_w,
    contact_pos_rel_com_w=trajectory.contact_pos_rel_com_w.numpy(),
    contact_state=contact_state.astype(np.uint8),
    contact_height_w=contact_height.astype(np.float32),
    contact_speed_w=contact_speed.astype(np.float32),
    centroidal_reference_body_names=np.asarray(reference_names),
    centroidal_model_body_names=np.asarray(resolved_model_names),
    contact_labels=np.asarray(labels),
    contact_reference_body_names=np.asarray(contact_reference_names),
    contact_point_offsets_b=contact_offsets.numpy(),
    total_mass=np.asarray(float(model.body_mass[model_ids].sum()), dtype=np.float32),
    source_motion=np.asarray(str(args.motion.resolve())),
    source_xml=np.asarray(str(args.xml.resolve())),
    source_urdf=np.asarray("" if args.urdf is None else str(args.urdf.resolve())),
    source_body_linear_velocity_point=np.asarray(velocity_point),
    height_threshold=np.asarray(args.height_threshold, dtype=np.float32),
    speed_threshold=np.asarray(args.speed_threshold, dtype=np.float32),
    min_stance_frames=np.asarray(args.min_stance_frames, dtype=np.int32),
    min_swing_frames=np.asarray(args.min_swing_frames, dtype=np.int32),
  )
  summary = {
    "motion": str(args.motion.resolve()),
    "xml": str(args.xml.resolve()),
    "urdf_provenance": None if args.urdf is None else str(args.urdf.resolve()),
    "output": str(args.output.resolve()),
    "frames": int(trajectory.com_pos_w.shape[0]),
    "fps": fps,
    "duration_s": float((trajectory.com_pos_w.shape[0] - 1) / fps),
    "total_mass_kg": float(model.body_mass[model_ids].sum()),
    "body_map": {name: model_name for name, model_name in zip(reference_names, resolved_model_names, strict=True)},
    "contacts": [
      {"label": label, "reference_body": body, "offset_b_m": list(offset)}
      for label, body, offset in contact_specs
    ],
    "contact_thresholds": {
      "height_m": args.height_threshold,
      "speed_mps": args.speed_threshold,
      "min_stance_frames": args.min_stance_frames,
      "min_swing_frames": args.min_swing_frames,
    },
  }
  summary_path = args.output.with_suffix(".json")
  summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
