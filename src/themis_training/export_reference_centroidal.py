"""Export per-frame centroidal reference data from a BeyondMimic NPZ clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from .reference_centroidal import compute_reference_centroidal


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion", type=Path, help="BeyondMimic-format reference .npz")
  parser.add_argument("model", type=Path, help="Matched MuJoCo XML/MJCF")
  parser.add_argument("output", type=Path)
  parser.add_argument("--body", action="append", default=None, help="Centroidal body name; repeat. Default: every motion body.")
  parser.add_argument("--contact-body", action="append", default=[], help="Candidate MPC contact body; repeat.")
  args = parser.parse_args()

  data = np.load(args.motion, allow_pickle=False)
  required = {"body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_names"}
  missing = required.difference(data.files)
  if missing:
    raise ValueError(f"Reference is missing {sorted(missing)}")
  motion_names = [str(name) for name in data["body_names"].tolist()]
  velocity_point = str(np.asarray(data.get("body_linear_velocity_point", "inertial_com")).reshape(-1)[0])
  if velocity_point not in {"inertial_com", "link_origin"}:
    raise ValueError(f"Unsupported body_linear_velocity_point={velocity_point!r}")
  selected_names = args.body or motion_names
  index = {name: i for i, name in enumerate(motion_names)}
  missing_motion = [name for name in selected_names if name not in index]
  missing_contact = [name for name in args.contact_body if name not in index]
  if missing_motion or missing_contact:
    raise ValueError(f"Names absent from reference: centroidal={missing_motion}, contact={missing_contact}")
  contact_not_centroidal = [name for name in args.contact_body if name not in selected_names]
  if contact_not_centroidal:
    raise ValueError(f"Contact bodies must be selected centroidal bodies: {contact_not_centroidal}")

  model = mujoco.MjModel.from_xml_path(str(args.model))
  try:
    model_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in selected_names]
  except Exception as exc:
    raise ValueError("Failed resolving reference names in MuJoCo model") from exc
  unresolved = [name for name, body_id in zip(selected_names, model_ids, strict=True) if body_id < 0]
  if unresolved:
    raise ValueError(f"Reference/model body name mismatch: {unresolved}")
  motion_ids = [index[name] for name in selected_names]
  contact_ids = torch.tensor([selected_names.index(name) for name in args.contact_body], dtype=torch.long)
  device = "cpu"
  trajectory = compute_reference_centroidal(
    body_pos_w=torch.tensor(data["body_pos_w"][:, motion_ids], dtype=torch.float32, device=device),
    body_quat_w=torch.tensor(data["body_quat_w"][:, motion_ids], dtype=torch.float32, device=device),
    body_lin_vel_w=torch.tensor(data["body_lin_vel_w"][:, motion_ids], dtype=torch.float32, device=device),
    body_ang_vel_w=torch.tensor(data["body_ang_vel_w"][:, motion_ids], dtype=torch.float32, device=device),
    body_mass=torch.tensor(model.body_mass[model_ids], dtype=torch.float32, device=device),
    body_com_offset_b=torch.tensor(model.body_ipos[model_ids], dtype=torch.float32, device=device),
    body_inertia_diag=torch.tensor(model.body_inertia[model_ids], dtype=torch.float32, device=device),
    body_inertial_quat_b=torch.tensor(model.body_iquat[model_ids], dtype=torch.float32, device=device),
    contact_body_indices=contact_ids,
    body_linear_velocity_point=velocity_point,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    com_pos_w=trajectory.com_pos_w.numpy(),
    com_vel_w=trajectory.com_vel_w.numpy(),
    linear_momentum_w=trajectory.linear_momentum_w.numpy(),
    angular_momentum_w=trajectory.angular_momentum_w.numpy(),
    contact_pos_w=trajectory.contact_pos_w.numpy(),
    contact_pos_rel_com_w=trajectory.contact_pos_rel_com_w.numpy(),
    centroidal_body_names=np.asarray(selected_names),
    contact_body_names=np.asarray(args.contact_body),
    source_body_linear_velocity_point=np.asarray(velocity_point),
  )


if __name__ == "__main__":
  main()
