"""Convert BeyondMimic retargeted CSV/PKL clips to MJLab reference NPZ.

The input and interpolation semantics intentionally match BeyondMimic's
``csv_to_npz.py``. It constructs the selected MJLab articulation, writes each
kinematic frame, runs a no-integration forward pass, and records the runtime
body state consumed by this repository's ``MotionReferenceCommand`` and
offline centroidal processor.

Example
-------
``python -m mjlab_tools.csv_to_npz_mjlab input.pkl output.npz --robot jingchu01 --output-fps 50``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mjlab_motion_io import (
  capture_motion_through_mjlab_simulator,
  load_retargeted_motion,
  resample_motion,
  robot_profile,
)


def _joint_names(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
  if value is None:
    return default
  names = tuple(name.strip() for name in value.split(",") if name.strip())
  if not names:
    raise ValueError("--joint-names must contain at least one comma-separated name")
  if len(set(names)) != len(names):
    raise ValueError("--joint-names contains duplicates")
  return names


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_file", type=Path, help="Retargeted CSV or PKL")
  parser.add_argument("output_file", type=Path, help="MJLab motion NPZ")
  parser.add_argument("--robot", choices=("themis", "g1", "jingchu01"), default="themis")
  parser.add_argument("--input-format", choices=("auto", "csv", "pkl"), default="auto")
  parser.add_argument("--input-fps", type=float, default=30.0, help="CSV frame rate; PKL's fps takes priority")
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument(
    "--sim-dt", type=float, default=0.02,
    help="MJLab simulator timestep for the no-integration capture pass; independent of output FPS",
  )
  parser.add_argument(
    "--device", default="cuda:0",
    help="MJLab/MJWarp device used for articulation write → forward → state capture",
  )
  parser.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"), default=None,
                      help="1-indexed inclusive input range")
  parser.add_argument("--joint-names", default=None, help="Optional input joint order, comma-separated")
  parser.add_argument("--pkl-root-rot-order", choices=("auto", "wxyz", "xyzw"), default="auto")
  args = parser.parse_args()
  if not args.input_file.is_file():
    raise FileNotFoundError(args.input_file)

  _, default_joint_names = robot_profile(args.robot)
  joint_names = _joint_names(args.joint_names, default_joint_names)
  raw = load_retargeted_motion(
    args.input_file, input_format=args.input_format, input_fps=args.input_fps,
    frame_range=None if args.frame_range is None else tuple(args.frame_range),
    pkl_root_rot_order=args.pkl_root_rot_order,
  )
  motion = resample_motion(raw, args.output_fps)
  result = capture_motion_through_mjlab_simulator(
    motion, robot=args.robot, joint_names=joint_names, sim_dt=args.sim_dt, device=args.device,
  )
  captured_joint_names = result.pop("joint_names")
  args.output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output_file,
    fps=np.asarray(motion.fps, dtype=np.float32),
    # ``joint_pos``/``joint_vel`` are read from EntityData and therefore use
    # the entity's native articulation order.  Preserve the input order as
    # provenance for auditing retargeting data.
    joint_names=captured_joint_names, source_joint_names=np.asarray(joint_names),
    source_joint_layout=np.asarray("retargeted_input_order"),
    saved_joint_layout=np.asarray("mjlab_entity_joint_order"),
    saved_body_layout=np.asarray("mjlab_entity_body_order"),
    source_motion_file=np.asarray(str(args.input_file.resolve())),
    source_format=np.asarray(raw.source_format),
    source_pkl_root_rot_order=np.asarray(raw.pkl_root_rot_order or ""),
    robot_profile=np.asarray(args.robot),
    # Preserve retargeter coordinates as a reproducible raw artifact.  The
    # static initial-anchor canonicalization is owned by MJLabMotionLoader at
    # training load, where it is also shared by tracker targets and CD-MPC.
    reference_frame_alignment=np.asarray("raw_source"),
    temporal_processing=np.asarray("lerp_joint_root_pos+slerp_root_quat+finite_difference_velocity"),
    capture_backend=np.asarray("mjlab_entity_write_forward_read"),
    capture_sim_dt=np.asarray(args.sim_dt, dtype=np.float32),
    capture_device=np.asarray(args.device),
    body_linear_velocity_point=np.asarray("inertial_com"),
    **result,
  )
  print(
    f"Wrote {args.output_file} | robot={args.robot}, frames={motion.root_pos_w.shape[0]}, "
    f"fps={motion.fps:g}, duration={(motion.root_pos_w.shape[0] - 1) / motion.fps:.3f}s, "
    f"joints={len(joint_names)}, bodies={len(result['body_names'])}"
  )


if __name__ == "__main__":
  main()
