"""Visualize or trim an MJLab motion NPZ without IsaacLab.

The viewer replays kinematic root/joint states through the same compiled
MuJoCo profile as training.  ``--trim-frame-range`` saves a self-consistent
clip: joint/root velocities and all body kinematics are reconstructed rather
than merely slicing old finite-difference arrays.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from .mjlab_motion_io import (
  RetargetedMotion,
  reconstruct_body_kinematics,
  robot_profile,
)


def _names(values: np.ndarray) -> tuple[str, ...]:
  return tuple(str(value) for value in values.tolist())


def _validate(data: np.lib.npyio.NpzFile) -> None:
  required = {"fps", "root_pos_w", "root_quat_w", "joint_pos", "joint_names"}
  if missing := required.difference(data.files):
    raise ValueError(f"Motion NPZ lacks required fields: {sorted(missing)}")
  frames = data["joint_pos"].shape[0]
  if frames < 2 or data["root_pos_w"].shape != (frames, 3) or data["root_quat_w"].shape != (frames, 4):
    raise ValueError("Motion root/joint arrays have invalid or inconsistent shapes")


def _save_trim(input_path: Path, output_path: Path, robot: str, frame_range: tuple[int, int]) -> None:
  with np.load(input_path, allow_pickle=False) as source:
    _validate(source)
    start, end = frame_range
    frames = source["joint_pos"].shape[0]
    if start < 1 or end < start or end > frames:
      raise ValueError(f"--trim-frame-range must satisfy 1 <= START <= END <= {frames}")
    selected = slice(start - 1, end)
    fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    motion = RetargetedMotion(
      root_pos_w=np.asarray(source["root_pos_w"][selected], dtype=np.float64),
      root_quat_w=np.asarray(source["root_quat_w"][selected], dtype=np.float64),
      joint_pos=np.asarray(source["joint_pos"][selected], dtype=np.float64),
      fps=fps, source_format="trimmed_npz",
    )
    model, _ = robot_profile(robot)
    joint_names = _names(source["joint_names"])
    reconstructed = reconstruct_body_kinematics(model, motion, joint_names)
    # Preserve scalar/provenance fields; old time-indexed kinematics are
    # intentionally discarded and replaced by the reconstruction above.
    payload: dict[str, np.ndarray] = {}
    for key in source.files:
      value = source[key]
      if value.ndim == 0 or value.shape[0] != frames:
        payload[key] = value
    payload.update(reconstructed)
    payload.update({
      "fps": np.asarray(fps, np.float32), "joint_names": np.asarray(joint_names),
      "source_joint_names": np.asarray(joint_names), "robot_profile": np.asarray(robot),
      "source_motion_file": np.asarray(str(input_path.resolve())),
      "trim_source_frame_range_1indexed": np.asarray((start, end), np.int32),
      # The CPU trimming helper reconstructs translational Jacobians at link
      # origins, whereas converted BeyondMimic/MJLab clips store CoM velocity.
      # Make that semantic change explicit so centroidal processing cannot
      # accidentally apply the wrong offset formula.
      "body_linear_velocity_point": np.asarray("link_origin"),
    })
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **payload)
  print(f"Wrote trimmed/reconstructed motion: {output_path}")


def _replay(path: Path, robot: str, loop: bool, playback_rate: float) -> None:
  with np.load(path, allow_pickle=False) as clip:
    _validate(clip)
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    root_pos, root_quat, joint_pos = clip["root_pos_w"], clip["root_quat_w"], clip["joint_pos"]
    joint_names = _names(clip["joint_names"])
  model, _ = robot_profile(robot)
  ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
  if np.any(ids < 0):
    raise ValueError("The motion joint names do not belong to --robot")
  free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if len(free_ids) != 1:
    raise ValueError("Compiled profile must have exactly one free root joint")
  root_qpos = int(model.jnt_qposadr[free_ids[0]])
  joint_qpos = model.jnt_qposadr[ids]
  data = mujoco.MjData(model)
  # Import lazily so trimming works on headless machines.
  from mujoco import viewer
  frame, frame_dt = 0, 1.0 / (fps * playback_rate)
  with viewer.launch_passive(model, data) as mj_viewer:
    while mj_viewer.is_running():
      data.qpos[:] = model.qpos0
      data.qpos[root_qpos:root_qpos + 3] = root_pos[frame]
      data.qpos[root_qpos + 3:root_qpos + 7] = root_quat[frame]
      data.qpos[joint_qpos] = joint_pos[frame]
      mujoco.mj_forward(model, data)
      mj_viewer.sync()
      time.sleep(frame_dt)
      frame += 1
      if frame >= len(root_pos):
        if not loop:
          break
        frame = 0


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion_file", type=Path)
  parser.add_argument("--robot", choices=("themis", "g1", "jingchu01"), required=True)
  parser.add_argument("--loop", action="store_true", help="Loop playback until the viewer is closed")
  parser.add_argument("--playback-rate", type=float, default=1.0)
  parser.add_argument("--trim-frame-range", nargs=2, type=int, metavar=("START", "END"))
  parser.add_argument("--trim-output", type=Path)
  args = parser.parse_args()
  if not args.motion_file.is_file():
    raise FileNotFoundError(args.motion_file)
  if args.playback_rate <= 0.0:
    raise ValueError("--playback-rate must be positive")
  if args.trim_frame_range is not None:
    if args.trim_output is None:
      raise ValueError("--trim-output is required with --trim-frame-range")
    _save_trim(args.motion_file, args.trim_output, args.robot, tuple(args.trim_frame_range))
    return
  _replay(args.motion_file, args.robot, args.loop, args.playback_rate)


if __name__ == "__main__":
  main()
