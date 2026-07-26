"""Create publication-style plots from a processed reference-centroidal NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
  fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
  fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
  plt.close(fig)


def _plot_xyz(ax: plt.Axes, time_s: np.ndarray, values: np.ndarray, ylabel: str) -> None:
  for component, color, name in zip(range(3), _COLORS, ("x", "y", "z")):
    ax.plot(time_s, values[:, component], color=color, linewidth=1.0, label=name)
  ax.set_ylabel(ylabel)
  ax.grid(True, alpha=0.25)
  ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)


def _equal_3d_axes(ax: plt.Axes, points: np.ndarray) -> None:
  minimum = points.min(axis=0)
  maximum = points.max(axis=0)
  center = 0.5 * (minimum + maximum)
  radius = max(float((maximum - minimum).max()) * 0.55, 0.05)
  ax.set_xlim(center[0] - radius, center[0] + radius)
  ax.set_ylim(center[1] - radius, center[1] + radius)
  ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("processed", type=Path, help="Processed NPZ from process_reference_centroidal")
  parser.add_argument("output_dir", type=Path)
  args = parser.parse_args()
  if not args.processed.is_file():
    raise FileNotFoundError(args.processed)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  data = np.load(args.processed, allow_pickle=False)
  required = {
    "fps", "com_pos_w", "com_vel_w", "linear_momentum_w", "angular_momentum_w",
    "contact_pos_w", "contact_state", "contact_height_w", "contact_speed_w", "contact_labels",
  }
  missing = required.difference(data.files)
  if missing:
    raise ValueError(f"Processed file is missing: {sorted(missing)}")
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  com = data["com_pos_w"]
  velocity = data["com_vel_w"]
  linear_momentum = data["linear_momentum_w"]
  angular_momentum = data["angular_momentum_w"]
  contact_pos = data["contact_pos_w"]
  contact_state = data["contact_state"].astype(bool)
  contact_height = data["contact_height_w"]
  contact_speed = data["contact_speed_w"]
  labels = [str(label) for label in data["contact_labels"].tolist()]
  time_s = np.arange(com.shape[0]) / fps

  plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
  })

  fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.5), sharex=True, layout="constrained")
  _plot_xyz(axes[0], time_s, com, "CoM [m]")
  _plot_xyz(axes[1], time_s, velocity, "CoM velocity [m/s]")
  _plot_xyz(axes[2], time_s, linear_momentum, "Linear momentum [kg m/s]")
  _plot_xyz(axes[3], time_s, angular_momentum, "Angular momentum [kg m²/s]")
  axes[0].set_title("Reference centroidal trajectory")
  axes[-1].set_xlabel("Time [s]")
  _save(fig, args.output_dir, "centroidal_time_series")

  contacts = contact_pos.shape[1]
  if contacts:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True, layout="constrained")
    for index, label in enumerate(labels):
      axes[0].plot(time_s, contact_height[:, index], linewidth=1.0, label=label)
      axes[1].plot(time_s, contact_speed[:, index], linewidth=1.0, label=label)
    axes[0].set_ylabel("Contact height [m]")
    axes[1].set_ylabel("Contact speed [m/s]")
    axes[0].set_title("Kinematic contact inference")
    for axis in axes[:2]:
      axis.grid(True, alpha=0.25)
      axis.legend(loc="upper right", frameon=False)
    axes[2].imshow(
      contact_state.T, aspect="auto", interpolation="nearest",
      cmap=ListedColormap(("#F2F2F2", "#0072B2")),
      extent=(time_s[0], time_s[-1], contacts - 0.5, -0.5), vmin=0, vmax=1,
    )
    axes[2].set_yticks(np.arange(contacts), labels)
    axes[2].set_ylabel("Contact")
    axes[2].set_title("Kinematic stance schedule (blue = stance)", fontsize=10)
    axes[2].set_xlabel("Time [s]")
    _save(fig, args.output_dir, "contact_kinematics")

    fig = plt.figure(figsize=(8.0, 7.0), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(com[:, 0], com[:, 1], com[:, 2], color="black", linewidth=1.2, label="CoM")
    all_points = [com]
    for index, label in enumerate(labels):
      trajectory = contact_pos[:, index]
      all_points.append(trajectory)
      ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.7, alpha=0.65, label=label)
      stance = contact_state[:, index]
      ax.scatter(
        trajectory[stance, 0], trajectory[stance, 1], trajectory[stance, 2],
        s=3.0, alpha=0.35,
      )
    points = np.concatenate(all_points, axis=0)
    _equal_3d_axes(ax, points)
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_zlabel("World z [m]")
    ax.set_title("Reference CoM and candidate contact trajectories")
    ax.legend(loc="upper left", frameon=False)
    ax.view_init(elev=22, azim=-58)
    _save(fig, args.output_dir, "centroidal_spatial_trajectory")


if __name__ == "__main__":
  main()
