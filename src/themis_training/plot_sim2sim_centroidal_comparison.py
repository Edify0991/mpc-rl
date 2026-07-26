"""Publication-style comparison of reference and sim2sim centroidal trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np


_AXIS_COLORS = ("#0072B2", "#D55E00", "#009E73")  # Okabe-Ito, colour-blind safe.
_SIM_COLOR = "#0072B2"
_ERROR_COLOR = "#CC79A7"


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
  fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
  fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
  plt.close(fig)


def _interpolate(values: np.ndarray, source_progress: np.ndarray, target_progress: np.ndarray) -> np.ndarray:
  flat = values.reshape(values.shape[0], -1)
  interpolated = np.stack(
    [np.interp(target_progress, source_progress, flat[:, component]) for component in range(flat.shape[1])],
    axis=-1,
  )
  return interpolated.reshape((len(target_progress),) + values.shape[1:])


def _nearest_state(values: np.ndarray, source_progress: np.ndarray, target_progress: np.ndarray) -> np.ndarray:
  right = np.searchsorted(source_progress, target_progress, side="left")
  right = np.clip(right, 0, len(source_progress) - 1)
  left = np.clip(right - 1, 0, len(source_progress) - 1)
  choose_right = (np.abs(source_progress[right] - target_progress) < np.abs(source_progress[left] - target_progress))
  return values[np.where(choose_right, right, left)]


def _component_plot(
  axes: np.ndarray, progress: np.ndarray, reference: np.ndarray, simulation: np.ndarray,
) -> None:
  for component in range(3):
    axis = axes[component]
    axis.plot(progress, reference[:, component], color="black", linestyle="--", linewidth=0.9, label="Reference")
    axis.plot(progress, simulation[:, component], color=_AXIS_COLORS[component], linewidth=1.05, label="Sim2sim")
    axis.grid(True, alpha=0.25)
    axis.set_title(("x", "y", "z")[component], fontsize=10)
    if component == 0:
      axis.legend(loc="upper right", frameon=False, fontsize=8)


def _equal_3d_axes(ax: plt.Axes, values: np.ndarray) -> None:
  minimum, maximum = values.min(axis=0), values.max(axis=0)
  center = 0.5 * (minimum + maximum)
  radius = max(float((maximum - minimum).max()) * 0.55, 0.05)
  ax.set_xlim(center[0] - radius, center[0] + radius)
  ax.set_ylim(center[1] - radius, center[1] + radius)
  ax.set_zlim(center[2] - radius, center[2] + radius)


def _time_colored_path(ax: plt.Axes, x: np.ndarray, y: np.ndarray, progress: np.ndarray, *, linewidth: float) -> None:
  """Plot a trajectory with both hue and opacity increasing in time."""
  points = np.column_stack((x, y))
  segments = np.stack((points[:-1], points[1:]), axis=1)
  midpoint_progress = 0.5 * (progress[:-1] + progress[1:])
  colors = plt.get_cmap("viridis")(midpoint_progress)
  colors[:, 3] = 0.30 + 0.70 * midpoint_progress
  collection = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round")
  ax.add_collection(collection)


def _add_progress_anchors(
  ax: plt.Axes,
  reference: np.ndarray,
  simulation: np.ndarray,
  progress: np.ndarray,
  components: tuple[int, int],
) -> None:
  """Show corresponding normalized-time landmarks with a readable identity cue."""
  for value in np.linspace(0.0, 1.0, 6):
    index = int(np.argmin(np.abs(progress - value)))
    x_ref, y_ref = reference[index, list(components)]
    x_sim, y_sim = simulation[index, list(components)]
    ax.plot((x_ref, x_sim), (y_ref, y_sim), color="#7F7F7F", linestyle=":", linewidth=0.65, alpha=0.65, zorder=0)
    ax.scatter(x_ref, y_ref, marker="o", s=22, facecolor="white", edgecolor="black", linewidth=0.7, zorder=4)
    ax.scatter(x_sim, y_sim, marker="s", s=20, facecolor="white", edgecolor="black", linewidth=0.7, zorder=4)
    if value in (0.0, 0.5, 1.0):
      ax.annotate(f"{value:.0%}", (x_sim, y_sim), xytext=(4, 4), textcoords="offset points", fontsize=7)


def _signal_metrics(reference: np.ndarray, simulation: np.ndarray) -> dict[str, object]:
  error = simulation - reference
  return {
    "rmse_per_axis": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
    "mae_per_axis": np.mean(np.abs(error), axis=0).tolist(),
    "max_abs_per_axis": np.max(np.abs(error), axis=0).tolist(),
    "vector_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1)))),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("reference", type=Path, help="NPZ from process_reference_centroidal")
  parser.add_argument("simulation", type=Path, help="NPZ from process_sim2sim_centroidal")
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--samples", type=int, default=1600, help="Common normalized-time samples for plots/metrics")
  args = parser.parse_args()
  if args.samples < 10:
    raise ValueError("--samples must be at least 10")
  if not args.reference.is_file() or not args.simulation.is_file():
    raise FileNotFoundError("Reference and simulation processed NPZ files must exist")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  reference = np.load(args.reference, allow_pickle=False)
  simulation = np.load(args.simulation, allow_pickle=False)
  required = {"com_pos_w", "com_vel_w", "linear_momentum_w", "angular_momentum_w", "contact_state", "contact_labels"}
  for name, data in (("reference", reference), ("simulation", simulation)):
    if missing := required.difference(data.files):
      raise ValueError(f"{name} NPZ is missing: {sorted(missing)}")
  labels_reference = [str(item) for item in reference["contact_labels"].tolist()]
  labels_simulation = [str(item) for item in simulation["contact_labels"].tolist()]
  if labels_reference != labels_simulation:
    raise ValueError(f"Contact labels differ: reference={labels_reference}, sim={labels_simulation}")

  ref_progress = np.linspace(0.0, 1.0, len(reference["com_pos_w"]))
  sim_progress = np.asarray(simulation["motion_progress"], dtype=np.float64) if "motion_progress" in simulation.files else np.linspace(0.0, 1.0, len(simulation["com_pos_w"]))
  progress = np.linspace(0.0, 1.0, args.samples)
  names = (
    ("com_pos_w", "CoM position [m]"),
    ("com_vel_w", "CoM velocity [m/s]"),
    ("linear_momentum_w", "Linear momentum [kg m/s]"),
    ("angular_momentum_w", "Angular momentum [kg m²/s]"),
  )
  resampled = {
    key: (_interpolate(reference[key], ref_progress, progress), _interpolate(simulation[key], sim_progress, progress))
    for key, _ in names
  }
  contact_reference = _nearest_state(reference["contact_state"].astype(bool), ref_progress, progress)
  contact_simulation = _nearest_state(simulation["contact_state"].astype(bool), sim_progress, progress)
  force_threshold = (
    float(np.asarray(simulation["contact_force_threshold_n"]).reshape(-1)[0])
    if "contact_force_threshold_n" in simulation.files else None
  )

  plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
  })
  fig, axes = plt.subplots(4, 3, figsize=(11.0, 8.8), sharex=True, layout="constrained")
  for row, (key, ylabel) in enumerate(names):
    _component_plot(axes[row], progress, *resampled[key])
    axes[row, 0].set_ylabel(ylabel)
  fig.suptitle("Centroidal tracking: reference and sim2sim (initial SE(2) aligned)", fontsize=12)
  for axis in axes[-1]:
    axis.set_xlabel("Normalized motion progress")
  _save(fig, args.output_dir, "centroidal_reference_sim2sim_components")

  fig, axes = plt.subplots(5, 1, figsize=(10.0, 9.0), sharex=True, layout="constrained")
  for axis, (key, ylabel) in zip(axes[:4], names):
    vector_error = np.linalg.norm(resampled[key][1] - resampled[key][0], axis=-1)
    axis.plot(progress, vector_error, color=_ERROR_COLOR, linewidth=1.0)
    axis.set_ylabel(f"‖error‖\n{ylabel.split(' [')[1][:-1]}")
    axis.grid(True, alpha=0.25)
  contacts = len(labels_reference)
  contact_image = np.concatenate((contact_reference.T, contact_simulation.T), axis=0)
  axes[4].imshow(
    contact_image, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
    cmap=ListedColormap(("#F2F2F2", _SIM_COLOR)), extent=(0.0, 1.0, 2 * contacts - 0.5, -0.5),
  )
  axes[4].set_yticks(np.arange(2 * contacts), [f"Ref. {name}" for name in labels_reference] + [f"Sim. {name}" for name in labels_reference])
  axes[4].set_ylabel("Contact state")
  sim_contact_title = "Sim2sim: logged contact label" if force_threshold is None else f"Sim2sim: foot-force norm ≥ {force_threshold:g} N"
  axes[4].set_title(f"Reference: kinematic schedule; {sim_contact_title}. White = swing, blue = stance", fontsize=9)
  axes[-1].set_xlabel("Normalized motion progress")
  fig.suptitle("Centroidal vector errors and contact schedule", fontsize=12)
  _save(fig, args.output_dir, "centroidal_error_and_contact")

  reference_com, simulation_com = resampled["com_pos_w"]
  fig = plt.figure(figsize=(10.0, 4.4), layout="constrained")
  ax_xy = fig.add_subplot(121)
  ax_xy.plot(reference_com[:, 0], reference_com[:, 1], "--", color="black", linewidth=1.0, label="Reference")
  ax_xy.plot(simulation_com[:, 0], simulation_com[:, 1], color=_SIM_COLOR, linewidth=1.1, label="Sim2sim")
  ax_xy.set_xlabel("World x [m]")
  ax_xy.set_ylabel("World y [m]")
  ax_xy.set_title("CoM horizontal trajectory")
  ax_xy.axis("equal")
  ax_xy.grid(True, alpha=0.25)
  ax_xy.legend(frameon=False)
  ax_3d = fig.add_subplot(122, projection="3d")
  ax_3d.plot(*reference_com.T, "--", color="black", linewidth=1.0, label="Reference")
  ax_3d.plot(*simulation_com.T, color=_SIM_COLOR, linewidth=1.1, label="Sim2sim")
  _equal_3d_axes(ax_3d, np.concatenate((reference_com, simulation_com)))
  ax_3d.set_xlabel("World x [m]")
  ax_3d.set_ylabel("World y [m]")
  ax_3d.set_zlabel("World z [m]")
  ax_3d.set_title("CoM spatial trajectory")
  ax_3d.view_init(elev=22, azim=-58)
  _save(fig, args.output_dir, "com_reference_sim2sim_spatial")

  # A spatial plot with a time coordinate.  Both trajectories use the same
  # colormap, so identical hue means identical normalized motion progress;
  # black dashed/solid backbones and circle/square anchors identify the source.
  figure = plt.figure(figsize=(10.4, 7.6), layout="constrained")
  grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.70))
  ax_xy = figure.add_subplot(grid[0, 0])
  ax_xz = figure.add_subplot(grid[0, 1])
  ax_distance = figure.add_subplot(grid[1, :])
  for axis, components, title, labels in (
    (ax_xy, (0, 1), "Horizontal CoM path", ("World x [m]", "World y [m]")),
    (ax_xz, (0, 2), "Sagittal-height CoM path", ("World x [m]", "World z [m]")),
  ):
    axis.plot(
      reference_com[:, components[0]], reference_com[:, components[1]],
      color="black", linestyle="--", linewidth=0.8, alpha=0.75, label="Reference backbone",
    )
    axis.plot(
      simulation_com[:, components[0]], simulation_com[:, components[1]],
      color="#555555", linestyle="-", linewidth=0.8, alpha=0.75, label="Sim2sim backbone",
    )
    _time_colored_path(axis, reference_com[:, components[0]], reference_com[:, components[1]], progress, linewidth=1.45)
    _time_colored_path(axis, simulation_com[:, components[0]], simulation_com[:, components[1]], progress, linewidth=2.05)
    _add_progress_anchors(axis, reference_com, simulation_com, progress, components)
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
  ax_xy.axis("equal")
  ax_xy.legend(loc="upper right", frameon=False, fontsize=8)
  ax_xz.legend(loc="upper right", frameon=False, fontsize=8)

  reference_distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(reference_com, axis=0), axis=-1))))
  simulation_distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(simulation_com, axis=0), axis=-1))))
  ax_distance.plot(progress, reference_distance, color="black", linestyle="--", linewidth=1.1, label="Reference")
  ax_distance.plot(progress, simulation_distance, color=_SIM_COLOR, linewidth=1.2, label="Sim2sim")
  for value in np.linspace(0.0, 1.0, 6):
    ax_distance.axvline(value, color="#7F7F7F", linewidth=0.55, linestyle=":", alpha=0.6)
  ax_distance.set_xlabel("Normalized motion progress (reference: 79.96 s; sim2sim: 79.98 s)")
  ax_distance.set_ylabel("Cumulative CoM path length [m]")
  ax_distance.set_title("Spatial progression over time")
  ax_distance.grid(True, alpha=0.25)
  ax_distance.legend(frameon=False, ncol=2)
  colorbar = figure.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap="viridis"),
    ax=(ax_xy, ax_xz), orientation="horizontal", pad=0.13, fraction=0.06,
  )
  colorbar.set_label("Normalized motion progress: early/transparent → late/opaque")
  figure.suptitle("Time-coloured CoM spatial alignment", fontsize=12)
  _save(figure, args.output_dir, "com_time_colored_spatial_alignment")

  contact_metrics: dict[str, dict[str, float]] = {}
  for index, label in enumerate(labels_reference):
    truth, prediction = contact_reference[:, index], contact_simulation[:, index]
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    contact_metrics[label] = {
      "agreement": float(np.mean(truth == prediction)),
      "precision": float(tp / max(tp + fp, 1)),
      "recall_against_reference_schedule": float(tp / max(tp + fn, 1)),
      "reference_stance_fraction": float(np.mean(truth)),
      "sim2sim_stance_fraction": float(np.mean(prediction)),
    }
  metrics = {
    "alignment": "One fixed initial-root yaw and translation transform was applied to sim2sim positions; velocity and momentum vectors use the same yaw rotation.",
    "temporal_alignment": "Both trajectories are linearly resampled over normalized motion progress [0, 1], because their recording rates/frame counts differ.",
    "sim2sim_contact_state": "logged boolean" if force_threshold is None else f"foot force norm >= {force_threshold:g} N",
    "samples": args.samples,
    **{key: _signal_metrics(*resampled[key]) for key, _ in names},
    "contact": contact_metrics,
  }
  (args.output_dir / "centroidal_comparison_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
  print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
