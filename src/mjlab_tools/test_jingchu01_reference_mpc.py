"""Batch-test Jingchu01 reference-conditioned centroidal MPC.

This is an *offline QP test*, not an RL environment.  It reconstructs the
centroidal trajectory from the retargeted motion and Jingchu01 MJCF inertias,
then solves many independent MPC problems in one torch batch.  The input
sampling exactly mirrors ``MimicLocoMPCCommand``:

* CoM, linear/angular momentum, and contact point positions are linearly
  sampled from the 50-Hz clip;
* the contact mode is sampled with zero-order hold (ZOH), never interpolated.

The report distinguishes reference-tracking error (which can be nonzero for a
centroidally infeasible dance segment) from hard-QP checks: dynamics residual,
inactive-contact wrench, and friction/CoP violations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from jingchu01_mpc.centroidal_mpc import (
  CentroidalMPC,
  MPCConfig,
  MPCInput,
  MPCModelParameters,
)
from jingchu01_mpc.contact_schedule import make_reference_contact_schedule
from jingchu01_training.jingchu01.jingchu01_constants import (
  JINGCHU01_MPC_FOOT_X_HEEL,
  JINGCHU01_MPC_FOOT_X_TOE,
  JINGCHU01_MPC_FOOT_Y_HALF,
  JINGCHU01_MPC_FZ_MAX_FOOT,
  JINGCHU01_MPC_MU_FOOT,
  JINGCHU01_MPC_MU_FOOT_YAW,
)
from training_common.reference_centroidal import compute_reference_centroidal

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MOTION = _ROOT / "ref/audit_dance1_subject2/jingchu01_fullbody_dance1_subject2_full2_with_reference_contacts.npz"
_DEFAULT_XML = _ROOT / "src/jingchu01_training/jingchu01/xmls/jingchu01.xml"


def _device(value: str) -> torch.device:
  if value == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  requested = torch.device(value)
  if requested.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
  return requested


def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
  """Quaternion [w,x,y,z] to a rotation matrix, preserving leading dims."""
  quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
  w, x, y, z = quat.unbind(dim=-1)
  return torch.stack((
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ), dim=-1).reshape(*quat.shape[:-1], 3, 3)


def _linear_sample(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
  """Linearly sample ``[T,...]`` values at frame-valued ``[B,K]`` times."""
  lo = times.floor().long().clamp(0, values.shape[0] - 1)
  hi = (lo + 1).clamp(max=values.shape[0] - 1)
  alpha = (times - lo).to(values.dtype)
  while alpha.ndim < values[lo].ndim:
    alpha = alpha.unsqueeze(-1)
  return (1.0 - alpha) * values[lo] + alpha * values[hi]


def _zoh_sample(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
  """ZOH sample ``[T,...]`` values at frame-valued ``[B,K]`` times."""
  return values[times.floor().long().clamp(0, values.shape[0] - 1)]


def _reference_from_clip(
  motion_path: Path, xml_path: Path, contact_key: str, device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, float]:
  """Reconstruct the exact channels used by the Mimic centroidal command."""
  with np.load(motion_path, allow_pickle=False) as data:
    required = {
      "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
      "body_names", "fps", contact_key,
    }
    missing = required.difference(data.files)
    if missing:
      raise ValueError(f"{motion_path} is missing {sorted(missing)}")
    names = tuple(str(name) for name in data["body_names"].tolist())
    contact_names = tuple(str(name) for name in data.get("reference_contact_body_names", np.asarray(("left_ankle_roll", "right_ankle_roll"))).tolist())
    if len(contact_names) != 2:
      raise ValueError(f"This biped MPC test requires exactly two contact bodies, got {contact_names}")
    contact_offsets = np.asarray(
      data.get("reference_contact_point_offsets_b", np.asarray(((0., 0., 0.), (0., 0., 0.)))),
      dtype=np.float32,
    )
    if contact_offsets.shape != (2, 3):
      raise ValueError("reference_contact_point_offsets_b must have shape [2,3]")
    raw = {name: torch.as_tensor(data[name], dtype=torch.float32, device=device) for name in (
      "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    )}
    contacts = torch.as_tensor(data[contact_key], dtype=torch.float32, device=device)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    velocity_point = str(np.asarray(data.get("body_linear_velocity_point", "inertial_com")).reshape(-1)[0])

  if contacts.shape != (raw["body_pos_w"].shape[0], 2):
    raise ValueError(f"{contact_key} must have shape [T,2], got {tuple(contacts.shape)}")
  if fps <= 0.0:
    raise ValueError(f"fps must be positive, got {fps}")
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
  missing_bodies = [name for name in names if name not in model_names]
  if missing_bodies:
    raise ValueError(f"Reference bodies absent from MJCF: {missing_bodies}")
  missing_contacts = [name for name in contact_names if name not in names]
  if missing_contacts:
    raise ValueError(f"Contact bodies absent from reference: {missing_contacts}")
  model_ids = [model_names[name] for name in names]
  name_to_index = {name: i for i, name in enumerate(names)}
  trajectory = compute_reference_centroidal(
    **raw,
    body_mass=torch.as_tensor(model.body_mass[model_ids], dtype=torch.float32, device=device),
    body_com_offset_b=torch.as_tensor(model.body_ipos[model_ids], dtype=torch.float32, device=device),
    body_inertia_diag=torch.as_tensor(model.body_inertia[model_ids], dtype=torch.float32, device=device),
    body_inertial_quat_b=torch.as_tensor(model.body_iquat[model_ids], dtype=torch.float32, device=device),
    contact_body_indices=torch.tensor([name_to_index[name] for name in contact_names], device=device),
    contact_point_offset_b=torch.as_tensor(contact_offsets, dtype=torch.float32, device=device),
    body_linear_velocity_point=velocity_point,
  )
  return {
    "com": trajectory.com_pos_w,
    "linear_momentum": trajectory.linear_momentum_w,
    "angular_momentum": trajectory.angular_momentum_w,
    "contact_pos": trajectory.contact_pos_w,
    "contact_state": (contacts >= 0.5).float(),
    "contact_quat": raw["body_quat_w"][:, [name_to_index[name] for name in contact_names]],
  }, fps, float(model.body_mass[model_ids].sum())


def _dynamics_residual(mpc: CentroidalMPC, mpc_in: MPCInput, output) -> torch.Tensor:
  params = mpc._model_parameters(mpc_in, mpc_in.x0.shape[0])
  A, d = mpc._batched_dynamics(params)
  Bk = mpc._build_Bk(mpc_in.x_ref[:, :-1, :3], mpc_in.schedule, params.mass)
  state = mpc_in.x0
  residuals = []
  for k in range(mpc.cfg.N):
    predicted = torch.bmm(A, state.unsqueeze(-1)).squeeze(-1)
    predicted += torch.bmm(Bk[:, k], output.u_pred[:, k].unsqueeze(-1)).squeeze(-1) + d
    residuals.append(output.x_pred[:, k] - predicted)
    state = output.x_pred[:, k]
  return torch.stack(residuals, dim=1)


def _constraint_metrics(u: torch.Tensor, sigma: torch.Tensor, *, mu: float, mu_yaw: float,
                        toe: float, heel: float, half_y: float, fz_max: float) -> dict[str, float]:
  wrench = u.reshape(*u.shape[:-1], 2, 6)
  force, moment = wrench[..., :3], wrench[..., 3:]
  active = sigma[..., :2] >= 0.5
  inactive_abs = force.abs().masked_select(~active.unsqueeze(-1)).amax().item() if (~active).any() else 0.0
  fz = force[..., 2]
  violation = torch.stack((
    -fz, fz - fz_max,
    force[..., 0].abs() - mu * fz,
    force[..., 1].abs() - mu * fz,
    moment[..., 0].abs() - half_y * fz,
    moment[..., 1] - toe * fz,
    -moment[..., 1] - heel * fz,
    moment[..., 2].abs() - mu_yaw * fz,
  ), dim=-1).clamp_min(0.0)
  return {
    "max_inactive_force_N": float(inactive_abs),
    "max_contact_constraint_violation": float(violation.max().item()),
  }


def _plot(output_path: Path, x_ref: np.ndarray, x_pred: np.ndarray, sigma: np.ndarray, wrench: np.ndarray) -> None:
  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplcache"))
  import matplotlib.pyplot as plt

  t = np.arange(x_pred.shape[0])
  fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
  labels = ("x", "y", "z")
  for j, label in enumerate(labels):
    axes[0].plot(np.arange(x_ref.shape[0]), x_ref[:, j], "--", label=f"ref c{label}")
    axes[0].plot(t + 1, x_pred[:, j], label=f"MPC c{label}")
    axes[1].plot(np.arange(x_ref.shape[0]), x_ref[:, 3 + j], "--", label=f"ref l{label}")
    axes[1].plot(t + 1, x_pred[:, 3 + j], label=f"MPC l{label}")
    axes[2].plot(np.arange(x_ref.shape[0]), x_ref[:, 6 + j], "--", label=f"ref k{label}")
    axes[2].plot(t + 1, x_pred[:, 6 + j], label=f"MPC k{label}")
  axes[0].set_ylabel("CoM [m]")
  axes[1].set_ylabel("momentum [kg m/s]")
  axes[2].set_ylabel("ang. momentum")
  axes[3].step(t, sigma[:, 0], where="post", label="left contact", color="C0")
  axes[3].step(t, sigma[:, 1], where="post", label="right contact", color="C1")
  axes[3].plot(t, wrench[:, 2], label="left $f_z$ [N]", color="C2")
  axes[3].plot(t, wrench[:, 8], label="right $f_z$ [N]", color="C3")
  axes[3].set_ylabel("mode / normal force")
  axes[3].set_xlabel("MPC grid index")
  for axis in axes:
    axis.grid(True, alpha=.3)
    axis.legend(ncol=3, fontsize=8)
  fig.suptitle("Jingchu01 reference-conditioned centroidal MPC (batch element 0)")
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", type=Path, default=_DEFAULT_MOTION)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--contact-key", default="reference_contact_state")
  parser.add_argument("--batch-size", type=int, default=64, help="Independent clip starts solved in one batched MPC call.")
  parser.add_argument("--horizon", type=int, default=10)
  parser.add_argument("--mpc-dt", type=float, default=.07)
  parser.add_argument("--solver", choices=("pimpc", "jax_pimpc", "admm"), default="pimpc")
  parser.add_argument("--max-iterations", type=int, default=1000, help="Solver iteration budget; 200 reproduces the training default.")
  parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
  parser.add_argument("--output-dir", type=Path, default=_ROOT / "outputs/mpc_tests/jingchu01_reference")
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument("--fail-on-violation", action="store_true", help="Return nonzero after writing outputs if hard checks fail.")
  args = parser.parse_args()
  if args.batch_size < 1 or args.horizon < 2 or args.mpc_dt <= 0.0 or args.max_iterations < 1:
    raise ValueError("batch-size >= 1, horizon >= 2, mpc-dt > 0, and max-iterations >= 1 are required")
  if not args.motion.is_file() or not args.xml.is_file():
    raise FileNotFoundError("--motion and --xml must exist")

  device = _device(args.device)
  ref, fps, mass = _reference_from_clip(args.motion, args.xml, args.contact_key, device)
  max_start = ref["com"].shape[0] - 1 - args.horizon * args.mpc_dt * fps
  if max_start < 0:
    raise ValueError("Reference clip is shorter than one requested MPC horizon")
  starts = torch.linspace(0, max_start, args.batch_size, device=device)
  grid = starts[:, None] + torch.arange(args.horizon + 1, device=device) * (args.mpc_dt * fps)
  x_ref = torch.cat((
    _linear_sample(ref["com"], grid),
    _linear_sample(ref["linear_momentum"], grid),
    _linear_sample(ref["angular_momentum"], grid),
  ), dim=-1)
  contact_pos = _linear_sample(ref["contact_pos"], grid)
  # Deliberately use the 50-Hz discrete labels with ZOH: no fractional support.
  contact_state = _zoh_sample(ref["contact_state"], grid)[:, :-1]
  contact_quat = _linear_sample(ref["contact_quat"], grid[:, :1]).squeeze(1)
  schedule = make_reference_contact_schedule(
    B=args.batch_size, N=args.horizon, reference_contact_state=contact_state,
    reference_r_LF=contact_pos[:, :-1, 0], reference_r_RF=contact_pos[:, :-1, 1],
    R_LF_rot=_quat_to_rot(contact_quat[:, 0]), R_RF_rot=_quat_to_rot(contact_quat[:, 1]),
    device=device,
  )
  config = MPCConfig(
    N=args.horizon, dt=args.mpc_dt, mass=mass,
    foot_x_toe=JINGCHU01_MPC_FOOT_X_TOE, foot_x_heel=JINGCHU01_MPC_FOOT_X_HEEL,
    foot_y_half=JINGCHU01_MPC_FOOT_Y_HALF, mu_foot=JINGCHU01_MPC_MU_FOOT,
    mu_foot_yaw=JINGCHU01_MPC_MU_FOOT_YAW, fz_max_foot=JINGCHU01_MPC_FZ_MAX_FOOT,
    solver_type=args.solver, admm_max_iter=args.max_iterations,
  )
  params = MPCModelParameters(
    mass=torch.full((args.batch_size,), mass, device=device),
    foot_friction=torch.full((args.batch_size, 2), JINGCHU01_MPC_MU_FOOT, device=device),
    foot_yaw_friction=torch.full((args.batch_size, 2), JINGCHU01_MPC_MU_FOOT_YAW, device=device),
    normal_force_limit=torch.full((args.batch_size, 2), JINGCHU01_MPC_FZ_MAX_FOOT, device=device),
  )
  mpc_in = MPCInput(
    x0=x_ref[:, 0], schedule=schedule, x_ref=x_ref,
    u_ref=torch.zeros(args.batch_size, args.horizon, 12, device=device),
    u_prev=torch.zeros(args.batch_size, 12, device=device), model_parameters=params,
  )
  mpc = CentroidalMPC(config, device=device)
  if device.type == "cuda":
    torch.cuda.synchronize()
  t0 = time.perf_counter()
  with torch.no_grad():
    result = mpc.solve(mpc_in)
  if device.type == "cuda":
    torch.cuda.synchronize()
  elapsed_ms = 1e3 * (time.perf_counter() - t0)

  residual = _dynamics_residual(mpc, mpc_in, result)
  errors = result.x_pred - x_ref[:, 1:]
  constraint = _constraint_metrics(
    result.u_pred, schedule.sigma, mu=JINGCHU01_MPC_MU_FOOT,
    mu_yaw=JINGCHU01_MPC_MU_FOOT_YAW, toe=JINGCHU01_MPC_FOOT_X_TOE,
    heel=JINGCHU01_MPC_FOOT_X_HEEL, half_y=JINGCHU01_MPC_FOOT_Y_HALF,
    fz_max=JINGCHU01_MPC_FZ_MAX_FOOT,
  )
  report = {
    "test": "jingchu01_reference_centroidal_mpc",
    "motion": str(args.motion.resolve()), "xml": str(args.xml.resolve()),
    "contact_key": args.contact_key, "device": str(device), "solver": args.solver,
    "batch_size": args.batch_size, "horizon": args.horizon, "mpc_dt_s": args.mpc_dt,
    "max_iterations": args.max_iterations,
    "reference_fps": fps, "model_mass_kg": mass, "wall_time_ms": elapsed_ms,
    "solves_per_second": 1e3 * args.batch_size / elapsed_ms,
    "finite_output": bool(torch.isfinite(result.x_pred).all() and torch.isfinite(result.u_pred).all()),
    "solver_feasible_fraction": float(result.feasible.float().mean().item()),
    "dynamics_residual_max": float(residual.abs().max().item()),
    "dynamics_residual_rms": float(residual.square().mean().sqrt().item()),
    "reference_tracking_rms": {
      "com_m": float(errors[..., :3].square().mean().sqrt().item()),
      "linear_momentum_kg_mps": float(errors[..., 3:6].square().mean().sqrt().item()),
      "angular_momentum_kg_m2ps": float(errors[..., 6:9].square().mean().sqrt().item()),
    },
    "flight_node_fraction": float((contact_state.sum(dim=-1) == 0).float().mean().item()),
    **constraint,
  }
  report["hard_qp_checks_passed"] = bool(
    report["finite_output"] and report["solver_feasible_fraction"] == 1.0
    and report["dynamics_residual_max"] <= 1e-3
    and report["max_inactive_force_N"] <= 1e-3
    and report["max_contact_constraint_violation"] <= 1e-3
  )
  if args.solver in {"pimpc", "jax_pimpc"}:
    report["solver_semantics_note"] = (
      "PiMPC backends in this repository project only box wrench bounds; "
      "the reported full friction/CoP-cone metric is therefore an external audit. "
      "Use --solver admm for a strict full-cone QP acceptance test."
    )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output_dir / "jingchu01_reference_mpc_batch.npz", start_frame=starts.detach().cpu().numpy(),
    x_ref=x_ref.detach().cpu().numpy(), x_pred=result.x_pred.detach().cpu().numpy(),
    u_pred=result.u_pred.detach().cpu().numpy(), contact_state=contact_state.detach().cpu().numpy(),
  )
  (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  if not args.no_plot:
    _plot(args.output_dir / "batch_element_0.png", x_ref[0].cpu().numpy(), result.x_pred[0].cpu().numpy(),
          contact_state[0].cpu().numpy(), result.u_pred[0].cpu().numpy())
  print(json.dumps(report, indent=2))
  if args.fail_on_violation and not report["hard_qp_checks_passed"]:
    raise SystemExit("Hard QP checks failed; inspect report.json and batch_element_0.png.")


if __name__ == "__main__":
  main()
