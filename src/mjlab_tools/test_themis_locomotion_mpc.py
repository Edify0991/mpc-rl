"""Batch-test the unmodified THEMIS locomotion centroidal MPC formulation.

The script constructs the same velocity-command reference and predictive
walking contact schedule used by ``themis_training.mpc_grf_mdp.LocoMPCCommand``
and solves all requested command/phase pairs in one batched QP call.  It does
not start MuJoCo or RL, so it isolates the original MPC formulation and makes
solver/contact-feasibility regressions inexpensive to diagnose.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from themis_mpc.centroidal_mpc import CentroidalMPC, MPCConfig, MPCInput
from themis_mpc.contact_schedule import make_walking_schedule

_ROOT = Path(__file__).resolve().parents[2]
_I_BODY = torch.tensor(((6.153, 0.0, 0.338), (0.0, 6.181, 0.0), (0.338, 0.0, 0.849)))


def _device(value: str) -> torch.device:
  if value == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  result = torch.device(value)
  if result.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
  return result


def _dynamics_residual(mpc: CentroidalMPC, mpc_in: MPCInput, output) -> torch.Tensor:
  Bk = mpc._build_Bk(mpc_in.x_ref[:, :-1, :3], mpc_in.schedule)
  state = mpc_in.x0
  residuals = []
  for k in range(mpc.cfg.N):
    predicted = state @ mpc.A.T + mpc.d
    predicted += torch.bmm(Bk[:, k], output.u_pred[:, k].unsqueeze(-1)).squeeze(-1)
    residuals.append(output.x_pred[:, k] - predicted)
    state = output.x_pred[:, k]
  return torch.stack(residuals, dim=1)


def _constraint_metrics(u: torch.Tensor, sigma: torch.Tensor, *, mu: float, mu_yaw: float,
                        toe: float, heel: float, half_y: float, fz_max: float) -> dict[str, float]:
  wrench = u.reshape(*u.shape[:-1], 2, 6)
  force, moment = wrench[..., :3], wrench[..., 3:]
  active = sigma[..., :2] >= .5
  inactive = force.abs().masked_select(~active.unsqueeze(-1))
  fz = force[..., 2]
  violation = torch.stack((
    -fz, fz - fz_max,
    force[..., 0].abs() - mu * fz,
    force[..., 1].abs() - mu * fz,
    moment[..., 0].abs() - half_y * fz,
    moment[..., 1] - toe * fz,
    -moment[..., 1] - heel * fz,
    moment[..., 2].abs() - mu_yaw * fz,
  ), dim=-1).clamp_min(0.)
  return {
    "max_inactive_force_N": float(inactive.amax().item()) if inactive.numel() else 0.0,
    "max_contact_constraint_violation": float(violation.max().item()),
  }


def _plot(path: Path, x_ref: np.ndarray, x_pred: np.ndarray, sigma: np.ndarray,
          u_pred: np.ndarray, v_cmd: np.ndarray, gait_phase: float) -> None:
  os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplcache"))
  import matplotlib.pyplot as plt

  t = np.arange(x_pred.shape[0])
  fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
  for j, label in enumerate(("x", "y", "z")):
    axes[0].plot(np.arange(x_ref.shape[0]), x_ref[:, j], "--", label=f"ref c{label}")
    axes[0].plot(t + 1, x_pred[:, j], label=f"MPC c{label}")
    axes[1].plot(np.arange(x_ref.shape[0]), x_ref[:, 3 + j], "--", label=f"ref l{label}")
    axes[1].plot(t + 1, x_pred[:, 3 + j], label=f"MPC l{label}")
    axes[2].plot(np.arange(x_ref.shape[0]), x_ref[:, 6 + j], "--", label=f"ref k{label}")
    axes[2].plot(t + 1, x_pred[:, 6 + j], label=f"MPC k{label}")
  axes[3].step(t, sigma[:, 0], where="post", label="left stance", color="C0")
  axes[3].step(t, sigma[:, 1], where="post", label="right stance", color="C1")
  axes[3].plot(t, u_pred[:, 2], label="left $f_z$ [N]", color="C2")
  axes[3].plot(t, u_pred[:, 8], label="right $f_z$ [N]", color="C3")
  axes[0].set_ylabel("CoM [m]")
  axes[1].set_ylabel("momentum [kg m/s]")
  axes[2].set_ylabel("ang. momentum")
  axes[3].set_ylabel("mode / normal force")
  axes[3].set_xlabel("MPC grid index")
  for axis in axes:
    axis.grid(True, alpha=.3)
    axis.legend(ncol=3, fontsize=8)
  fig.suptitle(
    "THEMIS locomotion MPC, batch element 0: "
    f"v=({v_cmd[0]:.2f}, {v_cmd[1]:.2f}) m/s, wz={v_cmd[2]:.2f} rad/s, phase={gait_phase:.2f}"
  )
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch-size", type=int, default=256, help="Number of velocity/phase cases solved concurrently.")
  parser.add_argument("--horizon", type=int, default=10)
  parser.add_argument("--mpc-dt", type=float, default=.07)
  parser.add_argument("--solver", choices=("pimpc", "jax_pimpc", "admm"), default="pimpc")
  parser.add_argument("--max-iterations", type=int, default=1000, help="Solver iteration budget; 200 reproduces the training default.")
  parser.add_argument("--device", default="auto")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output-dir", type=Path, default=_ROOT / "outputs/mpc_tests/themis_locomotion")
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument("--fail-on-violation", action="store_true", help="Return nonzero after writing outputs if hard checks fail.")
  args = parser.parse_args()
  if args.batch_size < 1 or args.horizon < 2 or args.mpc_dt <= 0.0 or args.max_iterations < 1:
    raise ValueError("batch-size >= 1, horizon >= 2, mpc-dt > 0, and max-iterations >= 1 are required")

  # These are the v1 THEMIS LocoMPCCommand settings in env_cfgs.py.
  mass, hip_width, gait_period, duty_factor, com_height = 37.0, .15, .9, .5, 1.17
  config = MPCConfig(
    N=args.horizon, dt=args.mpc_dt, mass=mass, solver_type=args.solver,
    admm_max_iter=args.max_iterations,
  )
  device = _device(args.device)
  generator = torch.Generator(device=device).manual_seed(args.seed)
  # UniformVelocityCommand can create both forward and lateral/yaw commands.
  v_cmd = torch.empty(args.batch_size, 3, device=device).uniform_(-1., 1., generator=generator)
  v_cmd[:, 0] = .1 + .6 * (v_cmd[:, 0] + 1.) * .5   # forward velocity [0.1, 0.7]
  v_cmd[:, 1] *= .25                                 # lateral velocity [-0.25, 0.25]
  v_cmd[:, 2] *= .8                                  # yaw rate [-0.8, 0.8]
  gait_phase = torch.linspace(0., 2. * torch.pi, args.batch_size, device=device)
  yaw = torch.zeros(args.batch_size, device=device)
  com = torch.zeros(args.batch_size, 3, device=device)
  com[:, 2] = com_height
  r_lf = torch.tensor((0., hip_width, 0.), device=device).expand(args.batch_size, -1).clone()
  r_rf = torch.tensor((0., -hip_width, 0.), device=device).expand(args.batch_size, -1).clone()
  schedule = make_walking_schedule(
    B=args.batch_size, N=args.horizon, r_LF=r_lf, r_RF=r_rf, gait_phase=gait_phase,
    period=gait_period, dt=args.mpc_dt, duty_factor=duty_factor, com_pos=com,
    v_cmd=v_cmd, yaw=yaw, yaw_rate=v_cmd[:, 2], hip_width=hip_width, device=device,
  )

  # Direct transcription of the production LocoMPCCommand velocity reference.
  grid = torch.arange(args.horizon + 1, device=device, dtype=torch.float32)
  yaw_grid = yaw[:, None] + v_cmd[:, 2:3] * grid * args.mpc_dt
  vx_grid = torch.cos(yaw_grid) * v_cmd[:, 0:1] - torch.sin(yaw_grid) * v_cmd[:, 1:2]
  vy_grid = torch.sin(yaw_grid) * v_cmd[:, 0:1] + torch.cos(yaw_grid) * v_cmd[:, 1:2]
  x_ref = torch.zeros(args.batch_size, args.horizon + 1, 9, device=device)
  x_ref[:, :, 2] = com_height
  x_ref[:, 1:, 0] = torch.cumsum(vx_grid[:, :-1] * args.mpc_dt, dim=1)
  x_ref[:, 1:, 1] = torch.cumsum(vy_grid[:, :-1] * args.mpc_dt, dim=1)
  x_ref[:, :, 3] = mass * vx_grid
  x_ref[:, :, 4] = mass * vy_grid
  x_ref[:, :, 8] = float(_I_BODY[2, 2]) * v_cmd[:, 2:3]
  mpc_in = MPCInput(
    x0=x_ref[:, 0], schedule=schedule, x_ref=x_ref,
    u_ref=torch.zeros(args.batch_size, args.horizon, 12, device=device),
    u_prev=torch.zeros(args.batch_size, 12, device=device),
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
  error = result.x_pred - x_ref[:, 1:]
  constraint = _constraint_metrics(
    result.u_pred, schedule.sigma, mu=config.mu_foot, mu_yaw=config.mu_foot_yaw,
    toe=config.foot_x_toe, heel=config.foot_x_heel, half_y=config.foot_y_half,
    fz_max=config.fz_max_foot,
  )
  report = {
    "test": "themis_original_locomotion_centroidal_mpc",
    "device": str(device), "solver": args.solver, "batch_size": args.batch_size,
    "horizon": args.horizon, "mpc_dt_s": args.mpc_dt, "max_iterations": args.max_iterations,
    "wall_time_ms": elapsed_ms,
    "solves_per_second": 1e3 * args.batch_size / elapsed_ms,
    "finite_output": bool(torch.isfinite(result.x_pred).all() and torch.isfinite(result.u_pred).all()),
    "solver_feasible_fraction": float(result.feasible.float().mean().item()),
    "dynamics_residual_max": float(residual.abs().max().item()),
    "dynamics_residual_rms": float(residual.square().mean().sqrt().item()),
    "reference_tracking_rms": {
      "com_m": float(error[..., :3].square().mean().sqrt().item()),
      "linear_momentum_kg_mps": float(error[..., 3:6].square().mean().sqrt().item()),
      "angular_momentum_kg_m2ps": float(error[..., 6:9].square().mean().sqrt().item()),
    },
    "flight_node_fraction": float((schedule.sigma[..., :2].sum(dim=-1) == 0).float().mean().item()),
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
    args.output_dir / "themis_locomotion_mpc_batch.npz", velocity_command=v_cmd.cpu().numpy(),
    gait_phase=gait_phase.cpu().numpy(), x_ref=x_ref.cpu().numpy(), x_pred=result.x_pred.cpu().numpy(),
    u_pred=result.u_pred.cpu().numpy(), contact_state=schedule.sigma[..., :2].cpu().numpy(),
  )
  (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  if not args.no_plot:
    _plot(args.output_dir / "batch_element_0.png", x_ref[0].cpu().numpy(), result.x_pred[0].cpu().numpy(),
          schedule.sigma[0, :, :2].cpu().numpy(), result.u_pred[0].cpu().numpy(),
          v_cmd[0].cpu().numpy(), float(gait_phase[0].item()))
  print(json.dumps(report, indent=2))
  if args.fail_on_violation and not report["hard_qp_checks_passed"]:
    raise SystemExit("Hard QP checks failed; inspect report.json and batch_element_0.png.")


if __name__ == "__main__":
  main()
