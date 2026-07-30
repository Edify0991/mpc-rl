# Training-time MPC Guidance in Reinforcement Learning for Humanoid Loco-Manipulation

<p align="center">
  <a href="https://arxiv.org/abs/2606.05687"><b>Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation</b></a>
  <br>
  Junheng Li · Liang Wu · Sergio A. Esteban · Lizhi Yang · Ján Drgoňa · Aaron D. Ames
  <br>
  <a href="https://arxiv.org/abs/2606.05687">arXiv:2606.05687</a>
</p>

<p align="center">
  <img src="img/hero3.png" width="85%"/>
</p>

Humanoid locomotion and manipulation RL, guided at training time by a **GPU-batched centroidal
model-predictive controller (MPC)**, solved in parallel by **$\pi^n$ MPC**.

A centroidal QP-MPC is solved for every environment at the policy rate. Its
optimal plan — CoM and momentum trajectory, ground-reaction forces, foot
placements, and (for manipulation) hand push forces — is fed into the RL
environment to shape the reward and inform the critic. The policy therefore
learns to *track an optimal model-based plan* rather than to discover dynamic
locomotion and manipulation from scratch.

<p align="center">
  <img src="img/MPCRL.png" width="85%"/>
</p>

This repository contains only the training environments, the MPC, and the
proposed batch MPC solver **$\pi^n$ MPC** in JAX and PyTorch.

## Tasks

| Task ID | Description |
|---|---|
| `Mjlab-MPC-Guided-Locomotion-Themis` | MPC-guided velocity locomotion on flat terrain. |
| `Mjlab-MPC-Guided-Loco-manipulation-Themis` | MPC-guided loco-manipulation — walking and pushing a box. |
| `Mjlab-HybridMimic-MPC-Residual-Themis` | Hybrid imitation: frozen motion tracker + MPC-guided residual contact policy. |
| `Mjlab-Hierarchical-HybridMimic-MPC-Themis` | Jointly trained whole-body mimic residuals with a 10 Hz bounded MPC-parameter action. |
| `Mjlab-MPC-RL-Mimic-Contact-Themis` | Basic contact-aware MPC-RL motion imitation; no slow MPC-parameter action. |
| `Mjlab-MPC-RL-Mimic-Student-Themis` | Causal 29-D joint-target student for Phase-2 DAgger/PPO training. |
| `Mjlab-MotionTracker-Themis` | Stage-one BeyondMimic-style motion tracker. |

Unitree G1-29DOF task IDs and the required G1 reference-motion convention are
documented in [`docs/g1_29dof_training.md`](docs/g1_29dof_training.md).
Jingchu01 28-DOF training is documented in
[`docs/jingchu01_28dof_training.md`](docs/jingchu01_28dof_training.md).

## Hybrid motion imitation

`Mjlab-HybridMimic-MPC-Residual-Themis` uses reference PD tracking:

```text
tau = Kp(q_tracker - q) + Kd(dq_ref - dq)
```

The frozen first policy is the BeyondMimic-style motion tracker.  It receives
the current named reference frame plus robot state and emits a joint-position
offset around the reference.  The end-to-end hierarchical variant trains a
whole-body joint-target correction and two continuous contact-intention actions.
It observes motion state and MPC targets; its contact actions condition the MPC
plan and are supervised by the simulator sensor, rather than by a separate
history-based contact-prediction network.

The intended two-stage training order is: train
`Mjlab-MotionTracker-Themis` on the motion rewards, export it to TorchScript,
then freeze it through `tracker_policy_path` while training
`Mjlab-HybridMimic-MPC-Residual-Themis`. This keeps policy-one action
generation separate from policy-two dynamics correction.

`Mjlab-Hierarchical-HybridMimic-MPC-Themis` is the end-to-end training-time
variant for reference clips that cannot be exactly realized. Its one PPO
rollout has a low-level whole-body joint action `Δq` and two continuous
foot-contact-intention actions every policy step, plus a 16-D high-level
action held for five policy steps. The
held action only changes bounded CD-MPC parameters: contact-clock rate, duty
offset, touchdown mean/std and centroidal momentum residual. The detached MPC
then provides CoM, momentum, contact-force and contact-moment landmarks; the
same PPO return combines these with the motion mimic rewards. It does not
differentiate through MPC or MuJoCo, and touchdown variance is metadata for a
future chance-constrained extension rather than per-solve random sampling.

`Mjlab-MPC-RL-Mimic-Contact-Themis` is the Phase-1 teacher task. It has
`Δq` plus a full-horizon `2 × H` contact-plan residual action, with no 16-D
slow parameter action. Its actor sees only one upcoming q/dq reference frame;
the MPC internally uses the complete motion-derived horizon. Its MPC uses
the motion-derived
\(x^{ref}=[c^{ref},l^{ref},k^{ref}]\), zero wrench reference, state-tracking
weights `Q_c/Q_l/Q_k`, and wrench magnitude/slew weights
`R_f_foot/R_tau_foot/R_delta`. Its PPO reward is the sum of motion-mimic,
MPC-landmark (CoM, momentum, GRF, contact consistency), and regularization/
safety terms. It does not use a phase clock: supply optional `[T, 2]` contact
labels through `reference_contact_key`, or use its flat-ground height/velocity
fallback to construct the reference horizon contact sequence.

`Mjlab-MPC-RL-Mimic-Student-Themis` is the Phase-2 causal deployment task.
It removes MPC and the contact-plan action, retaining only the 29-D joint
target correction. `docs/two_stage_mpc_rl_mimic_contact.md` specifies the
DAgger teacher-query and loss interface.

The motion loader accepts BeyondMimic NPZ files containing `joint_names`,
`body_names`, `joint_pos`, `joint_vel`, and world-frame body trajectories. It
strictly validates names. The current checked-in simulator entity is THEMIS,
whereas BeyondMimic's supplied clips are G1 clips, so a G1 clip cannot be used
with the registered THEMIS task directly: first retarget it to the THEMIS
names, or replace the entity with a G1 MJCF/actuator configuration. This is a
safety check, not a format conversion.

To run the task, set `env.commands.motion.motion_file` to the retargeted NPZ
and optionally set `env.actions.hybrid_mimic.tracker_policy_path` to a
TorchScript export of the frozen tracker.  The hierarchical task instead uses
the raw reference plus its jointly trained joint-target action.  Its two
continuous foot-contact actions are contact intentions: they modulate the
frozen-before-solve MPC contact activation and are rewarded against the actual
MuJoCo foot-contact sensor.  No standalone GRU or offline contact dataset is
used. For exact whole-body centroidal quantities, pass every reference body
through `centroidal_body_names`; the default tracker-body subset is only a
compatibility fallback.

## Setup

Requires Python 3.11–3.13 and a CUDA 12 GPU.

```bash
uv sync
```

This installs everything: `mjlab` (MuJoCo / `mujoco-warp` / PyTorch / `rsl_rl`),
`scipy`, and `jax[cuda12]` for the JAX PiMPC solver.

## Training

Trained with mjlab's `train` entry point and 4096 parallel environments:

```bash
# Locomotion
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MPC-Guided-Locomotion-Themis \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 15000

# Loco-manipulation
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MPC-Guided-Loco-manipulation-Themis \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 25000
```

Resume from a checkpoint (regex match on the run-directory timestamp; omit
`--agent.load-checkpoint` to take the latest):

```bash
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MPC-Guided-Locomotion-Themis \
  --agent.resume True \
  --agent.load-run "2026-05-08_22-55-00" \
  --agent.load-checkpoint model_5000.pt
```

Play back a trained policy:

```bash
uv run play Mjlab-MPC-Guided-Locomotion-Themis --wandb-run-path <entity/project/run-id>
```

## MPC & solvers

The centroidal MPC lives in [`src/themis_mpc/`](src/themis_mpc/): `CentroidalMPC`
(locomotion) and `LocoManipMPC` (adds hand push-force variables). The QP is built
over a short horizon from a phase-driven contact schedule and solved **batched
across all environments in a single call**, selected per task via
`MPCConfig.solver_type`:

- **`jax_pimpc`** — $\pi^n$ MPC (parallel-in-horizon) compiled with JAX/XLA; the
  default for both tasks and the fastest at training scale.
- **`pimpc`** — the same $\pi^n$ MPC algorithm in PyTorch, more memory-efficient for lighter PC setups.
- **`admm`** — batched consensus-ADMM QP solver (PyTorch).

### Scalability of $\pi^n$ MPC

$\pi^n$MPC parallelizes the solve across **both** the prediction horizon and the
thousands of parallel environments, so a single batched call replaces 4096
sequential MPC solves. This is what makes running an MPC *inside* the RL loop
tractable.

<p align="center">
  <img src="img/batched_solvers.png" width="85%"/>
</p>

### Tuning MPC parameters

MPC parameters are set at two levels:

- **Per-task (horizon, solve rate, solver, robot/contact geometry)** — on the MPC
  command term in [`env_cfgs.py`](src/themis_training/env_cfgs.py): `mpc_dt`,
  `mpc_horizon`, `run_every_n_steps`, `solver_type`, `mass`, `gait_period`,
  `duty_factor`, `com_height` (plus, for loco-manipulation, `mu_hand`,
  `f_hand_max`, `R_f_hand`, `R_hand_balance`). Locomotion is configured in
  `_apply_mpc_grf_features` / `_apply_mpc_grf_v2_features`; loco-manipulation in
  the `LocoManipMPCCommandCfg(...)` block of
  `themis_loco_manip_mpc_push_box_flat_env_cfg`.
- **Cost weights, friction limits, and solver internals** — the `MPCConfig`
  dataclass in [`centroidal_mpc.py`](src/themis_mpc/centroidal_mpc.py) (and
  `LocoManipMPCConfig` in [`loco_manip_mpc.py`](src/themis_mpc/loco_manip_mpc.py)):
  tracking weights `Q_c` / `Q_l` / `Q_k`, terminal scale `Qf_scale`, input
  regularization `R_f_foot` / `R_tau_foot` / `R_delta`, friction `mu_foot` /
  `fz_max_foot` and foot geometry, and solver settings `admm_max_iter`,
  `pimpc_rho`, `pimpc_accel`, `pimpc_precondition`.

### MPC-parameter adaptor and control landmarks

`MPCParameterNet` (`src/themis_training/mpc_parameter_net.py`) is an optional
low-rate GRU adaptor for MPC parameters, not contact prediction. A TorchScript export receives
`[batch, history, 29]` state/reference features and outputs 16 raw values.  The
runtime bounds these into contact-clock rate, duty-factor offset, two XY
touchdown residual means and standard deviations, plus linear/angular momentum
reference residuals.  In the mimic task the retargeted reference touchdown is
the Gaussian mean; the adaptor outputs its bounded XY residual. Pass its export through
`themis_hybrid_mimic_env_cfg(..., mpc_parameter_network_path=...)`.

The numerical CD-MPC `dt` is deliberately fixed.  The network changes contact
timing through the phase clock before the QP is assembled, so contact position,
timing, and the CoM linearization are constants during each solve.  Gaussian
touchdown candidates are available for offline/low-rate robust candidate
selection; independent sampling inside every QP is intentionally not enabled.

Each MPC solve stores its full contact-wrench sequence
`[f_L, tau_L, f_R, tau_R]`, and the command time-interpolates that sequence for
the critic landmarks and force-tracking reward.  The default contact sensor can
supervise **forces** only.  A contact-moment reward needs a calibrated six-axis
force/torque sensor (or valid CoP/pressure reconstruction); moment landmarks
are therefore critic references by default.  In the MPC-guided imitation task,
these wrench trajectories remain landmarks and are not mapped to actuator
torques through a Jacobian.

The stochastic-contact extension path, conditional stability result, and joint
mimic/MPC-RL training design are in
[`docs/stochastic_contact_mpc_and_joint_mimic.md`](docs/stochastic_contact_mpc_and_joint_mimic.md).

## Acknowledgements

This work builds on:

- [**mjlab**](https://github.com/mujocolab/mjlab) — MuJoCo-Warp GPU RL training
  framework (provides MuJoCo / `mujoco-warp` / `rsl_rl` and the `train`/`play`
  entry points).
- [**$\pi$-MPC**](https://github.com/SOLARIS-JHU/PiMPC.jl) — the Julia-based parallel-in-horizon
  MPC solving method for time-invariant mono. MPCs.

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026accelerating,
  title={Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation},
  author={Li, Junheng and Wu, Liang and Esteban, Sergio A and Yang, Lizhi and Drgo{\v{n}}a, J{\'a}n and Ames, Aaron D},
  journal={arXiv preprint arXiv:2606.05687},
  year={2026}
}
```
