"""MPC-guided RL training tasks for the Westwood Robotics THEMIS humanoid.

Registers two tasks with mjlab:

* ``Mjlab-MPC-Guided-Locomotion-Themis`` — MPC-guided velocity locomotion.
  A centroidal QP-MPC runs in parallel with the policy at the policy rate and
  supplies CoM / angular-momentum / GRF / foot-placement reference targets that
  shape the reward (solver: batched JAX PiMPC).

* ``Mjlab-MPC-Guided-Loco-manipulation-Themis`` — MPC-guided
  loco-manipulation. Extends the locomotion task with a box-pushing scene and
  swaps the locomotion-only MPC for the loco-manipulation MPC that also solves
  for hand push forces (solver: batched PyTorch PiMPC).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  themis_hierarchical_hybrid_mimic_env_cfg,
  themis_hybrid_mimic_env_cfg,
  themis_loco_manip_mpc_push_box_flat_env_cfg,
  themis_mpc_rl_mimic_contact_env_cfg,
  themis_mpc_rl_mimic_student_env_cfg,
  themis_mpc_grf_v2_flat_env_cfg,
  themis_motion_tracker_env_cfg,
)
from .rl_cfg import (
  themis_hierarchical_mimic_ppo_runner_cfg,
  themis_mpc_rl_mimic_contact_ppo_runner_cfg,
  themis_mpc_rl_mimic_student_ppo_runner_cfg,
  themis_ppo_runner_cfg,
)
from .g1_env_cfgs import (
  g1_flat_env_cfg,
  g1_hierarchical_hybrid_mimic_env_cfg,
  g1_motion_tracker_env_cfg,
  g1_mpc_rl_mimic_contact_env_cfg,
  g1_mpc_rl_mimic_student_env_cfg,
)
from .jingchu01_env_cfgs import (
  jingchu01_flat_env_cfg,
  jingchu01_hierarchical_hybrid_mimic_env_cfg,
  jingchu01_motion_tracker_env_cfg,
  jingchu01_mpc_rl_mimic_contact_env_cfg,
  jingchu01_mpc_rl_mimic_student_env_cfg,
)

# ── MPC-guided velocity locomotion ──────────────────────────────────────────
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Locomotion-Themis",
  env_cfg=themis_mpc_grf_v2_flat_env_cfg(),
  play_env_cfg=themis_mpc_grf_v2_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Stage one of the two-policy pipeline.  It has the BeyondMimic observation
# and reward design but requires an explicit retargeted reference at runtime.
register_mjlab_task(
  task_id="Mjlab-MotionTracker-Themis",
  env_cfg=themis_motion_tracker_env_cfg(),
  play_env_cfg=themis_motion_tracker_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# HybridMimic is deliberately registered without a default clip.  Supply a
# retargeted motion path through the config before training; strict name checks
# prevent accidentally applying a G1 reference to the THEMIS articulation.
register_mjlab_task(
  task_id="Mjlab-HybridMimic-MPC-Residual-Themis",
  env_cfg=themis_hybrid_mimic_env_cfg(),
  play_env_cfg=themis_hybrid_mimic_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Jointly train the BeyondMimic-style low-level tracker and a slower
# MPC-parameter policy head.  The MPC is detached and supplies centroidal /
# wrench landmarks; it does not take part in simulator back-propagation.
register_mjlab_task(
  task_id="Mjlab-Hierarchical-HybridMimic-MPC-Themis",
  env_cfg=themis_hierarchical_hybrid_mimic_env_cfg(),
  play_env_cfg=themis_hierarchical_hybrid_mimic_env_cfg(play=True),
  rl_cfg=themis_hierarchical_mimic_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Basic MPC-RL extension for motion imitation: reference centroidal tracking,
# zero-wrench control regularization, and fast learned contact activation.
# It intentionally does not enable the slower 16-D MPC-parameter action used
# by the hierarchical task above.
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-Themis",
  env_cfg=themis_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=themis_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Causal deployment student. The registered PPO task provides the same mimic
# and regularization rewards; use dagger_distillation.py to interleave teacher
# queries and supervised updates during Phase-2 training.
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Student-Themis",
  env_cfg=themis_mpc_rl_mimic_student_env_cfg(),
  play_env_cfg=themis_mpc_rl_mimic_student_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_student_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# ── MPC-guided loco-manipulation (box pushing) ──────────────────────────────
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Loco-manipulation-Themis",
  env_cfg=themis_loco_manip_mpc_push_box_flat_env_cfg(),
  play_env_cfg=themis_loco_manip_mpc_push_box_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# ── Unitree G1-29DOF ──────────────────────────────────────────────────────
# Kept in separate factories from THEMIS: joint order, feet/contact geometry,
# motor limits and centroidal mass/inertia are all G1-specific.
register_mjlab_task(
  task_id="Mjlab-Velocity-G1-29DOF",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MotionTracker-G1-29DOF",
  env_cfg=g1_motion_tracker_env_cfg(),
  play_env_cfg=g1_motion_tracker_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-G1-29DOF",
  env_cfg=g1_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=g1_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Student-G1-29DOF",
  env_cfg=g1_mpc_rl_mimic_student_env_cfg(),
  play_env_cfg=g1_mpc_rl_mimic_student_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_student_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Hierarchical-HybridMimic-MPC-G1-29DOF",
  env_cfg=g1_hierarchical_hybrid_mimic_env_cfg(),
  play_env_cfg=g1_hierarchical_hybrid_mimic_env_cfg(play=True),
  rl_cfg=themis_hierarchical_mimic_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# ── Jingchu01 28-DOF ──────────────────────────────────────────────────────
register_mjlab_task(
  task_id="Mjlab-Velocity-Jingchu01-28DOF",
  env_cfg=jingchu01_flat_env_cfg(),
  play_env_cfg=jingchu01_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MotionTracker-Jingchu01-28DOF",
  env_cfg=jingchu01_motion_tracker_env_cfg(),
  play_env_cfg=jingchu01_motion_tracker_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Hierarchical-HybridMimic-MPC-Jingchu01-28DOF",
  env_cfg=jingchu01_hierarchical_hybrid_mimic_env_cfg(),
  play_env_cfg=jingchu01_hierarchical_hybrid_mimic_env_cfg(play=True),
  rl_cfg=themis_hierarchical_mimic_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Student-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_student_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_student_env_cfg(play=True),
  rl_cfg=themis_mpc_rl_mimic_student_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
