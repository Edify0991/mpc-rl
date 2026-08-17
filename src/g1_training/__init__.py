"""Unitree G1 training-task entry point.

The package owns every registered G1 environment.  Paper-compatible velocity
locomotion and loco-manipulation IDs intentionally live beside (but never
share factories with) the reference-motion/mimic pipeline.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .multi_critic import MultiCriticVelocityOnPolicyRunner

from .env_cfgs import (
  g1_flat_env_cfg,
  g1_hierarchical_hybrid_mimic_env_cfg,
  g1_motion_tracker_env_cfg,
  g1_mpc_loco_manipulation_env_cfg,
  g1_mpc_locomotion_env_cfg,
  g1_mpc_rl_mimic_contact_env_cfg,
  g1_mpc_rl_mimic_reference_env_cfg,
  g1_mpc_rl_mimic_student_env_cfg,
)
from .rl_cfg import (
  g1_hierarchical_mimic_ppo_runner_cfg,
  g1_multi_critic_hierarchical_mimic_ppo_runner_cfg,
  g1_multi_critic_mpc_rl_mimic_ppo_runner_cfg,
  g1_mpc_rl_mimic_contact_ppo_runner_cfg,
  g1_mpc_rl_mimic_student_ppo_runner_cfg,
  g1_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Velocity-G1-29DOF",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Locomotion-G1-29DOF",
  env_cfg=g1_mpc_locomotion_env_cfg(),
  play_env_cfg=g1_mpc_locomotion_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Loco-manipulation-G1-29DOF",
  env_cfg=g1_mpc_loco_manipulation_env_cfg(),
  play_env_cfg=g1_mpc_loco_manipulation_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Reference-motion tasks: separate from the paper-compatible command tasks.
register_mjlab_task(
  task_id="Mjlab-MotionTracker-G1-29DOF",
  env_cfg=g1_motion_tracker_env_cfg(),
  play_env_cfg=g1_motion_tracker_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-G1-29DOF",
  env_cfg=g1_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=g1_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=g1_multi_critic_mpc_rl_mimic_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Reference-G1-29DOF",
  env_cfg=g1_mpc_rl_mimic_reference_env_cfg(),
  play_env_cfg=g1_mpc_rl_mimic_reference_env_cfg(play=True),
  rl_cfg=g1_multi_critic_mpc_rl_mimic_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Student-G1-29DOF",
  env_cfg=g1_mpc_rl_mimic_student_env_cfg(),
  play_env_cfg=g1_mpc_rl_mimic_student_env_cfg(play=True),
  rl_cfg=g1_mpc_rl_mimic_student_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Hierarchical-HybridMimic-MPC-G1-29DOF",
  env_cfg=g1_hierarchical_hybrid_mimic_env_cfg(),
  play_env_cfg=g1_hierarchical_hybrid_mimic_env_cfg(play=True),
  rl_cfg=g1_multi_critic_hierarchical_mimic_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
