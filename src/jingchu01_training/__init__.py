"""Jingchu01 training-task entry point.

The package owns every registered Jingchu01 environment and keeps the
paper-compatible velocity-command tasks separate from reference-motion/mimic
tasks.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .multi_critic import MultiCriticVelocityOnPolicyRunner

from .env_cfgs import (
  jingchu01_flat_env_cfg,
  jingchu01_hierarchical_hybrid_mimic_env_cfg,
  jingchu01_motion_tracker_env_cfg,
  jingchu01_mpc_loco_manipulation_env_cfg,
  jingchu01_mpc_locomotion_env_cfg,
  jingchu01_mpc_rl_mimic_contact_env_cfg,
  jingchu01_mpc_rl_mimic_reference_env_cfg,
  jingchu01_mpc_rl_mimic_student_env_cfg,
)
from .rl_cfg import (
  jingchu01_hierarchical_mimic_ppo_runner_cfg,
  jingchu01_multi_critic_hierarchical_mimic_ppo_runner_cfg,
  jingchu01_multi_critic_mpc_rl_mimic_ppo_runner_cfg,
  jingchu01_multi_critic_mpc_rl_mimic_reference_ppo_runner_cfg,
  jingchu01_mpc_rl_mimic_contact_ppo_runner_cfg,
  jingchu01_mpc_rl_mimic_reference_ppo_runner_cfg,
  jingchu01_mpc_rl_mimic_student_ppo_runner_cfg,
  jingchu01_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Velocity-Jingchu01-28DOF",
  env_cfg=jingchu01_flat_env_cfg(),
  play_env_cfg=jingchu01_flat_env_cfg(play=True),
  rl_cfg=jingchu01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Locomotion-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_locomotion_env_cfg(),
  play_env_cfg=jingchu01_mpc_locomotion_env_cfg(play=True),
  rl_cfg=jingchu01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Loco-manipulation-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_loco_manipulation_env_cfg(),
  play_env_cfg=jingchu01_mpc_loco_manipulation_env_cfg(play=True),
  rl_cfg=jingchu01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Reference-motion tasks: separate from the paper-compatible command tasks.
register_mjlab_task(
  task_id="Mjlab-MotionTracker-Jingchu01-28DOF",
  env_cfg=jingchu01_motion_tracker_env_cfg(),
  play_env_cfg=jingchu01_motion_tracker_env_cfg(play=True),
  rl_cfg=jingchu01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=jingchu01_mpc_rl_mimic_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Contact-MultiCritic-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_contact_env_cfg(play=True),
  rl_cfg=jingchu01_multi_critic_mpc_rl_mimic_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Reference-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_reference_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_reference_env_cfg(play=True),
  rl_cfg=jingchu01_mpc_rl_mimic_reference_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Reference-MultiCritic-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_reference_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_reference_env_cfg(play=True),
  rl_cfg=jingchu01_multi_critic_mpc_rl_mimic_reference_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-MPC-RL-Mimic-Student-Jingchu01-28DOF",
  env_cfg=jingchu01_mpc_rl_mimic_student_env_cfg(),
  play_env_cfg=jingchu01_mpc_rl_mimic_student_env_cfg(play=True),
  rl_cfg=jingchu01_mpc_rl_mimic_student_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Hierarchical-HybridMimic-MPC-Jingchu01-28DOF",
  env_cfg=jingchu01_hierarchical_hybrid_mimic_env_cfg(),
  play_env_cfg=jingchu01_hierarchical_hybrid_mimic_env_cfg(play=True),
  rl_cfg=jingchu01_multi_critic_hierarchical_mimic_ppo_runner_cfg(),
  runner_cls=MultiCriticVelocityOnPolicyRunner,
)
