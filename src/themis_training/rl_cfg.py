"""RL configurations for THEMIS tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def themis_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for THEMIS velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="themis_velocity",
    max_iterations=10_000,
    save_interval=500,
  )


def themis_hierarchical_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO settings for the two-time-scale HybridMimic action layout.

  ``num_steps_per_env`` is a multiple of the five-step high-level hold so
  each rollout contains complete MPC macro-actions.  The actor remains a
  single factorized Gaussian because the installed rsl_rl runner owns one
  rollout and one critic; action dimensions are separated by the environment.
  """
  cfg = themis_ppo_runner_cfg()
  cfg.experiment_name = "themis_hierarchical_hybrid_mimic"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def themis_mpc_rl_mimic_contact_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO setup for the basic contact-aware MPC-RL imitation task."""
  cfg = themis_ppo_runner_cfg()
  cfg.experiment_name = "themis_mpc_rl_mimic_contact"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def themis_mpc_rl_mimic_student_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Causal student PPO configuration; DAgger updates are added externally."""
  cfg = themis_ppo_runner_cfg()
  cfg.experiment_name = "themis_mpc_rl_mimic_student"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.006
  return cfg
