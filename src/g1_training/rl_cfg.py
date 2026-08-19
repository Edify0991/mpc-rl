"""PPO network and optimization settings owned by G1 training tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .multi_critic import MultiCriticPpoAlgorithmCfg


def g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
    experiment_name="g1_velocity",
    max_iterations=10_000,
    save_interval=500,
  )


def g1_hierarchical_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_hierarchical_hybrid_mimic"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def g1_mpc_rl_mimic_contact_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Single-critic PPO baseline for the full scalar Mimic+MPC reward."""
  cfg = g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_mpc_rl_mimic_contact"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def g1_mpc_rl_mimic_reference_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Single-critic PPO baseline for the fixed-reference Mimic MPC task."""
  cfg = g1_mpc_rl_mimic_contact_ppo_runner_cfg()
  cfg.experiment_name = "g1_mpc_rl_mimic_reference"
  return cfg


def g1_multi_critic_mpc_rl_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Four-value-head PPO variant of the MPC-contact Mimic teacher."""
  cfg = g1_mpc_rl_mimic_contact_ppo_runner_cfg()
  cfg.algorithm = MultiCriticPpoAlgorithmCfg(
    entropy_coef=cfg.algorithm.entropy_coef,
    learning_rate=cfg.algorithm.learning_rate,
    gamma=cfg.algorithm.gamma,
    lam=cfg.algorithm.lam,
  )
  cfg.experiment_name = "g1_mpc_rl_mimic_multi_critic"
  return cfg


def g1_multi_critic_mpc_rl_mimic_reference_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Four-value-head PPO variant of the fixed-reference Mimic MPC task."""
  cfg = g1_mpc_rl_mimic_reference_ppo_runner_cfg()
  cfg.algorithm = MultiCriticPpoAlgorithmCfg(
    entropy_coef=cfg.algorithm.entropy_coef,
    learning_rate=cfg.algorithm.learning_rate,
    gamma=cfg.algorithm.gamma,
    lam=cfg.algorithm.lam,
  )
  cfg.experiment_name = "g1_mpc_rl_mimic_reference_multi_critic"
  return cfg


def g1_multi_critic_hierarchical_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = g1_hierarchical_mimic_ppo_runner_cfg()
  cfg.algorithm = MultiCriticPpoAlgorithmCfg(
    entropy_coef=cfg.algorithm.entropy_coef,
    learning_rate=cfg.algorithm.learning_rate,
    gamma=cfg.algorithm.gamma,
    lam=cfg.algorithm.lam,
  )
  cfg.experiment_name = "g1_hierarchical_hybrid_mimic_multi_critic"
  return cfg


def g1_mpc_rl_mimic_student_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_mpc_rl_mimic_student"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.006
  return cfg
