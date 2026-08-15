"""PPO network and optimization settings owned by Jingchu01 training tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def jingchu01_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
    experiment_name="jingchu01_velocity",
    max_iterations=10_000,
    save_interval=500,
  )


def jingchu01_hierarchical_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = jingchu01_ppo_runner_cfg()
  cfg.experiment_name = "jingchu01_hierarchical_hybrid_mimic"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def jingchu01_mpc_rl_mimic_contact_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = jingchu01_ppo_runner_cfg()
  cfg.experiment_name = "jingchu01_mpc_rl_mimic_contact"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.008
  return cfg


def jingchu01_mpc_rl_mimic_student_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = jingchu01_ppo_runner_cfg()
  cfg.experiment_name = "jingchu01_mpc_rl_mimic_student"
  cfg.num_steps_per_env = 40
  cfg.algorithm.entropy_coef = 0.006
  return cfg
