"""Multi-critic PPO for MPC-guided motion imitation.

The environment still exposes the scalar reward expected by MJLab, while this
runner reads the reward manager's per-term contributions after every step and
forms four *exactly additive* reward channels: MPC landmark, motion imitation,
task/stability, and regularization.  PPO uses a configured fixed linear
combination of their GAE advantages for the actor update and trains one
independent privileged critic per channel.  Jingchu01 Phase-1 uses
``(1.5, 1.0, 1.0, 1.0)`` to prioritize MPC landmarks; this is deliberately a
non-adaptive baseline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.rl import RslRlPpoAlgorithmCfg

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


CRITIC_NAMES = ("mpc_landmark", "mimic", "task", "regularization")
# Fixed Phase-1 scalarization.  This is intentionally not an adaptive gate:
# the MPC-landmark advantage is emphasized while the motion-mimic critic
# remains active.  Any future state-dependent fusion must be applied to the
# stored per-transition rewards before GAE, not multiplied onto completed GAE.
JINGCHU01_MIMIC_ACTOR_ADVANTAGE_WEIGHTS = (1.5, 1.0, 1.0, 1.0)
_REGULARIZATION_TERMS = frozenset({
  "dof_pos_limits", "action_rate_l2", "hybrid_torque", "residual_action",
  "hierarchical_mpc_parameter",
})


@dataclass
class MultiCriticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Algorithm settings for the four-channel Jingchu01 imitation return."""

  class_name: str = "jingchu01_training.multi_critic:MultiCriticPPO"
  # rsl_rl >= 4 expects these keys to be present even when the corresponding
  # extension is disabled.  The mjlab config base used by this repository
  # predates those fields, so add explicit null values to keep the serialized
  # runner configuration compatible.  MultiCriticPPO rejects non-null values:
  # decomposed returns are intentionally not combined with RND/symmetry yet.
  rnd_cfg: dict | None = None
  symmetry_cfg: dict | None = None
  critic_names: tuple[str, ...] = CRITIC_NAMES
  critic_value_loss_coefficients: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
  actor_advantage_weights: tuple[float, ...] = JINGCHU01_MIMIC_ACTOR_ADVANTAGE_WEIGHTS


def reward_group_for_term(name: str) -> str:
  """Return a disjoint critic channel for an active MJLab reward term."""
  if name.startswith("mpc_") or name == "future_contact_plan":
    return "mpc_landmark"
  if name.startswith("motion_"):
    return "mimic"
  if name in _REGULARIZATION_TERMS:
    return "regularization"
  return "task"


class MultiCriticRolloutStorage:
  """Feed-forward PPO storage with one reward/value/return per critic."""

  class Transition:
    def __init__(self) -> None:
      self.observations: TensorDict | None = None
      self.actions: torch.Tensor | None = None
      self.rewards: torch.Tensor | None = None
      self.dones: torch.Tensor | None = None
      self.values: torch.Tensor | None = None
      self.actions_log_prob: torch.Tensor | None = None
      self.distribution_params: tuple[torch.Tensor, ...] | None = None

    def clear(self) -> None:
      self.__init__()

  @dataclass
  class Batch:
    observations: TensorDict
    actions: torch.Tensor
    values: torch.Tensor
    actor_advantages: torch.Tensor
    returns: torch.Tensor
    old_actions_log_prob: torch.Tensor
    old_distribution_params: tuple[torch.Tensor, ...]

  def __init__(
    self,
    num_envs: int,
    num_steps: int,
    obs: TensorDict,
    action_dim: int,
    num_critics: int,
    device: str,
  ) -> None:
    self.device = device
    self.num_envs = num_envs
    self.num_transitions_per_env = num_steps
    self.num_critics = num_critics
    self.observations = TensorDict(
      {key: torch.zeros(num_steps, *value.shape, device=device) for key, value in obs.items()},
      batch_size=[num_steps, num_envs], device=device,
    )
    shape = (num_steps, num_envs, num_critics)
    self.rewards = torch.zeros(shape, device=device)
    self.values = torch.zeros(shape, device=device)
    self.returns = torch.zeros(shape, device=device)
    self.actor_advantages = torch.zeros(num_steps, num_envs, 1, device=device)
    self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
    self.dones = torch.zeros(num_steps, num_envs, 1, dtype=torch.uint8, device=device)
    self.actions_log_prob = torch.zeros(num_steps, num_envs, 1, device=device)
    self.distribution_params: tuple[torch.Tensor, ...] | None = None
    self.step = 0

  def add_transition(self, transition: Transition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Multi-critic rollout buffer overflow")
    assert transition.observations is not None
    assert transition.actions is not None and transition.rewards is not None
    assert transition.dones is not None and transition.values is not None
    assert transition.actions_log_prob is not None and transition.distribution_params is not None
    self.observations[self.step].copy_(transition.observations)
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards)
    self.values[self.step].copy_(transition.values)
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    if self.distribution_params is None:
      self.distribution_params = tuple(
        torch.zeros(self.num_transitions_per_env, *parameter.shape, device=self.device)
        for parameter in transition.distribution_params
      )
    for index, parameter in enumerate(transition.distribution_params):
      self.distribution_params[index][self.step].copy_(parameter)
    self.step += 1

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
    if self.distribution_params is None:
      raise RuntimeError("No rollout transitions were stored")
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
    observations = self.observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    advantages = self.actor_advantages.flatten(0, 1)
    log_prob = self.actions_log_prob.flatten(0, 1)
    distribution = tuple(parameter.flatten(0, 1) for parameter in self.distribution_params)
    for _ in range(num_epochs):
      for index in range(num_mini_batches):
        batch_indices = indices[index * mini_batch_size : (index + 1) * mini_batch_size]
        yield self.Batch(
          observations=observations[batch_indices], actions=actions[batch_indices],
          values=values[batch_indices], actor_advantages=advantages[batch_indices],
          returns=returns[batch_indices], old_actions_log_prob=log_prob[batch_indices],
          old_distribution_params=tuple(parameter[batch_indices] for parameter in distribution),
        )

  def clear(self) -> None:
    self.step = 0


class MultiCriticPPO:
  """PPO with one value function for each additive reward component."""

  def __init__(
    self,
    actor: MLPModel,
    critics: nn.ModuleDict,
    storage: MultiCriticRolloutStorage,
    *,
    critic_names: tuple[str, ...] = CRITIC_NAMES,
    critic_value_loss_coefficients: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    actor_advantage_weights: tuple[float, ...] = JINGCHU01_MIMIC_ACTOR_ADVANTAGE_WEIGHTS,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    learning_rate: float = 1.0e-3,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float | None = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    value_loss_coef: float = 1.0,
    device: str = "cpu",
    multi_gpu_cfg: dict | None = None,
    **unsupported: object,
  ) -> None:
    if unsupported.get("rnd_cfg") is not None or unsupported.get("symmetry_cfg") is not None:
      raise ValueError("MultiCriticPPO currently does not combine RND/symmetry with decomposed returns")
    if tuple(critics.keys()) != tuple(critic_names):
      raise ValueError("critic_names and critic ModuleDict keys must agree")
    if len(critic_names) != len(critic_value_loss_coefficients) or len(critic_names) != len(actor_advantage_weights):
      raise ValueError("Every critic needs one value-loss and actor-advantage coefficient")
    self.device = device
    self.actor = actor.to(device)
    self.critics = critics.to(device)
    # Compatibility for generic checkpoint tooling; the full checkpoint uses
    # ``critics_state_dict`` and never reduces training to this head.
    self.critic = self.critics["task"]
    self.critic_names = tuple(critic_names)
    self.storage = storage
    self.transition = MultiCriticRolloutStorage.Transition()
    self.optimizer = resolve_optimizer(optimizer)(chain(self.actor.parameters(), self.critics.parameters()), lr=learning_rate)
    self.value_loss_coefficients = torch.tensor(critic_value_loss_coefficients, device=device)
    self.actor_advantage_weights = torch.tensor(actor_advantage_weights, device=device)
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.clip_param = clip_param
    self.gamma = gamma
    self.lam = lam
    self.entropy_coef = entropy_coef
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.schedule = schedule
    self.desired_kl = desired_kl
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
    self.value_loss_coef = value_loss_coef
    self.is_multi_gpu = multi_gpu_cfg is not None
    self.gpu_global_rank = 0 if multi_gpu_cfg is None else multi_gpu_cfg["global_rank"]
    self.gpu_world_size = 1 if multi_gpu_cfg is None else multi_gpu_cfg["world_size"]

  @classmethod
  def construct_algorithm(cls, obs: TensorDict, env: "VecEnv", cfg: dict, device: str) -> "MultiCriticPPO":
    # Normally stripped by ``MjlabOnPolicyRunner.__init__``. Keep the
    # constructor independently usable by unit tests and custom launchers.
    for model_key in ("actor", "critic"):
      for option in ("cnn_cfg", "distribution_cfg"):
        if cfg[model_key].get(option) is None:
          cfg[model_key].pop(option, None)
    algorithm_cfg = cfg["algorithm"]
    algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    critic_names = tuple(algorithm_cfg.pop("critic_names"))
    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    critics = nn.ModuleDict({
      name: critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
      for name in critic_names
    })
    if actor.is_recurrent or any(critic.is_recurrent for critic in critics.values()):
      raise ValueError("MultiCriticPPO currently supports feed-forward actor and critics only")
    storage = MultiCriticRolloutStorage(
      env.num_envs, cfg["num_steps_per_env"], obs, env.num_actions, len(critic_names), device,
    )
    return algorithm_class(
      actor, critics, storage, critic_names=critic_names, device=device,
      multi_gpu_cfg=cfg["multi_gpu"], **algorithm_cfg,
    )

  def _values(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([self.critics[name](obs) for name in self.critic_names], dim=-1)

  def act(self, obs: TensorDict) -> torch.Tensor:
    self.transition.actions = self.actor(obs, stochastic_output=True).detach()
    self.transition.values = self._values(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
    self.transition.distribution_params = tuple(parameter.detach() for parameter in self.actor.output_distribution_params)
    self.transition.observations = obs
    return self.transition.actions

  def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
    if rewards.shape != (self.storage.num_envs, len(self.critic_names)):
      raise ValueError(f"Expected grouped rewards {(self.storage.num_envs, len(self.critic_names))}, got {tuple(rewards.shape)}")
    self.actor.update_normalization(obs)
    for critic in self.critics.values():
      critic.update_normalization(obs)
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device)
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    for critic in self.critics.values():
      critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    last_values = self._values(obs).detach()
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(self.storage.num_transitions_per_env)):
      next_values = last_values if step == self.storage.num_transitions_per_env - 1 else self.storage.values[step + 1]
      not_terminal = 1.0 - self.storage.dones[step].float()
      delta = self.storage.rewards[step] + not_terminal * self.gamma * next_values - self.storage.values[step]
      advantage = delta + not_terminal * self.gamma * self.lam * advantage
      self.storage.returns[step] = advantage + self.storage.values[step]
    self.storage.actor_advantages = (self.storage.returns - self.storage.values).matmul(self.actor_advantage_weights).unsqueeze(-1)
    if not self.normalize_advantage_per_mini_batch:
      advantage = self.storage.actor_advantages
      self.storage.actor_advantages = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)

  def update(self) -> dict[str, float]:
    mean_surrogate = 0.0
    mean_entropy = 0.0
    mean_values = torch.zeros(len(self.critic_names), device=self.device)
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      advantages = batch.actor_advantages
      if self.normalize_advantage_per_mini_batch:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self._values(batch.observations)
      entropy = self.actor.output_entropy
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl_mean = self.actor.get_kl_divergence(batch.old_distribution_params, self.actor.output_distribution_params).mean()
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate
      ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
      surrogate = -advantages.squeeze(-1) * ratio
      surrogate_clipped = -advantages.squeeze(-1) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        clipped_values = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        per_critic_loss = torch.maximum((values - batch.returns).square(), (clipped_values - batch.returns).square()).mean(dim=0)
      else:
        per_critic_loss = (values - batch.returns).square().mean(dim=0)
      value_loss = (per_critic_loss * self.value_loss_coefficients).sum()
      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critics.parameters()), self.max_grad_norm)
      self.optimizer.step()
      mean_surrogate += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_values += per_critic_loss.detach()
    updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    losses = {"surrogate": mean_surrogate / updates, "entropy": mean_entropy / updates}
    for index, name in enumerate(self.critic_names):
      losses[f"value_{name}"] = (mean_values[index] / updates).item()
    losses["value"] = sum(losses[f"value_{name}"] for name in self.critic_names) / len(self.critic_names)
    return losses

  def train_mode(self) -> None:
    self.actor.train()
    self.critics.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critics.eval()

  def get_policy(self) -> MLPModel:
    return self.actor

  def save(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critics_state_dict": self.critics.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "critic_names": self.critic_names,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_cfg = load_cfg or {"actor": True, "critics": True, "optimizer": True, "iteration": True}
    if load_cfg.get("actor"):
      self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if load_cfg.get("critics") and "critics_state_dict" in loaded_dict:
      self.critics.load_state_dict(loaded_dict["critics_state_dict"], strict=strict)
    elif load_cfg.get("critic") and "critic_state_dict" in loaded_dict:
      self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=False)
    if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return bool(load_cfg.get("iteration", False))

  def broadcast_parameters(self) -> None:
    state = [self.actor.state_dict(), self.critics.state_dict()]
    torch.distributed.broadcast_object_list(state, src=0)
    self.actor.load_state_dict(state[0])
    self.critics.load_state_dict(state[1])

  def reduce_parameters(self) -> None:
    parameters = [parameter for parameter in chain(self.actor.parameters(), self.critics.parameters()) if parameter.grad is not None]
    if not parameters:
      return
    flattened = torch.cat([parameter.grad.view(-1) for parameter in parameters])
    torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
    flattened /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      size = parameter.numel()
      parameter.grad.copy_(flattened[offset : offset + size].view_as(parameter.grad))
      offset += size


class MultiCriticVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that supplies exact reward-manager channels to PPO."""

  alg: MultiCriticPPO

  def _group_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
    manager = self.env.unwrapped.reward_manager
    scale = self.env.unwrapped.step_dt if manager._scale_by_dt else 1.0
    term_values = manager._step_reward * scale
    grouped = torch.zeros(self.env.num_envs, len(self.alg.critic_names), device=rewards.device)
    name_to_index = {name: index for index, name in enumerate(self.alg.critic_names)}
    for term_index, term_name in enumerate(manager.active_terms):
      grouped[:, name_to_index[reward_group_for_term(term_name)]] += term_values[:, term_index].to(rewards.device)
    if not torch.allclose(grouped.sum(dim=-1), rewards, rtol=1.0e-4, atol=1.0e-6):
      max_error = (grouped.sum(dim=-1) - rewards).abs().max().item()
      raise RuntimeError(f"Multi-critic reward partition no longer matches scalar environment reward (max error={max_error:.3e})")
    return grouped

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    for iteration in range(start_it, start_it + num_learning_iterations):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          grouped_rewards = self._group_rewards(rewards)
          obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
          self.alg.process_env_step(obs, grouped_rewards.to(self.device), dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
        collect_time = time.time() - start
        start = time.time()
        self.alg.compute_returns(obs)
      loss_dict = self.alg.update()
      learn_time = time.time() - start
      self.current_learning_iteration = iteration
      self.logger.log(
        it=iteration, start_it=start_it, total_it=start_it + num_learning_iterations,
        collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate, action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      if self.logger.writer is not None and iteration % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{iteration}.pt"))
    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.logger.stop_logging_writer()
