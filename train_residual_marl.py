#!/usr/bin/env python
"""Legacy custom PyTorch PPO trainer (superseded by train_rllib.py).

Use `python train_rllib.py` for RLlib + Gymnasium training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from traffic_env import EnvConfig, MultiAgentTrafficEnv
from utility_model import RESIDUAL_PARAM_KEYS

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


def normalize_obs(obs: torch.Tensor, highway_length: float = 500.0) -> torch.Tensor:
    """Scale observation features to a PPO-friendly numeric range."""
    obs = obs.clone()
    obs[..., 0] = obs[..., 0] / highway_length
    obs[..., 1] = obs[..., 1] / 12.0
    obs[..., 2] = obs[..., 2] / 16.0
    obs[..., 3] = obs[..., 3] / np.pi
    obs[..., 4] = obs[..., 4] / np.pi
    obs[..., 5] = obs[..., 5] / 24.0
    obs[..., 6] = obs[..., 6] / 24.0

    # Neighbor blocks: [dx, dy, dvx, dvy]
    for start in range(7, obs.shape[-1], 4):
        obs[..., start] = obs[..., start] / 60.0
        obs[..., start + 1] = obs[..., start + 1] / 24.0
        obs[..., start + 2] = obs[..., start + 2] / 16.0
        obs[..., start + 3] = obs[..., start + 3] / 16.0
    return obs


class TorchResidualPolicy(nn.Module):
    """Shared actor-critic π(o)->ΔΘ with tanh-bounded Gaussian actor."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, residual_scale: float = 0.25):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(RESIDUAL_PARAM_KEYS)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((len(RESIDUAL_PARAM_KEYS),), -1.2))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_n = normalize_obs(obs)
        mean = self.actor(obs_n) * self.residual_scale
        value = self.critic(obs_n).squeeze(-1)
        return mean, value

    def distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def action_to_dict(self, action: np.ndarray) -> dict[str, float]:
        clipped = np.clip(action, -self.residual_scale, self.residual_scale)
        return dict(zip(RESIDUAL_PARAM_KEYS, clipped.astype(float)))

    def act(self, obs: np.ndarray, explore_std: float = 0.0) -> tuple[dict[str, float], torch.Tensor]:
        """Compatibility method used by visualization/demo scripts."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.forward(obs_t)
            if explore_std > 0:
                std = torch.full_like(mean, explore_std)
                action = torch.distributions.Normal(mean, std).sample()
                log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum()
            else:
                action = mean
                log_prob = torch.zeros(())
        return self.action_to_dict(action.cpu().numpy()), log_prob

    def sample_action(self, obs: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum()
        action_np = action_t.cpu().numpy()
        return self.action_to_dict(action_np), action_np, float(log_prob_t), float(value)


@dataclass
class PPOMemory:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[float]


def collect_rollouts(
    env: MultiAgentTrafficEnv,
    policy: TorchResidualPolicy,
    episodes_per_update: int,
) -> tuple[PPOMemory, list[float], list[int]]:
    memory = PPOMemory([], [], [], [], [], [])
    metrics: list[float] = []
    collisions: list[int] = []

    for _ in range(episodes_per_update):
        obs_list = env.reset()
        done = False
        while not done:
            residual_actions = []
            step_transition_indices: list[int] = []
            for obs in obs_list:
                action_dict, action_np, log_prob, value = policy.sample_action(obs)
                residual_actions.append(action_dict)
                memory.observations.append(obs)
                memory.actions.append(action_np)
                memory.log_probs.append(log_prob)
                memory.values.append(value)
                step_transition_indices.append(len(memory.rewards))

            obs_list, rewards, done, _ = env.step(residual_actions)
            for reward, _ in zip(rewards, step_transition_indices):
                memory.rewards.append(float(reward))
                memory.dones.append(float(done))

        metrics.append(env.rollout_metric())
        collisions.append(env.collision_count)

    return memory, metrics, collisions


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: TorchResidualPolicy,
    optimizer: torch.optim.Optimizer,
    memory: PPOMemory,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
) -> dict[str, float]:
    obs = torch.as_tensor(np.array(memory.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(memory.actions), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.array(memory.log_probs), dtype=torch.float32)
    values_np = np.array(memory.values, dtype=np.float32)
    rewards_np = np.array(memory.rewards, dtype=np.float32)
    dones_np = np.array(memory.dones, dtype=np.float32)

    advantages_np, returns_np = compute_gae(rewards_np, values_np, dones_np, gamma, gae_lambda)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = np.arange(n)
    last_stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            dist, value = policy.distribution(obs[batch_idx])
            new_log_probs = dist.log_prob(actions[batch_idx]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            log_ratio = new_log_probs - old_log_probs[batch_idx]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[batch_idx] - value).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            last_stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
    return last_stats


def baseline_metric(seed: int, episodes: int, max_steps: int) -> float:
    metrics = []
    for i in range(episodes):
        env = MultiAgentTrafficEnv(EnvConfig(max_steps=max_steps), seed=seed + i)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(None)
        metrics.append(env.rollout_metric())
    return float(np.mean(metrics))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = MultiAgentTrafficEnv(EnvConfig(max_steps=args.max_steps), seed=args.seed)
    policy = TorchResidualPolicy(
        probe_env.obs_dim,
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    base = baseline_metric(args.seed, episodes=max(1, args.baseline_episodes), max_steps=args.max_steps)
    print(f"Utility-only baseline metric over {args.baseline_episodes} episodes: {base:.3f}")

    best_metric = float("inf")
    for update in range(1, args.updates + 1):
        env = MultiAgentTrafficEnv(EnvConfig(max_steps=args.max_steps), seed=args.seed + update * 1000)
        memory, metrics, collisions = collect_rollouts(env, policy, args.episodes_per_update)
        stats = ppo_update(
            policy,
            optimizer,
            memory,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )

        mean_metric = float(np.mean(metrics))
        best_metric = min(best_metric, mean_metric)
        if update == 1 or update % max(args.updates // 10, 1) == 0:
            print(
                f"Update {update:4d}/{args.updates} | metric={mean_metric:8.3f} | "
                f"best={best_metric:8.3f} | collisions={np.mean(collisions):5.2f} | "
                f"loss={stats['loss']:8.3f} | entropy={stats['entropy']:6.3f}"
            )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "obs_dim": probe_env.obs_dim,
                "hidden_dim": args.hidden_dim,
                "residual_scale": policy.residual_scale,
                "algo": "ppo",
            },
            args.save,
        )
        print(f"Saved PPO residual policy to {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual MARL policy with PPO")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--baseline-episodes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/residual_policy.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
