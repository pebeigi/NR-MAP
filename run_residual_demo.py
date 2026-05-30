#!/usr/bin/env python
"""Compare utility-only baseline vs residual-modulated rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from traffic_env import EnvConfig, MultiAgentTrafficEnv

try:
    import torch
    from train_residual_marl import TorchResidualPolicy
except ImportError:
    torch = None
    TorchResidualPolicy = None


def run_rollout(env: MultiAgentTrafficEnv, policy=None, explore_std: float = 0.0) -> dict:
    obs_list = env.reset()
    done = False
    while not done:
        if policy is None:
            residual_actions = None
        else:
            residual_actions = []
            for obs in obs_list:
                action, _ = policy.act(obs, explore_std=explore_std)
                residual_actions.append(action)
        obs_list, rewards, done, info = env.step(residual_actions)

    return {
        "metric": env.rollout_metric(),
        "collisions": info["collision_count"],
        "steps": info["steps"],
        "destinations_reached": info["destinations_reached"],
        "mean_step_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/residual_policy.pt"))
    args = parser.parse_args()

    env_cfg = EnvConfig()
    baseline_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    baseline = run_rollout(baseline_env, policy=None)
    print("=== Utility-only baseline ===")
    print(baseline)

    if torch is None or not args.checkpoint.exists():
        print("\nNo trained checkpoint found. Run: python train_residual_marl.py")
        return

    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    policy = TorchResidualPolicy(payload["obs_dim"], hidden_dim=payload["hidden_dim"])
    policy.residual_scale = payload["residual_scale"]
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    residual_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    residual = run_rollout(residual_env, policy=policy, explore_std=0.0)
    print("\n=== Residual policy rollout ===")
    print(residual)


if __name__ == "__main__":
    main()
