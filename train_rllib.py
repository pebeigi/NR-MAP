#!/usr/bin/env python
"""Train residual utility policy with RLlib PPO on the TrafficMAR-v0 environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from traffic_gym_env import TrafficMARLEnv


ENV_NAME = "TrafficMAR-v0"


def checkpoint_path(save_result) -> Path:
    """Extract filesystem path from RLlib save() return value."""
    if isinstance(save_result, (str, Path)):
        return Path(save_result)
    checkpoint = getattr(save_result, "checkpoint", save_result)
    if hasattr(checkpoint, "path"):
        return Path(checkpoint.path)
    if hasattr(checkpoint, "filesystem") and hasattr(checkpoint, "path"):
        return Path(checkpoint.path)
    raise TypeError(f"Unexpected checkpoint type: {type(save_result)}")


def env_creator(env_config: dict | None = None):
    return TrafficMARLEnv(env_config or {})


def load_calibrated_base_params(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    params = payload.get("best_params", payload)
    return {key: float(value) for key, value in params.items()}


def build_ppo_config(
    env_config: dict,
    num_agents: int,
    num_env_runners: int,
    train_batch_size: int,
    lr: float,
) -> PPOConfig:
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=TrafficMARLEnv,
            env_config=env_config,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length="auto",
            num_envs_per_env_runner=1,
        )
        .training(
            gamma=0.99,
            lr=lr,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            train_batch_size=train_batch_size,
            minibatch_size=min(256, train_batch_size),
            num_epochs=10,
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=2,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )


def baseline_metric(env_config: dict, episodes: int = 3) -> float:
    metrics = []
    for ep in range(episodes):
        env = TrafficMARLEnv({**env_config, "seed": ep})
        obs, _ = env.reset()
        done = False
        while not done:
            zero = {aid: np.zeros(env.action_spaces[aid].shape, dtype=np.float32) for aid in env.agents}
            obs, _, terminateds, truncateds, _ = env.step(zero)
            done = terminateds["__all__"] or truncateds["__all__"]
        metrics.append(env.rollout_metric())
    return float(np.mean(metrics))


def train(args: argparse.Namespace) -> Path:
    env_config = {
        "num_agents": args.num_agents,
        "max_steps": args.max_steps,
        "highway_length": args.highway_length,
        "residual_scale": args.residual_scale,
        "seed": args.seed,
    }
    calibrated_params = load_calibrated_base_params(args.calibration)
    if calibrated_params is not None:
        env_config["base_params"] = calibrated_params
        print(f"Using calibrated base utility parameters from {args.calibration}")

    tune.register_env(ENV_NAME, env_creator)
    config = build_ppo_config(
        env_config=env_config,
        num_agents=args.num_agents,
        num_env_runners=args.num_env_runners,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
    )

    base = baseline_metric(env_config, episodes=args.baseline_episodes)
    print(f"Utility-only baseline rollout metric (lower is better): {base:.3f}")

    algo = config.build_algo()
    best_metric = float("inf")
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        result = algo.train()
        env_runner = result.get("env_runners", {})
        episode_return_mean = env_runner.get("episode_return_mean", float("nan"))
        episode_len_mean = env_runner.get("episode_len_mean", float("nan"))

        eval_metrics = []
        eval_result = result.get("evaluation", {}).get("env_runners", {})
        if eval_result:
            # Negative return is used as proxy; lower rollout metric is better.
            pass

        if iteration % max(args.iterations // 10, 1) == 0 or iteration == 1:
            print(
                f"Iter {iteration:4d}/{args.iterations} | "
                f"episode_return_mean={episode_return_mean:8.3f} | "
                f"episode_len_mean={episode_len_mean:6.1f}"
            )

        if iteration % args.checkpoint_freq == 0 or iteration == args.iterations:
            ckpt = checkpoint_path(
                algo.save(str(checkpoint_dir / f"checkpoint_{iteration:05d}"))
            )
            print(f"Saved checkpoint: {ckpt}")

    final_ckpt = checkpoint_path(algo.save(str(checkpoint_dir / "checkpoint_final")))
    print(f"Training complete. Final checkpoint: {final_ckpt}")
    return final_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TrafficMAR with RLlib PPO")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--highway-length", type=float, default=500.0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--baseline-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional calibration JSON with best_params to use as Θ_base",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/rllib_ppo"),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
