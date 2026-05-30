"""Gymnasium / RLlib multi-agent wrapper for the traffic simulator."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - RLlib not installed yet
    MultiAgentEnv = gym.Env  # type: ignore[misc,assignment]

from traffic_env import EnvConfig, MultiAgentTrafficEnv
from utility_model import (
    DEFAULT_RESIDUAL_SCALE,
    RESIDUAL_PARAM_KEYS,
    residual_vector_to_dict,
)


def agent_id(index: int) -> str:
    return f"agent_{index}"


def agent_index(agent_key: str) -> int:
    return int(agent_key.split("_")[1])


try:
    from gymnasium.envs.registration import register

    register(
        id="TrafficMAR-v0",
        entry_point="traffic_gym_env:TrafficMARLEnv",
        max_episode_steps=500,
    )
except Exception:
    # Registration can fail if called multiple times in the same interpreter.
    pass


class TrafficMARLEnv(MultiAgentEnv):
    """
    RLlib-compatible multi-agent environment.

    Each agent outputs a 7D residual vector ΔΘ (Paper Eq. 18). The environment
    applies utility maximization and kinematic updates internally.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None):
        env_config = env_config or {}
        self.residual_scale = float(env_config.get("residual_scale", DEFAULT_RESIDUAL_SCALE))
        self._cfg = EnvConfig(
            dt=float(env_config.get("dt", 0.5)),
            max_steps=int(env_config.get("max_steps", 240)),
            num_agents=int(env_config.get("num_agents", 10)),
            base_desired_speed=float(env_config.get("base_desired_speed", 8.0)),
            highway_length=float(env_config.get("highway_length", 500.0)),
            spawn_x_range=tuple(env_config.get("spawn_x_range", (0.0, 120.0))),
            target_x_range=tuple(env_config.get("target_x_range", (380.0, 500.0))),
            min_initial_spacing=float(env_config.get("min_initial_spacing", 4.0)),
        )
        self._env = MultiAgentTrafficEnv(self._cfg, seed=env_config.get("seed"))
        self._agent_ids = {agent_id(i) for i in range(self._cfg.num_agents)}

        obs_dim = self._env.obs_dim
        act_dim = len(RESIDUAL_PARAM_KEYS)
        high_obs = np.full(obs_dim, np.inf, dtype=np.float32)
        low_obs = -high_obs

        agent_list = sorted(self._agent_ids)
        self.observation_spaces = {
            aid: spaces.Box(low=low_obs, high=high_obs, dtype=np.float32) for aid in agent_list
        }
        self.action_spaces = {
            aid: spaces.Box(
                low=-self.residual_scale,
                high=self.residual_scale,
                shape=(act_dim,),
                dtype=np.float32,
            )
            for aid in agent_list
        }
        self.observation_space = self.observation_spaces[agent_list[0]]
        self.action_space = self.action_spaces[agent_list[0]]
        super().__init__()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        obs_list = self._env.reset()
        agent_list = sorted(self._agent_ids)
        observations = {agent_list[i]: obs_list[i] for i in range(len(obs_list))}
        infos = {aid: {} for aid in agent_list}
        return observations, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        residual_actions: list[dict[str, float]] = []
        agent_list = sorted(self._agent_ids)
        for i in range(self._cfg.num_agents):
            aid = agent_list[i]
            if aid in action_dict and action_dict[aid] is not None:
                vec = np.clip(action_dict[aid], -self.residual_scale, self.residual_scale)
                residual_actions.append(residual_vector_to_dict(vec))
            else:
                residual_actions.append({})

        obs_list, rewards_list, done, info = self._env.step(residual_actions)
        observations = {agent_list[i]: obs_list[i] for i in range(len(obs_list))}
        rewards = {agent_list[i]: float(rewards_list[i]) for i in range(len(rewards_list))}
        terminateds = {aid: False for aid in agent_list}
        truncateds = {aid: False for aid in agent_list}
        terminateds["__all__"] = done
        truncateds["__all__"] = done
        infos = {agent_list[i]: dict(info) for i in range(len(agent_list))}
        return observations, rewards, terminateds, truncateds, infos

    def rollout_metric(self) -> float:
        return self._env.rollout_metric()
