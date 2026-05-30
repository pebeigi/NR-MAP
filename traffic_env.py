"""2D multi-agent traffic environment with utility-based action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utility_model import (
    DEFAULT_BASE_PARAMS,
    DEFAULT_SIM_CONFIG,
    RESIDUAL_PARAM_KEYS,
    TrafficAgent,
    apply_residual,
    select_best_candidate,
)


@dataclass
class EnvConfig:
    dt: float = 0.5
    max_steps: int = 240
    num_agents: int = 10
    base_desired_speed: float = 8.0
    highway_length: float = 500.0
    spawn_x_range: tuple[float, float] = (0.0, 120.0)
    target_x_range: tuple[float, float] = (380.0, 500.0)
    min_initial_spacing: float = 4.0
    reward_weights: dict[str, float] | None = None
    sim_config: dict[str, Any] | None = None
    base_params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = {
                "progress": 1.0,
                "safety": 0.5,
                "smooth": 0.2,
                "traj": 0.0,
            }
        if self.sim_config is None:
            self.sim_config = dict(DEFAULT_SIM_CONFIG)
            self.sim_config["dt"] = self.dt
            self.sim_config.update(
                {
                    "road_y_min": -12.0,
                    "road_y_max": 12.0,
                    "road_x_min": 0.0,
                    "road_x_max": self.highway_length,
                    "boundary_buffer": 1.5,
                    "path_mode": "boundary",
                    "perception_radius": 60.0,
                    "max_neighbors": 6,
                    "max_agent_speed": 16.0,
                    "collision_threshold": 1.5,
                    "wheelbase": 2.8,
                    "candidate_accel_grid": [-3.0, -1.5, 0.0, 1.5, 3.0],
                    "candidate_steering_grid": [-0.35, -0.17, 0.0, 0.17, 0.35],
                    "steering_penalty_weight": 0.5,
                }
            )
        if self.base_params is None:
            self.base_params = dict(DEFAULT_BASE_PARAMS)


class MultiAgentTrafficEnv:
    """Decentralized multi-agent environment (Paper Eqs. 1-4, 16-24)."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.agents: list[TrafficAgent] = []
        self.step_count = 0
        self.collision_count = 0

    @property
    def obs_dim(self) -> int:
        # [x, y, v, theta, theta_goal, dist_to_lower_boundary, dist_to_upper_boundary]
        # + max_neighbors * [dx, dy, dvx, dvy]
        k = self.config.sim_config["max_neighbors"]
        return 7 + 4 * k

    @property
    def residual_dim(self) -> int:
        return len(RESIDUAL_PARAM_KEYS)

    def reset(self) -> list[np.ndarray]:
        self.step_count = 0
        self.collision_count = 0
        self.agents = self._spawn_agents()
        return [self.get_observation(i) for i in range(len(self.agents))]

    def _spawn_agents(self) -> list[TrafficAgent]:
        agents: list[TrafficAgent] = []
        n = self.config.num_agents
        base_v = self.config.base_desired_speed
        for i in range(n):
            pos = self._sample_start_position(agents)
            speed = max(1.0, self.rng.normal(base_v, 1.2))
            heading_noise = self.rng.normal(0.0, 0.08)
            vel = [speed * np.cos(heading_noise), speed * np.sin(heading_noise)]
            dest = [
                self.rng.uniform(*self.config.target_x_range),
                self.rng.uniform(
                    self.config.sim_config["road_y_min"] + 1.0,
                    self.config.sim_config["road_y_max"] - 1.0,
                ),
            ]
            agents.append(
                TrafficAgent(
                    agent_id=i,
                    pos=np.array(pos, dtype=float),
                    vel=np.array(vel, dtype=float),
                    dest=np.array(dest, dtype=float),
                    desired_speed=float(np.clip(self.rng.normal(base_v, 1.5), 4.0, 14.0)),
                    nominal_y=float(pos[1]),
                )
            )
        return agents

    def _sample_start_position(self, existing_agents: list[TrafficAgent]) -> list[float]:
        """Sample a lane-free highway start position with minimum spacing."""
        y_min = self.config.sim_config["road_y_min"] + 1.0
        y_max = self.config.sim_config["road_y_max"] - 1.0
        for _ in range(200):
            pos = np.array(
                [
                    self.rng.uniform(*self.config.spawn_x_range),
                    self.rng.uniform(y_min, y_max),
                ],
                dtype=float,
            )
            if all(np.linalg.norm(pos - agent.pos) >= self.config.min_initial_spacing for agent in existing_agents):
                return pos.tolist()
        return [self.rng.uniform(*self.config.spawn_x_range), self.rng.uniform(y_min, y_max)]

    def get_neighbors(self, agent_idx: int) -> list[int]:
        ego = self.agents[agent_idx]
        rp = self.config.sim_config["perception_radius"]
        neighbors: list[tuple[float, int]] = []
        for j, other in enumerate(self.agents):
            if j == agent_idx:
                continue
            d = float(np.linalg.norm(other.pos - ego.pos))
            if d <= rp:
                neighbors.append((d, j))
        neighbors.sort(key=lambda x: x[0])
        max_n = self.config.sim_config["max_neighbors"]
        return [j for _, j in neighbors[:max_n]]

    def get_observation(self, agent_idx: int) -> np.ndarray:
        """Paper Eq. 17: local decentralized observation."""
        ego = self.agents[agent_idx]
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = ego.pos[0]
        obs[1] = ego.pos[1]
        obs[2] = ego.speed
        obs[3] = ego.heading
        obs[4] = ego.goal_heading
        obs[5] = ego.pos[1] - self.config.sim_config["road_y_min"]
        obs[6] = self.config.sim_config["road_y_max"] - ego.pos[1]

        start = 7
        for j in self.get_neighbors(agent_idx):
            other = self.agents[j]
            obs[start : start + 4] = [
                other.pos[0] - ego.pos[0],
                other.pos[1] - ego.pos[1],
                other.vel[0] - ego.vel[0],
                other.vel[1] - ego.vel[1],
            ]
            start += 4
        return obs

    def _compute_reward(
        self,
        agent_idx: int,
        accel: np.ndarray,
        control: dict[str, float] | None = None,
    ) -> float:
        """Paper Eq. 20-24 with bicycle control smoothness."""
        ego = self.agents[agent_idx]
        w = self.config.reward_weights
        sim = self.config.sim_config
        steering_weight = float(sim.get("steering_penalty_weight", 0.5))

        r_progress = ego.speed * np.cos(ego.heading - ego.goal_heading)

        r_safety = 0.0
        for j in self.get_neighbors(agent_idx):
            d_ij = float(np.linalg.norm(self.agents[j].pos - ego.pos))
            r_safety -= np.exp(-d_ij)

        if control is not None:
            a = float(control.get("accel", 0.0))
            delta = float(control.get("steering", 0.0))
            r_smooth = -(a**2 + steering_weight * delta**2)
        else:
            r_smooth = -float(np.sum(accel**2))

        y = ego.pos[1]
        y_min = sim["road_y_min"]
        y_max = sim["road_y_max"]
        r_boundary = -10.0 if y < y_min or y > y_max else 0.0
        r_traj = 0.0  # placeholder until trajectory dataset is wired in

        return (
            w["progress"] * r_progress
            + w["safety"] * r_safety
            + w["smooth"] * r_smooth
            + r_boundary
            + w["traj"] * r_traj
        )

    def step(
        self,
        residual_actions: list[dict[str, float]] | None = None,
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """
        One synchronized simulation step.

        residual_actions: per-agent ΔΘ dicts from π(o). If None, use base utility only.
        """
        if residual_actions is None:
            residual_actions = [{} for _ in self.agents]

        intended_moves: list[dict[str, Any] | None] = []
        rewards: list[float] = []
        selected_controls: list[dict[str, float]] = []
        dt = self.config.sim_config["dt"]

        for i, agent in enumerate(self.agents):
            if agent.reached_destination:
                hold = {
                    "pos": agent.pos.copy(),
                    "vel": agent.vel.copy(),
                    "heading": agent.heading,
                    "speed": agent.speed,
                    "accel_longitudinal": 0.0,
                    "steering_angle": 0.0,
                    "time_to_reach": dt,
                }
                intended_moves.append(hold)
                selected_controls.append({"accel": 0.0, "steering": 0.0})
                rewards.append(0.0)
                continue

            params = apply_residual(self.config.base_params, residual_actions[i])
            chosen = select_best_candidate(i, agent, self.agents, params, self.config.sim_config)
            control = {
                "accel": float(chosen.get("accel_longitudinal", 0.0)),
                "steering": float(chosen.get("steering_angle", 0.0)),
            }
            selected_controls.append(control)
            rewards.append(self._compute_reward(i, agent.prev_accel, control=control))
            intended_moves.append(chosen)

        for i, agent in enumerate(self.agents):
            move = intended_moves[i]
            if move is not None:
                agent.update_state_from_candidate(
                    move,
                    dt,
                    self.config.sim_config["destination_threshold"],
                )

        self._check_collisions()
        self.step_count += 1

        observations = [self.get_observation(i) for i in range(len(self.agents))]
        done = self.step_count >= self.config.max_steps or all(a.reached_destination for a in self.agents)
        info = {
            "collision_count": self.collision_count,
            "steps": self.step_count,
            "destinations_reached": sum(a.reached_destination for a in self.agents),
            "selected_controls": selected_controls,
        }
        return observations, rewards, done, info

    def _check_collisions(self) -> None:
        threshold = self.config.sim_config["collision_threshold"]
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.agents[i].reached_destination or self.agents[j].reached_destination:
                    continue
                if np.linalg.norm(self.agents[i].pos - self.agents[j].pos) < threshold:
                    self.collision_count += 1

    def rollout_metric(self) -> float:
        """GSA-style episode metric: final distance + collision penalty."""
        total_dist = sum(
            float(np.linalg.norm(a.pos - a.dest))
            for a in self.agents
            if not a.reached_destination
        )
        penalty = 10.0 * self.collision_count
        return total_dist + penalty
