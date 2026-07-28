"""2D multi-agent traffic environment with utility-based action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import RL._paths  # noqa: F401 — put repo root on sys.path
from RL.calibration_io import RESIDUAL_PARAM_KEYS, apply_residual
from RL.corridor import (
    DEFAULT_LANE_KF,
    DEFAULT_RUN_ID,
    DEFAULT_VEHICLE_LENGTH,
    DEFAULT_VEHICLE_WIDTH,
    boxes_overlap,
    corridor_sim_defaults,
    load_corridor,
)
from utility_model import (
    DEFAULT_BASE_PARAMS,
    DEFAULT_SIM_CONFIG,
    TrafficAgent,
    select_best_candidate,
)


@dataclass
class EnvConfig:
    dt: float = 0.5
    max_steps: int = 240
    num_agents: int = 10
    base_desired_speed: float = 8.0
    highway_length: float = 500.0  # overridden by corridor length when available
    spawn_s_range: tuple[float, float] = (20.0, 120.0)
    # Destination station: random near the corridor end (absolute s, not offset from start).
    dest_s_range_from_end: tuple[float, float] = (5.0, 40.0)
    spawn_lateral_frac: float = 0.35  # fraction of half-width used at spawn
    min_initial_spacing: float = 8.0
    run_id: int = DEFAULT_RUN_ID
    lane_kf: int = DEFAULT_LANE_KF
    vehicle_length: float = DEFAULT_VEHICLE_LENGTH
    vehicle_width: float = DEFAULT_VEHICLE_WIDTH
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
        corridor = load_corridor(self.run_id, self.lane_kf)
        self.highway_length = float(corridor.length)
        if self.sim_config is None:
            self.sim_config = dict(DEFAULT_SIM_CONFIG)
            self.sim_config["dt"] = self.dt
            self.sim_config.update(corridor_sim_defaults(corridor))
            self.sim_config.update(
                {
                    "perception_radius": 60.0,
                    "max_neighbors": 6,
                    "max_agent_speed": 16.0,
                    "collision_threshold": 1.5,  # unused when OBB collisions enabled
                    "use_obb_collisions": True,
                    "wheelbase": 2.8,
                    "candidate_accel_grid": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                    "candidate_steering_grid": [
                        -0.45,
                        -0.3375,
                        -0.225,
                        -0.1125,
                        0.0,
                        0.1125,
                        0.225,
                        0.3375,
                        0.45,
                    ],
                    "steering_penalty_weight": 0.5,
                    "vehicle_length": self.vehicle_length,
                    "vehicle_width": self.vehicle_width,
                }
            )
        else:
            self.sim_config.setdefault("run_id", self.run_id)
            self.sim_config.setdefault("lane_kf", self.lane_kf)
            self.sim_config.setdefault("path_mode", "polyline")
            self.sim_config.setdefault("utility_frame", "corridor")
        if self.base_params is None:
            self.base_params = dict(DEFAULT_BASE_PARAMS)


class MultiAgentTrafficEnv:
    """Decentralized multi-agent environment on the measured highway corridor."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.corridor = load_corridor(self.config.run_id, self.config.lane_kf)
        self.rng = np.random.default_rng(seed)
        self.agents: list[TrafficAgent] = []
        self.step_count = 0
        self.collision_count = 0

    @property
    def obs_dim(self) -> int:
        # [x, y, v, theta, theta_goal, clearance_lower, clearance_upper]
        # + max_neighbors * [dx, dy, dvx, dvy]
        k = self.config.sim_config["max_neighbors"]
        return 7 + 4 * k

    @property
    def residual_dim(self) -> int:
        return len(RESIDUAL_PARAM_KEYS)

    def reset(self) -> list[np.ndarray]:
        self.step_count = 0
        self.collision_count = 0
        self._dest_s = []
        self.agents = self._spawn_agents()
        return [self.get_observation(i) for i in range(len(self.agents))]

    def _spawn_agents(self) -> list[TrafficAgent]:
        agents: list[TrafficAgent] = []
        n = self.config.num_agents
        base_v = self.config.base_desired_speed
        self._dest_s: list[float] = []
        for i in range(n):
            pos, tangent, s0 = self._sample_start_pose(agents)
            speed = max(1.0, self.rng.normal(base_v, 1.2))
            heading = float(np.arctan2(tangent[1], tangent[0]) + self.rng.normal(0.0, 0.05))
            vel = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)
            # Random destination near the end of the highway corridor.
            end_margin = float(self.rng.uniform(*self.config.dest_s_range_from_end))
            dest_s = max(s0 + 30.0, self.corridor.length - end_margin)
            dest_s = min(dest_s, self.corridor.length - 1.0)
            dest, _ = self.corridor.xy_from_frenet(dest_s, 0.0)
            self._dest_s.append(float(dest_s))
            agents.append(
                TrafficAgent(
                    agent_id=i,
                    pos=np.asarray(pos, dtype=float),
                    vel=vel,
                    dest=np.asarray(dest, dtype=float),
                    desired_speed=float(np.clip(self.rng.normal(base_v, 1.5), 4.0, 14.0)),
                    nominal_y=float(pos[1]),
                    run_id=self.config.run_id,
                    lane_kf=self.config.lane_kf,
                )
            )
        return agents

    def _sample_start_pose(
        self, existing_agents: list[TrafficAgent]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample a start pose inside the corridor with minimum spacing."""
        s_lo, s_hi = self.config.spawn_s_range
        s_hi = min(s_hi, max(s_lo + 1.0, self.corridor.length * 0.35))
        margin = 0.5 * float(self.config.sim_config.get("vehicle_width", self.config.vehicle_width))
        for _ in range(300):
            s = float(self.rng.uniform(s_lo, s_hi))
            # Probe local half-width
            mid, tangent = self.corridor.xy_from_frenet(s, 0.0)
            c_lo, c_hi, _ = self.corridor.clearances(mid)
            half = 0.5 * (c_lo + c_hi)
            max_lat = max(0.0, self.config.spawn_lateral_frac * half - margin)
            lateral = float(self.rng.uniform(-max_lat, max_lat)) if max_lat > 0 else 0.0
            pos, tangent = self.corridor.xy_from_frenet(s, lateral)
            if not self.corridor.inside(pos, margin=margin):
                continue
            if all(
                np.linalg.norm(pos - agent.pos) >= self.config.min_initial_spacing
                for agent in existing_agents
            ):
                return pos, tangent, s
        pos, tangent = self.corridor.xy_from_frenet(float(self.rng.uniform(s_lo, s_hi)), 0.0)
        return pos, tangent, float(self.corridor.project(pos)[0])

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
        """Local decentralized observation with corridor clearances."""
        ego = self.agents[agent_idx]
        c_lo, c_hi, _ = self.corridor.clearances(ego.pos)
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = ego.pos[0]
        obs[1] = ego.pos[1]
        obs[2] = ego.speed
        obs[3] = ego.heading
        obs[4] = ego.goal_heading
        obs[5] = c_lo
        obs[6] = c_hi

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
        ego = self.agents[agent_idx]
        w = self.config.reward_weights
        sim = self.config.sim_config
        steering_weight = float(sim.get("steering_penalty_weight", 0.5))

        _, _, tangent, _, _ = self.corridor.project(ego.pos)
        tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
        r_progress = ego.speed * np.cos(ego.heading - tangent_angle)

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

        c_lo, c_hi, _ = self.corridor.clearances(ego.pos)
        r_boundary = -10.0 if min(c_lo, c_hi) < 0.0 else 0.0
        r_traj = 0.0

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
            self._update_destination_flag(i)

        self._check_collisions()
        self.step_count += 1

        observations = [self.get_observation(i) for i in range(len(self.agents))]
        done = self.step_count >= self.config.max_steps or all(a.reached_destination for a in self.agents)
        info = {
            "collision_count": self.collision_count,
            "steps": self.step_count,
            "destinations_reached": sum(a.reached_destination for a in self.agents),
            "selected_controls": selected_controls,
            "run_id": self.config.run_id,
            "lane_kf": self.config.lane_kf,
        }
        return observations, rewards, done, info

    def _update_destination_flag(self, agent_idx: int) -> None:
        """
        Mark arrival by corridor progress: once along-track s reaches dest_s,
        stop the agent. Euclidean 1 m checks fail when cars are laterally offset
        from the centerline destination star.
        """
        agent = self.agents[agent_idx]
        if agent.reached_destination:
            agent.vel[:] = 0.0
            return
        s, _, _, _, _ = self.corridor.project(agent.pos)
        dest_s = self._dest_s[agent_idx] if agent_idx < len(getattr(self, "_dest_s", [])) else None
        if dest_s is None:
            dest_s = float(self.corridor.project(agent.dest)[0])
        tol = float(self.config.sim_config.get("destination_threshold", 1.0))
        # Arrive when we reach/pass the destination station (with small tolerance).
        if s >= dest_s - tol:
            agent.reached_destination = True
            agent.vel[:] = 0.0
            agent.prev_control = {"accel": 0.0, "steering": 0.0}

    def _check_collisions(self) -> None:
        sim = self.config.sim_config
        length = float(sim.get("vehicle_length", self.config.vehicle_length))
        width = float(sim.get("vehicle_width", self.config.vehicle_width))
        use_obb = bool(sim.get("use_obb_collisions", True))
        threshold = float(sim.get("collision_threshold", 1.5))
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.agents[i].reached_destination or self.agents[j].reached_destination:
                    continue
                if use_obb:
                    hit = boxes_overlap(
                        self.agents[i].pos,
                        self.agents[i].heading,
                        self.agents[j].pos,
                        self.agents[j].heading,
                        length=length,
                        width=width,
                    )
                else:
                    hit = np.linalg.norm(self.agents[i].pos - self.agents[j].pos) < threshold
                if hit:
                    self.collision_count += 1

    def rollout_metric(self) -> float:
        total_dist = sum(
            float(np.linalg.norm(a.pos - a.dest))
            for a in self.agents
            if not a.reached_destination
        )
        penalty = 10.0 * self.collision_count
        return total_dist + penalty
