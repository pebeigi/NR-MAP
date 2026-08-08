"""Shared kinematics, observations and reward used by every benchmarked model.

All controllers plug into the same bicycle integrator, the same oriented-box
collision test and the same corridor-progress arrival rule, so differences in
the metrics come from the policy and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import Baselines._paths  # noqa: F401
from RL.corridor import boundary_reward
from utility_model import TrafficAgent, kinematic_bicycle_rollout

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

MAX_STEERING = 0.45


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def project_and_clearances(corridor, point: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
    """Single corridor projection reused for both Frenet state and edge clearances."""
    s, lateral, tangent, seg_i, t = corridor.project(point)
    lower, upper = corridor.edge_points_at(seg_i, t)
    chord = upper - lower
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-6:
        return s, lateral, tangent, 0.0, 0.0
    unit = chord / chord_len
    mid = 0.5 * (lower + upper)
    signed = float((np.asarray(point, dtype=float) - mid) @ unit)
    half = 0.5 * chord_len
    return s, lateral, tangent, half + signed, half - signed


def neighbors_of(agents: list[TrafficAgent], idx: int, scenario: "Scenario") -> list[int]:
    """Nearest neighbours inside the perception radius (same rule as the RL env)."""
    ego = agents[idx]
    radius = float(scenario.sim_config["perception_radius"])
    max_n = int(scenario.sim_config["max_neighbors"])
    ranked: list[tuple[float, int]] = []
    for j, other in enumerate(agents):
        if j == idx:
            continue
        d = float(np.linalg.norm(other.pos - ego.pos))
        if d <= radius:
            ranked.append((d, j))
    ranked.sort(key=lambda x: x[0])
    return [j for _, j in ranked[:max_n]]


def observation(agents: list[TrafficAgent], idx: int, scenario: "Scenario") -> np.ndarray:
    """[x, y, v, psi, psi_goal, clear_low, clear_up] + k * [dx, dy, dvx, dvy]."""
    ego = agents[idx]
    max_n = int(scenario.sim_config["max_neighbors"])
    obs = np.zeros(7 + 4 * max_n, dtype=np.float32)
    _, _, _, c_lo, c_hi = project_and_clearances(scenario.corridor, ego.pos)
    obs[0] = ego.pos[0]
    obs[1] = ego.pos[1]
    obs[2] = ego.speed
    obs[3] = ego.heading
    obs[4] = ego.goal_heading
    obs[5] = c_lo
    obs[6] = c_hi
    start = 7
    for j in neighbors_of(agents, idx, scenario):
        other = agents[j]
        obs[start : start + 4] = [
            other.pos[0] - ego.pos[0],
            other.pos[1] - ego.pos[1],
            other.vel[0] - ego.vel[0],
            other.vel[1] - ego.vel[1],
        ]
        start += 4
    return obs


def observation_dim(scenario: "Scenario") -> int:
    return 7 + 4 * int(scenario.sim_config["max_neighbors"])


DEFAULT_REWARD_WEIGHTS = {"progress": 1.0, "safety": 0.5, "smooth": 0.2}


def compute_reward(
    agents: list[TrafficAgent],
    idx: int,
    scenario: "Scenario",
    control: tuple[float, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Mirror of the RL environment reward (progress / safety / smooth / boundary)."""
    w = weights or DEFAULT_REWARD_WEIGHTS
    ego = agents[idx]
    steering_weight = float(scenario.sim_config.get("steering_penalty_weight", 0.5))

    _, _, tangent, c_lo, c_hi = project_and_clearances(scenario.corridor, ego.pos)
    tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
    r_progress = ego.speed * np.cos(ego.heading - tangent_angle)

    r_safety = 0.0
    for j in neighbors_of(agents, idx, scenario):
        d_ij = float(np.linalg.norm(agents[j].pos - ego.pos))
        r_safety -= float(np.exp(-d_ij))

    accel, steering = control
    r_smooth = -(accel**2 + steering_weight * steering**2)

    r_boundary, _ = boundary_reward(c_lo, c_hi)

    return float(
        w.get("progress", 1.0) * r_progress
        + w.get("safety", 0.5) * r_safety
        + w.get("smooth", 0.2) * r_smooth
        + r_boundary
    )


def lookahead_point(
    agent: TrafficAgent,
    scenario: "Scenario",
    dest_s: float,
    lookahead: float = 15.0,
) -> np.ndarray:
    """
    Carrot point on the corridor ahead of the agent.

    The corridor is not lane-structured and arrival is defined by along-track
    station, so the agent holds its own lateral offset instead of being pulled
    onto the centreline.
    """
    s, lateral, _, _, _ = scenario.corridor.project(agent.pos)
    target_s = min(s + lookahead, dest_s)
    point, _ = scenario.corridor.xy_from_frenet(target_s, lateral)
    return point


def preferred_velocity(
    agent: TrafficAgent,
    scenario: "Scenario",
    dest_s: float,
    lookahead: float = 15.0,
) -> np.ndarray:
    """Goal-directed velocity that follows corridor curvature."""
    target = lookahead_point(agent, scenario, dest_s, lookahead)
    delta = target - agent.pos
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.zeros(2, dtype=float)
    s, _, _, _, _ = scenario.corridor.project(agent.pos)
    remaining = max(dest_s - s, 0.0)
    # Slow down smoothly over the last few metres so the agent stops at the goal.
    speed = min(agent.desired_speed, max(0.0, remaining) / max(scenario.dt, 1e-6))
    return (delta / norm) * speed


def velocity_to_control(
    agent: TrafficAgent,
    v_desired: np.ndarray,
    scenario: "Scenario",
    heading_gain: float = 1.0,
) -> tuple[float, float]:
    """Invert the bicycle model: desired velocity -> (accel, steering)."""
    dt = float(scenario.dt)
    sim = scenario.sim_config
    max_accel = float(sim.get("max_accel", 4.0))
    wheelbase = float(sim.get("wheelbase", 2.8))

    target_speed = float(np.linalg.norm(v_desired))
    accel = float(np.clip((target_speed - agent.speed) / max(dt, 1e-6), -max_accel, max_accel))

    if target_speed < 1e-3:
        return accel, 0.0

    target_heading = float(np.arctan2(v_desired[1], v_desired[0]))
    heading_error = wrap_angle(target_heading - agent.heading)
    yaw_rate = heading_gain * heading_error / max(dt, 1e-6)
    # psi_dot = (v / L) tan(delta)
    speed = max(agent.speed, 1e-3)
    steering = float(np.arctan(np.clip(yaw_rate * wheelbase / speed, -20.0, 20.0)))
    return accel, float(np.clip(steering, -MAX_STEERING, MAX_STEERING))


def apply_control(
    agent: TrafficAgent,
    control: tuple[float, float],
    scenario: "Scenario",
) -> dict[str, Any]:
    """Advance one agent by one step with the shared bicycle integrator."""
    accel, steering = control
    candidate = kinematic_bicycle_rollout(
        agent.pos,
        float(agent.heading),
        float(agent.speed),
        float(accel),
        float(np.clip(steering, -MAX_STEERING, MAX_STEERING)),
        float(scenario.dt),
        scenario.sim_config,
    )
    agent.update_state_from_candidate(
        candidate,
        float(scenario.dt),
        float(scenario.sim_config.get("destination_threshold", 1.0)),
    )
    return candidate


def simulate_bicycle_batch(
    pos: np.ndarray,
    heading: float,
    speed: float,
    accels: np.ndarray,
    steerings: np.ndarray,
    scenario: "Scenario",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out K control sequences with the same bicycle model used by the sim.

    `accels` and `steerings` are (K, H). Returns positions (K, H+1, 2), speeds
    (K, H+1) and headings (K, H+1).
    """
    sim = scenario.sim_config
    dt = float(scenario.dt)
    max_speed = float(sim["max_agent_speed"])
    max_accel = float(sim.get("max_accel", 4.0))
    wheelbase = float(sim.get("wheelbase", 2.8))

    k, horizon = accels.shape
    positions = np.empty((k, horizon + 1, 2), dtype=float)
    speeds = np.empty((k, horizon + 1), dtype=float)
    headings = np.empty((k, horizon + 1), dtype=float)

    positions[:, 0] = np.asarray(pos, dtype=float)
    speeds[:, 0] = float(speed)
    headings[:, 0] = float(heading)

    accels = np.clip(accels, -max_accel, max_accel)
    steerings = np.clip(steerings, -MAX_STEERING, MAX_STEERING)

    for h in range(horizon):
        v_prev = speeds[:, h]
        v_next = np.clip(v_prev + accels[:, h] * dt, 0.0, max_speed)
        yaw_rate = (v_prev / wheelbase) * np.tan(steerings[:, h])
        psi = headings[:, h] + yaw_rate * dt
        positions[:, h + 1, 0] = positions[:, h, 0] + v_next * np.cos(psi) * dt
        positions[:, h + 1, 1] = positions[:, h, 1] + v_next * np.sin(psi) * dt
        speeds[:, h + 1] = v_next
        headings[:, h + 1] = psi
    return positions, speeds, headings


def hold_still(agent: TrafficAgent) -> None:
    agent.vel[:] = 0.0
    agent.prev_accel[:] = 0.0
    agent.prev_control = {"accel": 0.0, "steering": 0.0}
