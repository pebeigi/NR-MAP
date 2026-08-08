"""Utility-based behavioral prior for 2D multi-agent traffic (Paper Eqs. 4-13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

# Full utility parameter set used inside the simulator.
# sigma_long / sigma_lat are the collision-kernel std-devs (m), defaulted to
# vehicle half-length / half-width so avoidance has support at car scale.
UTILITY_PARAM_KEYS = (
    "S_theta",
    "S_v",
    "xi_i",
    "S_d",
    "gamma",
    "w_x",
    "w_y",
    "w_c",
    "w_ell",
    "beta",
    "sigma_long",
    "sigma_lat",
)

# Residual policy modulates these terms (weights + collision-kernel scales).
RESIDUAL_PARAM_KEYS = (
    "S_v",
    "S_theta",
    "S_d",
    "w_c",
    "xi_i",
    "gamma",
    "w_ell",
    "sigma_long",
    "sigma_lat",
)

DEFAULT_RESIDUAL_SCALE = 0.25

# Vehicle half-extents used as the default collision-kernel scale (m).
DEFAULT_SIGMA_LONG = 2.25  # 4.5 m length / 2
DEFAULT_SIGMA_LAT = 0.9  # 1.8 m width / 2
DEFAULT_KERNEL_PARAMS = {
    "sigma_long": DEFAULT_SIGMA_LONG,
    "sigma_lat": DEFAULT_SIGMA_LAT,
}


def residual_vector_to_dict(residual: np.ndarray) -> dict[str, float]:
    """Map a residual action vector to named utility-parameter deltas."""
    arr = np.asarray(residual, dtype=float).reshape(-1)
    if arr.size != len(RESIDUAL_PARAM_KEYS):
        raise ValueError(f"Expected {len(RESIDUAL_PARAM_KEYS)} residuals, got {arr.size}")
    return dict(zip(RESIDUAL_PARAM_KEYS, arr))


def normalize_observation(obs: np.ndarray, highway_length: float = 500.0) -> np.ndarray:
    """Normalize observations for neural policy inputs."""
    out = np.asarray(obs, dtype=np.float32).copy()
    out[0] /= highway_length
    out[1] /= 12.0
    out[2] /= 16.0
    out[3] /= np.pi
    out[4] /= np.pi
    out[5] /= 24.0
    out[6] /= 24.0
    for start in range(7, out.shape[0], 4):
        out[start] /= 60.0
        out[start + 1] /= 24.0
        out[start + 2] /= 16.0
        out[start + 3] /= 16.0
    return out

DEFAULT_BASE_PARAMS: dict[str, float] = {
    "S_theta": 0.6,
    "S_v": 0.6,
    "xi_i": 2.55,
    "S_d": 0.6,
    "gamma": 1.75,
    "w_x": 2.75,
    "w_y": 2.75,
    "w_c": 2.75,
    "w_ell": 2.75,
    "beta": 1.05,
    "sigma_long": DEFAULT_SIGMA_LONG,
    "sigma_lat": DEFAULT_SIGMA_LAT,
}

DEFAULT_SIM_CONFIG: dict[str, Any] = {
    "dt": 0.5,
    "destination_threshold": 1.0,
    "collision_threshold": 0.5,
    "kappa_perception_horizon": 2.0,
    "min_perception_horizon": 5.0,
    # Variances = sigma^2 at vehicle half-extents (fallback if Θ has no sigma_*).
    "collision_pred_variances": [DEFAULT_SIGMA_LONG**2, DEFAULT_SIGMA_LAT**2],
    "max_agent_speed": 10.0,
    "max_accel": 4.0,
    "perception_radius": 40.0,
    "max_neighbors": 5,
    "road_y_min": -12.0,
    "road_y_max": 12.0,
    "path_mode": "boundary",
    "wheelbase": 2.8,
    "candidate_accel_grid": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "candidate_steering_grid": [-0.45, -0.3375, -0.225, -0.1125, 0.0, 0.1125, 0.225, 0.3375, 0.45],
    "steering_penalty_weight": 0.5,
    "utility_frame": "corridor",
}


def params_from_vector(values: list[float] | np.ndarray) -> dict[str, float]:
    """Map a 10-element vector to the full utility parameter dict."""
    arr = np.asarray(values, dtype=float)
    if arr.size != len(UTILITY_PARAM_KEYS):
        raise ValueError(f"Expected {len(UTILITY_PARAM_KEYS)} values, got {arr.size}")
    return dict(zip(UTILITY_PARAM_KEYS, arr))


def clip_params(params: dict[str, float]) -> dict[str, float]:
    """Keep utility parameters in physically valid ranges."""
    bounds = {
        "S_theta": (0.05, 2.0),
        "S_v": (0.05, 2.0),
        "xi_i": (1.1, 5.0),
        "S_d": (0.05, 2.0),
        "gamma": (0.1, 5.0),
        "w_x": (0.1, 10.0),
        "w_y": (0.1, 10.0),
        "w_c": (0.01, 50.0),
        "w_ell": (0.1, 100.0),
        "beta": (0.01, 10.0),
        "sigma_long": (0.3, 6.0),
        "sigma_lat": (0.2, 3.0),
    }
    filled = {**DEFAULT_KERNEL_PARAMS, **params}
    return {k: float(np.clip(filled[k], *bounds[k])) for k in UTILITY_PARAM_KEYS}


def apply_residual(
    base_params: dict[str, float],
    delta_theta: dict[str, float],
) -> dict[str, float]:
    """Θ_i = Θ_base + ΔΘ_i."""
    merged = {**DEFAULT_KERNEL_PARAMS, **base_params}
    for key in RESIDUAL_PARAM_KEYS:
        merged[key] = float(merged.get(key, 0.0)) + float(delta_theta.get(key, 0.0))
    return clip_params(merged)


def collision_variances(
    params: dict[str, float] | None,
    sim_config: dict[str, Any],
) -> np.ndarray:
    """Collision-kernel variances (m^2): prefer Θ.sigma_* over sim_config fallback."""
    if params is not None and "sigma_long" in params and "sigma_lat" in params:
        return np.array(
            [float(params["sigma_long"]) ** 2, float(params["sigma_lat"]) ** 2],
            dtype=float,
        )
    return np.asarray(sim_config["collision_pred_variances"], dtype=float)


@dataclass(frozen=True)
class CorridorGeometry:
    center: np.ndarray
    cumulative_s: np.ndarray
    tangents: np.ndarray

    @classmethod
    def from_center(cls, center: np.ndarray) -> "CorridorGeometry | None":
        center = np.asarray(center, dtype=float)
        if center.shape[0] < 2:
            return None
        seg = center[1:] - center[:-1]
        seg_lens = np.linalg.norm(seg, axis=1)
        cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        tangents = seg / np.maximum(seg_lens[:, None], 1e-12)
        return cls(center=center, cumulative_s=cumulative_s, tangents=tangents)

    def project(self, point: np.ndarray) -> tuple[float, float, np.ndarray]:
        point = np.asarray(point, dtype=float)
        d2 = np.sum((self.center - point) ** 2, axis=1)
        i0 = int(np.argmin(d2))
        i_lo = max(0, i0 - 2)
        i_hi = min(len(self.center) - 2, i0 + 2)
        best_dist = float("inf")
        best_s = 0.0
        best_lateral = 0.0
        best_tangent = self.tangents[max(0, min(i0, len(self.tangents) - 1))]
        for i in range(i_lo, i_hi + 1):
            a = self.center[i]
            b = self.center[i + 1]
            ab = b - a
            denom = float(ab @ ab) + 1e-12
            t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
            q = a + t * ab
            dist = float(np.linalg.norm(point - q))
            if dist < best_dist:
                tangent = self.tangents[i]
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                best_dist = dist
                best_s = float(self.cumulative_s[i] + t * np.linalg.norm(ab))
                best_lateral = float((point - q) @ normal)
                best_tangent = tangent
        return best_s, best_lateral, best_tangent


@lru_cache(maxsize=32)
def corridor_geometry_for_run_lane(run_id: int, lane_kf: int) -> CorridorGeometry | None:
    try:
        from data.highway_geometry import center_polyline, load_highway_boundaries

        load_highway_boundaries()
        center = center_polyline(run_id, lane_kf)
    except Exception:
        center = None
    if center is None:
        return None
    return CorridorGeometry.from_center(center)


def corridor_directional_alignment_utility(
    current_pos: np.ndarray,
    candidate_pos: np.ndarray,
    tangent_at_ego: np.ndarray,
    params: dict[str, float],
) -> float:
    """Reward moving along the local corridor tangent."""
    move = np.asarray(candidate_pos, dtype=float) - np.asarray(current_pos, dtype=float)
    move_norm = float(np.linalg.norm(move))
    if move_norm < 1e-6:
        return params["S_theta"]
    move_dir = move / move_norm
    tangent = np.asarray(tangent_at_ego, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-6:
        return 0.0
    tangent = tangent / tangent_norm
    return params["S_theta"] * float(np.dot(move_dir, tangent))


def corridor_distance_reward_utility(
    candidate_s: float,
    candidate_lateral: float,
    reference_s: float,
    reference_lateral: float,
    current_speed: float,
    params: dict[str, float],
    sim_config: dict[str, Any],
) -> float:
    """Reward Frenet coordinates close to a local reference point on the corridor."""
    d_long = abs(candidate_s - reference_s)
    d_lat = abs(candidate_lateral - reference_lateral)
    d_eff = params["w_x"] * d_long + params["w_y"] * d_lat
    h_p = sim_config["kappa_perception_horizon"] * current_speed
    h_p = max(h_p, sim_config["min_perception_horizon"])
    if h_p < 1e-6:
        return 0.0
    ratio = d_eff / h_p
    return params["S_d"] / (1 + ratio ** params["gamma"])


def directional_alignment_utility(
    current_pos: np.ndarray,
    candidate_pos: np.ndarray,
    destination_pos: np.ndarray,
    current_heading_vector: np.ndarray,
    params: dict[str, float],
) -> float:
    """Paper Eq. 5."""
    vec_to_candidate = np.array(candidate_pos) - np.array(current_pos)
    dist_to_candidate = np.linalg.norm(vec_to_candidate)
    if dist_to_candidate < 1e-6:
        vec_current_to_dest = np.array(destination_pos) - np.array(current_pos)
        if np.linalg.norm(vec_current_to_dest) < 1e-6:
            return params["S_theta"]
        dir_to_dest = vec_current_to_dest / np.linalg.norm(vec_current_to_dest)
        return params["S_theta"] * float(np.dot(current_heading_vector, dir_to_dest))

    new_heading = vec_to_candidate / dist_to_candidate
    vec_candidate_to_dest = np.array(destination_pos) - np.array(candidate_pos)
    dist_candidate_to_dest = np.linalg.norm(vec_candidate_to_dest)
    if dist_candidate_to_dest < 1e-6:
        return params["S_theta"]

    dir_to_dest = vec_candidate_to_dest / dist_candidate_to_dest
    cos_theta = float(np.dot(new_heading, dir_to_dest))
    return params["S_theta"] * cos_theta


def speed_alignment_utility(
    candidate_speed: float,
    leading_agent_speed: float | None,
    desired_speed: float,
    params: dict[str, float],
    perception_horizon_empty: bool,
) -> float:
    """Reward candidate rollout speed close to desired speed (symmetric ratio form)."""
    if perception_horizon_empty:
        v_ref = desired_speed
    else:
        v_ref = leading_agent_speed if leading_agent_speed is not None else desired_speed

    v_cand = max(candidate_speed, 0.0)
    v_ref = max(v_ref, 1e-6)
    rho = min(v_cand, v_ref) / max(v_cand, v_ref)
    rho = max(rho, 1e-6)
    exponent = (params["xi_i"] - 1) / 2
    return params["S_v"] * (rho / (1 + rho**exponent))


def distance_reward_utility(
    candidate_pos: np.ndarray,
    destination_pos: np.ndarray,
    current_speed: float,
    params: dict[str, float],
    sim_config: dict[str, Any],
) -> float:
    """Paper Eq. 7."""
    diff = np.abs(np.array(candidate_pos) - np.array(destination_pos))
    d_eff = params["w_x"] * diff[0] + params["w_y"] * diff[1]

    h_p = sim_config["kappa_perception_horizon"] * current_speed
    h_p = max(h_p, sim_config["min_perception_horizon"])
    if h_p < 1e-6:
        return 0.0

    ratio = d_eff / h_p
    return params["S_d"] / (1 + ratio ** params["gamma"])


def collision_penalty(
    agent_i_idx: int,
    candidate_pos: np.ndarray,
    agents: list["TrafficAgent"],
    params: dict[str, float],
    sim_config: dict[str, Any],
    time_to_reach: float,
) -> float:
    """Paper Eqs. 8-12. Kernel width comes from Θ.sigma_* (vehicle-scale by default)."""
    p_sum = 0.0
    variances = collision_variances(params, sim_config)
    det_sigma = variances[0] * variances[1]
    pdf_norm = 1.0 / ((2 * np.pi) ** 1 * np.sqrt(det_sigma + 1e-18))
    inv_sigma = np.diag(1.0 / (variances + 1e-9))

    for j, agent_j in enumerate(agents):
        if j == agent_i_idx:
            continue
        mu_j = np.array(agent_j.pos) + np.array(agent_j.vel) * time_to_reach
        diff_vec = np.array(candidate_pos) - mu_j
        mahal_sq = float(diff_vec.T @ inv_sigma @ diff_vec)
        p_sum += pdf_norm * np.exp(-0.5 * mahal_sq)

    return params["w_c"] * min(p_sum, 1.0)


def path_adherence_penalty(
    candidate_pos: np.ndarray,
    nominal_y: float,
    params: dict[str, float],
    sim_config: dict[str, Any] | None = None,
) -> float:
    """Paper Eq. 13."""
    if sim_config is not None and sim_config.get("path_mode") == "polyline":
        # Data-derived highway envelope (run_id, lane_kf).
        try:
            from RL.corridor import load_corridor

            corridor = load_corridor(
                int(sim_config["run_id"]),
                int(sim_config["lane_kf"]),
            )
            ell_i = corridor.path_error(
                candidate_pos,
                boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
            )
        except Exception:
            ell_i = abs(float(candidate_pos[1]) - float(nominal_y))
    elif sim_config is not None and sim_config.get("path_mode") == "boundary":
        y = float(candidate_pos[1])
        y_min = float(sim_config["road_y_min"])
        y_max = float(sim_config["road_y_max"])
        boundary_buffer = float(sim_config.get("boundary_buffer", 1.5))
        if y < y_min:
            ell_i = y_min - y + boundary_buffer
        elif y > y_max:
            ell_i = y - y_max + boundary_buffer
        else:
            ell_i = max(0.0, boundary_buffer - min(y - y_min, y_max - y))
    else:
        ell_i = abs(candidate_pos[1] - nominal_y)
    return params["w_ell"] * (1 - np.exp(-params["beta"] * ell_i**2))


@dataclass
class TrafficAgent:
    agent_id: int
    pos: np.ndarray
    vel: np.ndarray
    dest: np.ndarray
    desired_speed: float
    nominal_y: float
    run_id: int | None = None
    lane_kf: int | None = None
    utility_reference_pos: np.ndarray | None = None
    heading_angle: float | None = None
    current_heading_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    reached_destination: bool = False
    prev_accel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev_control: dict[str, float] = field(default_factory=lambda: {"accel": 0.0, "steering": 0.0})

    def __post_init__(self) -> None:
        if self.heading_angle is None:
            speed = float(np.linalg.norm(self.vel))
            if speed > 1e-6:
                self.heading_angle = float(np.arctan2(self.vel[1], self.vel[0]))
            else:
                vec = self.dest - self.pos
                if np.linalg.norm(vec) > 1e-6:
                    self.heading_angle = float(np.arctan2(vec[1], vec[0]))
                else:
                    self.heading_angle = 0.0
        self._sync_heading_vector()

    def _sync_heading_vector(self) -> None:
        psi = float(self.heading_angle)
        self.current_heading_vector = np.array([np.cos(psi), np.sin(psi)], dtype=float)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def heading(self) -> float:
        return float(self.heading_angle)

    @property
    def goal_heading(self) -> float:
        vec = self.dest - self.pos
        if np.linalg.norm(vec) < 1e-6:
            return self.heading
        return float(np.arctan2(vec[1], vec[0]))

    def update_state_from_candidate(
        self,
        candidate: dict[str, Any],
        dt: float,
        dest_threshold: float,
    ) -> None:
        """Apply utility-selected bicycle control candidate."""
        old_vel = np.array(self.vel, dtype=float)
        self.pos = np.array(candidate["pos"], dtype=float)
        self.heading_angle = float(candidate["heading"])
        self.vel = np.array(candidate["vel"], dtype=float)
        self._sync_heading_vector()
        self.prev_accel = (self.vel - old_vel) / max(dt, 1e-6)
        self.prev_control = {
            "accel": float(candidate.get("accel_longitudinal", 0.0)),
            "steering": float(candidate.get("steering_angle", 0.0)),
        }
        if np.linalg.norm(self.pos - self.dest) < dest_threshold:
            self.reached_destination = True

    def update_state(self, new_pos: np.ndarray, new_vel: np.ndarray, dt: float, dest_threshold: float) -> None:
        """Legacy direct pos/vel update (kept for compatibility)."""
        accel = (np.array(new_vel) - np.array(self.vel)) / max(dt, 1e-6)
        self.prev_accel = accel
        self.pos = np.array(new_pos, dtype=float)
        self.vel = np.array(new_vel, dtype=float)
        speed = self.speed
        if speed > 1e-6:
            self.heading_angle = float(np.arctan2(self.vel[1], self.vel[0]))
            self._sync_heading_vector()
        if np.linalg.norm(self.pos - self.dest) < dest_threshold:
            self.reached_destination = True


def kinematic_bicycle_rollout(
    pos: np.ndarray,
    heading: float,
    speed: float,
    accel: float,
    steering: float,
    dt: float,
    sim_config: dict[str, Any],
) -> dict[str, Any]:
    """
    One-step kinematic bicycle model.

    v_next = clip(v + a dt, 0, v_max)
    psi_next = psi + (v / L) tan(delta) dt
    x_next = x + v_next cos(psi_next) dt
    y_next = y + v_next sin(psi_next) dt
    """
    wheelbase = float(sim_config.get("wheelbase", 2.8))
    max_speed = float(sim_config["max_agent_speed"])
    max_accel = float(sim_config.get("max_accel", 4.0))
    accel_clipped = float(np.clip(accel, -max_accel, max_accel))

    v_next = float(np.clip(speed + accel_clipped * dt, 0.0, max_speed))
    yaw_rate = (speed / max(wheelbase, 1e-6)) * np.tan(steering)
    psi_next = float(heading + yaw_rate * dt)
    x_next = float(pos[0] + v_next * np.cos(psi_next) * dt)
    y_next = float(pos[1] + v_next * np.sin(psi_next) * dt)
    vel = np.array([v_next * np.cos(psi_next), v_next * np.sin(psi_next)], dtype=float)

    return {
        "accel_longitudinal": accel_clipped,
        "steering_angle": float(steering),
        "pos": np.array([x_next, y_next], dtype=float),
        "vel": vel,
        "heading": psi_next,
        "speed": v_next,
        "time_to_reach": dt,
    }


def generate_candidate_actions(
    agent: TrafficAgent,
    dt: float,
    sim_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Discrete acceleration/steering candidates via kinematic bicycle rollout."""
    accel_grid = sim_config.get("candidate_accel_grid", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    steering_grid = sim_config.get(
        "candidate_steering_grid",
        [-0.45, -0.3375, -0.225, -0.1125, 0.0, 0.1125, 0.225, 0.3375, 0.45],
    )

    current_pos = agent.pos
    current_heading = float(agent.heading_angle)
    current_speed = agent.speed

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for accel in accel_grid:
        for steering in steering_grid:
            cand = kinematic_bicycle_rollout(
                current_pos,
                current_heading,
                current_speed,
                float(accel),
                float(steering),
                dt,
                sim_config,
            )
            key = (
                round(float(cand["pos"][0]), 4),
                round(float(cand["pos"][1]), 4),
                round(float(cand["speed"]), 4),
                round(float(cand["heading"]), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

    if not candidates:
        candidates.append(
            kinematic_bicycle_rollout(
                current_pos, current_heading, current_speed, 0.0, 0.0, dt, sim_config
            )
        )
    return candidates


def evaluate_candidate_utility(
    agent_idx: int,
    agent: TrafficAgent,
    candidate: dict[str, Any],
    agents: list[TrafficAgent],
    params: dict[str, float],
    sim_config: dict[str, Any],
) -> float:
    """Total utility for one candidate action (Paper Eq. 4)."""
    cand_pos = candidate["pos"]
    cand_vel = candidate["vel"]
    cand_speed = float(candidate.get("speed", np.linalg.norm(cand_vel)))
    time_to_reach = candidate["time_to_reach"]

    use_corridor = sim_config.get("utility_frame", "destination") == "corridor"
    corridor_geom = None
    if use_corridor and agent.run_id is not None and agent.lane_kf is not None:
        corridor_geom = corridor_geometry_for_run_lane(int(agent.run_id), int(agent.lane_kf))

    if corridor_geom is not None:
        _, _, tangent_at_ego = corridor_geom.project(agent.pos)
        util_dir = corridor_directional_alignment_utility(agent.pos, cand_pos, tangent_at_ego, params)
        reference_pos = agent.utility_reference_pos
        if reference_pos is None:
            reference_pos = agent.dest
        ref_s, ref_n, _ = corridor_geom.project(reference_pos)
        cand_s, cand_n, _ = corridor_geom.project(cand_pos)
        util_dist = corridor_distance_reward_utility(
            cand_s, cand_n, ref_s, ref_n, agent.speed, params, sim_config
        )
    else:
        util_dir = directional_alignment_utility(
            agent.pos, cand_pos, agent.dest, agent.current_heading_vector, params
        )
        util_dist = distance_reward_utility(cand_pos, agent.dest, agent.speed, params, sim_config)

    util_speed = speed_alignment_utility(
        cand_speed,
        None,
        agent.desired_speed,
        params,
        perception_horizon_empty=True,
    )
    penalty_coll = collision_penalty(agent_idx, cand_pos, agents, params, sim_config, time_to_reach)
    penalty_path = path_adherence_penalty(cand_pos, agent.nominal_y, params, sim_config)

    return util_dir + util_speed + util_dist - penalty_coll - penalty_path


def select_best_candidate(
    agent_idx: int,
    agent: TrafficAgent,
    agents: list[TrafficAgent],
    params: dict[str, float],
    sim_config: dict[str, Any],
) -> dict[str, Any]:
    """argmax_a U(a; Θ) over discrete candidate actions."""
    candidates = generate_candidate_actions(agent, sim_config["dt"], sim_config)
    best = candidates[0]
    best_u = -float("inf")
    for cand in candidates:
        u = evaluate_candidate_utility(agent_idx, agent, cand, agents, params, sim_config)
        if u > best_u:
            best_u = u
            best = cand
    return best
