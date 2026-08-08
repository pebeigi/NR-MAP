#!/usr/bin/env python
# coding: utf-8

import Sensitivity._paths  # noqa: F401 — repo root on sys.path

import numpy as np
import math

# --- Utility Function Components (from your Methodology) ---
# These functions will use the 'params' dictionary, which holds the GSA parameter set.

def _unit(vec):
    arr = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return None
    return arr / n


def _rotate2d(vec, angle_deg):
    """Rotate a 2D vector by angle_deg (counter-clockwise, degrees)."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=float)


def corridor_tangent_for_agent(agent):
    """Local corridor tangent estimate for sensitivity runs without map polylines."""
    dest_dir = _unit(np.asarray(agent.dest, dtype=float) - np.asarray(agent.pos, dtype=float))
    if dest_dir is not None:
        return dest_dir
    heading = _unit(agent.current_heading_vector)
    if heading is not None:
        return heading
    return np.array([1.0, 0.0], dtype=float)


def directional_alignment_utility(current_pos, candidate_pos, destination_pos, current_heading_vector, params):
    """
    Corridor-frame directional utility:
    U_dir = S_theta * (move_direction · corridor_tangent).
    Destination is only used to approximate the local corridor tangent when map geometry
    is unavailable in this sensitivity script.
    """
    move = np.asarray(candidate_pos, dtype=float) - np.asarray(current_pos, dtype=float)
    move_dir = _unit(move)
    if move_dir is None:
        return params["S_theta"]

    tangent = _unit(np.asarray(destination_pos, dtype=float) - np.asarray(current_pos, dtype=float))
    if tangent is None:
        tangent = _unit(current_heading_vector)
    if tangent is None:
        return 0.0
    return params["S_theta"] * float(np.dot(move_dir, tangent))


def speed_alignment_utility(
    current_speed_val,
    candidate_speed_val,
    leading_agent_speed,
    desired_speed,
    params,
    perception_horizon_empty,
):
    """
    Speed utility: reward candidate rollout speed close to desired speed.
    Symmetric ratio form used by utility_model.py.
    """
    if perception_horizon_empty:
        v_ref = desired_speed
    else:
        v_ref = leading_agent_speed if leading_agent_speed is not None else desired_speed

    v_cand = max(float(candidate_speed_val), 0.0)
    v_ref = max(float(v_ref), 1e-6)
    rho = min(v_cand, v_ref) / max(v_cand, v_ref)
    rho = max(rho, 1e-6)
    exponent = (params["xi_i"] - 1) / 2
    return params["S_v"] * (rho / (1 + rho**exponent))


def distance_reward_utility(candidate_pos, destination_pos, current_speed_val, params, sim_config, current_pos=None):
    """
    Proximity utility in a local Frenet frame estimated from ego→destination.
    d_eff = w_x |Δs| + w_y |Δn|, with reference at the destination projection.
    """
    ego = np.asarray(current_pos if current_pos is not None else [0.0, 0.0], dtype=float)
    cand = np.asarray(candidate_pos, dtype=float)
    dest = np.asarray(destination_pos, dtype=float)

    tangent = _unit(dest - ego)
    if tangent is None:
        tangent = np.array([1.0, 0.0], dtype=float)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)

    cand_s = float((cand - ego) @ tangent)
    cand_n = float((cand - ego) @ normal)
    ref_s = float((dest - ego) @ tangent)
    ref_n = float((dest - ego) @ normal)

    d_eff = params["w_x"] * abs(cand_s - ref_s) + params["w_y"] * abs(cand_n - ref_n)
    h_p = sim_config["kappa_perception_horizon"] * current_speed_val
    h_p = max(h_p, sim_config["min_perception_horizon"])
    if h_p < 1e-6:
        return 0.0
    return params["S_d"] / (1 + (d_eff / h_p) ** params["gamma"])

def collision_penalty(agent_i_idx, candidate_pos_agent_i, agents, params, sim_config, time_to_reach_candidate):
    """
    Calculates collision penalty.
    time_to_reach_candidate: time (in seconds) for agent_i to reach its candidate_pos.
    """
    P_i_g_sum = 0.0
    variances = np.array(sim_config['collision_pred_variances']) 
    
    det_Sigma_j = variances[0] * variances[1]
    if det_Sigma_j < 1e-18: 
        # This implies very high certainty. If candidate_pos_agent_i is near any predicted mu_j, penalty should be high.
        # For simplicity, if determinant is near zero, any overlap is maximum risk.
        # A more robust check would be needed here based on actual proximity if using this path.
        # For now, let's assume it doesn't happen or results in a high penalty if an agent is near.
        pass # Fall through, but exp term will likely dominate if Mahalanobis is small.

    # Ensure variances are not zero before inversion and sqrt for pdf_norm_factor
    if np.any(variances < 1e-9): # If any variance is effectively zero
         # Handle cases where prediction is too certain, making PDF problematic.
         # This might mean a very high penalty if candidate_pos is exactly where agent_j is predicted.
         # For now, we'll let it proceed, but inv_Sigma_j might have issues.
         # The 1e-9 in inv_Sigma_j helps, but very small det_Sigma_j is still an issue for sqrt.
         pass


    pdf_norm_factor = 1.0 / ((2 * np.pi)**(2/2) * np.sqrt(det_Sigma_j + 1e-18)) # d=2 (paper Eq. 10)
    inv_Sigma_j = np.diag(1.0 / (variances + 1e-9)) 

    for j, agent_j in enumerate(agents):
        if j == agent_i_idx:
            continue

        # Corrected: Use attribute access for Agent objects
        mu_j = np.array(agent_j.pos) + np.array(agent_j.vel) * time_to_reach_candidate
        
        diff_vec = np.array(candidate_pos_agent_i) - mu_j
        mahalanobis_dist_sq = diff_vec.T @ inv_Sigma_j @ diff_vec
        
        P_j_g = pdf_norm_factor * np.exp(-0.5 * mahalanobis_dist_sq)
        P_i_g_sum += P_j_g
        
    effective_P_i_g = min(P_i_g_sum, 1.0) 

    utility_penalty = params['w_c'] * effective_P_i_g
    return utility_penalty


def lane_plane_keeping_penalty(candidate_pos, nominal_path_point, params, agent_type=None):
    """Boundary adherence penalty (paper Eq. 13): U_path = w_p (1 - exp(-beta * l_i^2)).
    l_i is the lateral (y) deviation from the agent's nominal path."""
    ell_i = abs(candidate_pos[1] - nominal_path_point[1])
    penalty = params['w_ell'] * (1 - np.exp(-params['beta'] * ell_i**2))
    return penalty

# --- Agent and Simulation Logic ---
class Agent:
    def __init__(self, id, pos, vel, dest, type, desired_speed, current_heading_vector):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float) 
        self.dest = np.array(dest, dtype=float)
        self.type = type 
        self.desired_speed = desired_speed
        self.current_speed = np.linalg.norm(self.vel)
        # Ensure current_heading_vector is normalized during initialization
        norm_heading = np.linalg.norm(current_heading_vector)
        if norm_heading < 1e-9: # Handle zero vector if provided
            self.current_heading_vector = np.array([1,0], dtype=float) # Default heading
        else:
            self.current_heading_vector = np.array(current_heading_vector, dtype=float) / norm_heading
        
        self.history = [self.pos.copy()]
        self.reached_destination = False
        self.nominal_y = pos[1]  # nominal lateral (y) position for boundary adherence

    def update_state(self, new_pos, new_vel, dt):
        self.pos = np.array(new_pos, dtype=float)
        self.vel = np.array(new_vel, dtype=float)
        self.current_speed = np.linalg.norm(self.vel)
        if self.current_speed > 1e-6:
            self.current_heading_vector = self.vel / self.current_speed
        
        self.history.append(self.pos.copy())
        if np.linalg.norm(self.pos - self.dest) < sim_config_g['destination_threshold']:
            self.reached_destination = True


def generate_candidate_actions(agent, dt, sim_config):
    """
    Generates a heading x speed grid of candidate future actions for the agent.

    A coarse action set (e.g. just "maintain speed" / "go to desired speed" /
    "turn to destination" / "slow" / "stop", ~5 candidates total) makes almost
    every parameter sweep pick the same winning action, because the handful of
    candidates rarely differ enough in geometry for the utility weights to
    matter. That collapses the GSA output toward a near-deterministic Y and
    starves the Sobol indices of signal.

    Instead, we build a much richer candidate set by crossing several heading
    offsets (rotations, in degrees, around both the agent's current heading
    and its bearing to the destination) with several candidate speeds
    (fractions of the reference speed). This gives dozens of geometrically
    distinct rollouts per decision, so the directional/speed/distance/
    collision/lane-keeping weights can actually change which candidate wins -
    which is what a meaningful GSA needs.

    Returns a list of dicts: {'pos': candidate_pos, 'vel': candidate_vel, 'time_to_reach': dt}
    """
    current_pos = agent.pos
    current_vel = agent.vel
    current_speed = agent.current_speed
    max_speed = sim_config['max_agent_speed']
    desired_speed = agent.desired_speed

    # --- Reference directions: current heading and bearing to destination ---
    heading_dir = _unit(agent.current_heading_vector)
    if heading_dir is None:
        heading_dir = np.array([1.0, 0.0])

    vec_to_dest = agent.dest - current_pos
    dist_to_dest = float(np.linalg.norm(vec_to_dest))
    dest_dir = _unit(vec_to_dest) if dist_to_dest > 1e-6 else heading_dir

    base_dirs = [heading_dir]
    if not np.allclose(dest_dir, heading_dir, atol=1e-6):
        base_dirs.append(dest_dir)

    # Angular offsets (deg) applied around each base direction, and candidate
    # speeds as fractions of the reference speed. Configurable via sim_config
    # so richness (and runtime) can be tuned without touching this function.
    angle_offsets_deg = sim_config.get(
        'candidate_heading_offsets_deg', [0.0, 10.0, -10.0, 22.0, -22.0, 40.0, -40.0]
    )
    speed_fractions = sim_config.get(
        'candidate_speed_fractions', [0.0, 0.4, 0.7, 1.0, 1.3]
    )

    # Dedup via rounded-tuple keys (fast, O(n) hashing) rather than pairwise
    # np.allclose (O(n^2) with numpy call overhead) - this loop runs once per
    # agent per timestep, so its own cost must stay negligible next to the
    # utility evaluations it feeds.
    directions = []
    seen_dir_keys = set()
    for base in base_dirs:
        for offset in angle_offsets_deg:
            d = _rotate2d(base, offset)
            key = (round(float(d[0]), 3), round(float(d[1]), 3))
            if key not in seen_dir_keys:
                seen_dir_keys.add(key)
                directions.append(d)

    speed_ref = max(desired_speed, current_speed, 1e-6)
    speeds = sorted({round(min(f * speed_ref, max_speed), 6) for f in speed_fractions})

    raw_velocities = [d * v for d in directions for v in speeds]

    # Always keep "maintain exact current velocity" in the mix even if its
    # precise heading/speed does not land exactly on the sampled grid.
    if current_speed > 1e-3:
        raw_velocities.append(current_vel.copy())

    seen_vel_keys = set()
    final_candidates = []
    for vel in raw_velocities:
        vel = np.asarray(vel, dtype=float)
        speed_val = float(np.linalg.norm(vel))
        if speed_val > max_speed:
            vel = (vel / speed_val) * max_speed
        key = (round(float(vel[0]), 3), round(float(vel[1]), 3))
        if key in seen_vel_keys:
            continue
        seen_vel_keys.add(key)
        pos = current_pos + vel * dt
        final_candidates.append({'pos': pos, 'vel': vel, 'time_to_reach': dt})

    # If after all this, no candidates exist (e.g. agent at destination and
    # every rollout collapses to "stay"), ensure at least one exists.
    if not final_candidates:
        final_candidates.append({'pos': current_pos.copy(), 'vel': np.array([0.0, 0.0]), 'time_to_reach': dt})

    return final_candidates


# Global sim_config for access by agent methods if needed
sim_config_g = {}

def evaluate_simulation_model(parameter_set_list):
    """
    Main function to be called by SALib.
    parameter_set_list: A list of GSA parameters [S_theta, S_v, xi_i, S_d, gamma, w_x, w_y, w_c, w_ell, beta]
    """
    global sim_config_g 

    params = {
        'S_theta': parameter_set_list[0],
        'S_v': parameter_set_list[1], 'xi_i': parameter_set_list[2],
        'S_d': parameter_set_list[3], 'gamma': parameter_set_list[4],
        'w_x': parameter_set_list[5], 'w_y': parameter_set_list[6],
        'w_c': parameter_set_list[7],
        'w_ell': parameter_set_list[8], 'beta': parameter_set_list[9]
    }

    sim_config = {
        'dt': 0.5, 
        'total_time_steps': 100, 
        'num_agents': 3,
        'destination_threshold': 1.0, 
        'collision_threshold': 0.5, 
        'kappa_perception_horizon': 2.0, 
        'min_perception_horizon': 5.0,   
        'collision_pred_variances': [2.25**2, 0.9**2], 
        'max_agent_speed': 10.0, 
        'base_desired_speed': 5.0, 
        'collision_penalty_metric_factor': 10.0,
        # Richer candidate action grid: heading offsets (deg) rotated around
        # both the current heading and the destination bearing, crossed with
        # speed fractions of the reference speed. ~30-40 unique candidates
        # per decision instead of ~5, so the swept utility parameters can
        # actually change which action wins. Trim these lists (or lower
        # N_samples below) if runtime becomes a concern.
        'candidate_heading_offsets_deg': [0.0, 10.0, -10.0, 22.0, -22.0, 40.0, -40.0],
        'candidate_speed_fractions': [0.0, 0.4, 0.7, 1.0, 1.3],
    }
    sim_config_g = sim_config 

    agents = []
    agents.append(Agent(id=0, pos=[0,0], vel=[1,0], dest=[20,5], type='car', desired_speed=sim_config['base_desired_speed'], current_heading_vector=[1,0]))
    agents.append(Agent(id=1, pos=[2,5], vel=[0,1], dest=[15,-5], type='car', desired_speed=sim_config['base_desired_speed'] + 2, current_heading_vector=[0,1]))
    agents.append(Agent(id=2, pos=[5,-2], vel=[-1,0], dest=[-10,0], type='car', desired_speed=sim_config['base_desired_speed'] -1, current_heading_vector=[-1,0]))
    
    if sim_config['num_agents'] > 3: 
        for i in range(3, sim_config['num_agents']):
             agents.append(Agent(id=i, pos=[i*2, i*-2], vel=[1,0],
                                 dest=[20-i*3, 5+i*2],
                                 type='car',
                                 desired_speed=sim_config['base_desired_speed'] + (i%3 -1),
                                 current_heading_vector=[1,0]))

    collision_events_count = 0
    for t_step in range(sim_config['total_time_steps']):
        if all(agent.reached_destination for agent in agents):
            break

        # Store intended moves for all agents before updating any positions
        intended_moves = [None] * len(agents)

        for i, agent in enumerate(agents):
            if agent.reached_destination:
                intended_moves[i] = {'pos': agent.pos, 'vel': agent.vel, 'time_to_reach': sim_config['dt']} # Stay put
                continue

            candidate_actions = generate_candidate_actions(agent, sim_config['dt'], sim_config)
            
            best_utility = -float('inf')
            chosen_action_for_agent = candidate_actions[0] # Default to first action

            for action in candidate_actions:
                cand_pos = action['pos']
                cand_vel_vec = action['vel']
                cand_speed_val = np.linalg.norm(cand_vel_vec)
                time_to_reach = action['time_to_reach'] 

                util_dir = directional_alignment_utility(
                    agent.pos, cand_pos, agent.dest, agent.current_heading_vector, params
                )

                # Desired-speed alignment on the candidate rollout speed (current formulation).
                util_speed = speed_alignment_utility(
                    agent.current_speed,
                    cand_speed_val,
                    None,
                    agent.desired_speed,
                    params,
                    perception_horizon_empty=True,
                )

                util_dist = distance_reward_utility(
                    cand_pos,
                    agent.dest,
                    agent.current_speed,
                    params,
                    sim_config,
                    current_pos=agent.pos,
                )

                U_pos = util_dir + util_speed + util_dist  # paper Eq. 4: additive components

                penalty_coll = collision_penalty(i, cand_pos, agents, params, sim_config, time_to_reach)

                nominal_point = np.array([cand_pos[0], agent.nominal_y])  # nominal lateral (y) position

                penalty_lane = lane_plane_keeping_penalty(cand_pos, nominal_point, params, agent.type)

                total_utility = U_pos - penalty_coll - penalty_lane

                if total_utility > best_utility:
                    best_utility = total_utility
                    chosen_action_for_agent = action
            
            intended_moves[i] = chosen_action_for_agent
        
        # Update all agent states based on their chosen actions
        for i, agent in enumerate(agents):
            if intended_moves[i]:
                 agent.update_state(intended_moves[i]['pos'], intended_moves[i]['vel'], sim_config['dt'])


        # Check for collisions after all agents have moved in this time step
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Only check if agents are not at their destination (or very close)
                if not agents[i].reached_destination and not agents[j].reached_destination:
                    dist_ij = np.linalg.norm(agents[i].pos - agents[j].pos)
                    if dist_ij < sim_config['collision_threshold']:
                        collision_events_count += 1
                        # Optional: Mark agents as "collided" to stop them or alter behavior
                        # print(f"Collision between agent {agents[i].id} and {agents[j].id} at step {t_step}")


    total_final_distance = 0
    for agent in agents:
        if not agent.reached_destination:
            total_final_distance += np.linalg.norm(agent.pos - agent.dest)

    performance_metric_Y = total_final_distance + (collision_events_count * sim_config['collision_penalty_metric_factor'])

    return performance_metric_Y


# --- CLI entry: single-run smoke test or full Sobol GSA ---
def run_sobol_gsa(n_samples: int = 4096) -> None:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    import pandas as pd

    problem = {
        "num_vars": 10,
        "names": [
            "S_theta", "S_v", "xi_i", "S_d", "gamma",
            "w_x", "w_y", "w_c", "w_ell", "beta",
        ],
        "bounds": [
            [0.2, 1.0], [0.2, 1.0], [1.5, 3.5], [0.2, 1.0], [0.5, 3.0],
            [0.5, 5.0], [0.5, 5.0], [0.1, 20.0], [5.0, 50], [0.1, 5.0],
        ],
    }

    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)
    print(f"Generated {param_values.shape[0]} parameter samples for GSA.")

    Y_outputs = np.zeros(param_values.shape[0])
    print(f"Running {param_values.shape[0]} model evaluations...")
    for i, X_sample_list in enumerate(param_values):
        if (i + 1) % 100 == 0 or i == 0 or i == param_values.shape[0] - 1:
            print(f"GSA: Evaluating sample {i + 1}/{param_values.shape[0]}")
        Y_outputs[i] = evaluate_simulation_model(X_sample_list)
    print("Model evaluations complete.")

    Si = sobol.analyze(problem, Y_outputs, calc_second_order=False, print_to_console=False)
    print("\n--- Global Sensitivity Analysis Results ---")
    results_data = []
    for i, name in enumerate(problem["names"]):
        results_data.append({
            "Parameter": name,
            "S1": Si["S1"][i],
            "S1_conf": Si["S1_conf"][i],
            "ST": Si["ST"][i],
            "ST_conf": Si["ST_conf"][i],
        })
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False, float_format="%.4f"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NR-MAP utility global sensitivity analysis")
    parser.add_argument(
        "--mode",
        choices=("smoke", "sobol"),
        default="sobol",
        help="smoke: one example simulation; sobol: full SALib Sobol analysis",
    )
    parser.add_argument("--n-samples", type=int, default=4096, help="Sobol base sample size N")
    args = parser.parse_args()

    if args.mode == "smoke":
        example_params = [0.6, 0.6, 2.55, 0.6, 1.75, 2.75, 2.75, 2.75, 2.75, 1.05]
        print("Running single simulation with example parameters...")
        metric = evaluate_simulation_model(example_params)
        print(f"Simulation finished. Performance Metric (Y): {metric}")
        return

    run_sobol_gsa(n_samples=args.n_samples)


if __name__ == "__main__":
    main()
