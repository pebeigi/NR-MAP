#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math

# --- Utility Function Components (from your Methodology) ---
# These functions will use the 'params' dictionary, which holds the GSA parameter set.

def directional_alignment_utility(current_pos, candidate_pos, destination_pos, current_heading_vector, params):
    """
    Directional utility (paper Eq. 5): U_dir = S_theta * cos(theta - theta_goal).
    cos(theta - theta_goal) is the alignment between the candidate heading and the
    direction from the candidate position toward the destination.
    """
    vec_to_candidate = np.array(candidate_pos) - np.array(current_pos)
    dist_to_candidate = np.linalg.norm(vec_to_candidate)
    if dist_to_candidate < 1e-6: # Not moving: align current heading with direction to destination
        vec_current_to_dest = np.array(destination_pos) - np.array(current_pos)
        if np.linalg.norm(vec_current_to_dest) < 1e-6: # Already at destination
            return params['S_theta']

        dir_to_dest_from_current = vec_current_to_dest / np.linalg.norm(vec_current_to_dest)
        cos_theta = np.dot(current_heading_vector, dir_to_dest_from_current)
        return params['S_theta'] * cos_theta

    new_heading_vector = vec_to_candidate / dist_to_candidate

    # Vector from the candidate position to the final destination
    vec_candidate_to_dest = np.array(destination_pos) - np.array(candidate_pos)
    dist_candidate_to_dest = np.linalg.norm(vec_candidate_to_dest)

    if dist_candidate_to_dest < 1e-6: # Candidate is destination
        return params['S_theta']

    dir_to_dest_from_candidate = vec_candidate_to_dest / dist_candidate_to_dest

    # cos(theta - theta_goal): plain cosine, kept signed as in the paper (Eq. 5).
    cos_theta = np.dot(new_heading_vector, dir_to_dest_from_candidate)
    return params['S_theta'] * cos_theta


def speed_alignment_utility(current_speed_val, candidate_speed_val, leading_agent_speed, desired_speed, params, perception_horizon_empty):
    """Calculates speed alignment utility."""
    if perception_horizon_empty:
        v_ref_g = desired_speed
    else:
        v_ref_g = leading_agent_speed if leading_agent_speed is not None else desired_speed

    # v_i_current here refers to the speed the agent *would have* if it takes the action leading to candidate_speed_val
    # For simplicity in this conceptual model, we can use candidate_speed_val directly as the speed post-action.
    # If the action implies a change from current_speed_val to candidate_speed_val,
    # the utility should reflect the desirability of candidate_speed_val.
    v_agent_at_candidate = candidate_speed_val

    if abs(v_agent_at_candidate) < 1e-6: 
        if abs(v_ref_g) < 1e-6: rho_g = 1.0 
        else: rho_g = 1000.0 
    else:
        rho_g = v_ref_g / v_agent_at_candidate
    
    rho_g = max(1e-6, rho_g) 

    exponent = (params['xi_i'] - 1) / 2
    utility = params['S_v'] * (rho_g / (1 + rho_g**exponent))
    return utility

def distance_reward_utility(candidate_pos, destination_pos, current_speed_val, params, sim_config):
    """Calculates distance reward utility."""
    diff = np.abs(np.array(candidate_pos) - np.array(destination_pos))
    d_eff = params['w_x'] * diff[0] + params['w_y'] * diff[1]

    H_p = sim_config['kappa_perception_horizon'] * current_speed_val
    H_p = max(H_p, sim_config['min_perception_horizon']) 

    if H_p < 1e-6: 
        return 0 

    ratio_d_eff_H_p = d_eff / H_p
    
    utility = params['S_d'] / (1 + ratio_d_eff_H_p**params['gamma'])
    return utility

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
    Generates a few candidate future positions for the agent.
    Returns a list of dicts: {'pos': candidate_pos, 'vel': candidate_vel, 'time_to_reach': dt}
    """
    candidates = []
    current_pos = agent.pos
    current_vel = agent.vel
    current_speed = agent.current_speed
    
    # Option 1: Maintain current velocity (if speed > 0)
    if current_speed > 1e-3: # Only if actually moving
        cand_vel1 = current_vel
        cand_pos1 = current_pos + cand_vel1 * dt
        candidates.append({'pos': cand_pos1, 'vel': cand_vel1, 'time_to_reach': dt})

    # Option 2: Accelerate/Decelerate towards desired_speed along current heading
    # This action implies changing speed while maintaining current direction.
    # The candidate velocity's magnitude will be agent.desired_speed.
    if current_speed > 1e-9: # If agent has a heading
        direction_current = agent.current_heading_vector # Use normalized heading
    else: # If agent is stopped, try to move towards destination or a default direction
        vec_to_dest_if_stopped = agent.dest - current_pos
        if np.linalg.norm(vec_to_dest_if_stopped) > 1e-6:
            direction_current = vec_to_dest_if_stopped / np.linalg.norm(vec_to_dest_if_stopped)
        else: # At destination and stopped
            direction_current = np.array([1,0]) # Default direction if truly stuck

    cand_vel2_speed = agent.desired_speed
    cand_vel2 = direction_current * cand_vel2_speed
    cand_pos2 = current_pos + cand_vel2 * dt
    # Add only if different from cand1 (e.g. if current_speed is already desired_speed)
    if not any(np.allclose(cand_vel2, existing_cand['vel']) for existing_cand in candidates):
        candidates.append({'pos': cand_pos2, 'vel': cand_vel2, 'time_to_reach': dt})


    # Option 3: Turn towards destination and move at desired_speed
    vec_to_dest = agent.dest - current_pos
    dist_to_dest = np.linalg.norm(vec_to_dest)
    if dist_to_dest > 1e-3: # If not already at destination
        dir_to_dest = vec_to_dest / dist_to_dest
        cand_vel3_speed = agent.desired_speed # Move at desired speed towards destination
        cand_vel3 = dir_to_dest * cand_vel3_speed
        cand_pos3 = current_pos + cand_vel3 * dt
        if not any(np.allclose(cand_vel3, existing_cand['vel']) for existing_cand in candidates):
             candidates.append({'pos': cand_pos3, 'vel': cand_vel3, 'time_to_reach': dt})

    # Option 4: Stop (or significantly reduce speed)
    cand_vel4 = current_vel * 0.1 
    cand_pos4 = current_pos + cand_vel4 * dt
    if not any(np.allclose(cand_vel4, existing_cand['vel']) for existing_cand in candidates):
        candidates.append({'pos': cand_pos4, 'vel': cand_vel4, 'time_to_reach': dt})
    
    # Option 5: Stay put (important if all moves are bad or agent is waiting)
    cand_vel_stay = np.array([0.0, 0.0])
    cand_pos_stay = current_pos
    if not any(np.allclose(cand_vel_stay, existing_cand['vel']) for existing_cand in candidates):
        candidates.append({'pos': cand_pos_stay, 'vel': cand_vel_stay, 'time_to_reach': dt})


    # Ensure all candidate velocities are physically possible
    final_candidates = []
    for cand in candidates:
        cand_speed_val = np.linalg.norm(cand['vel'])
        if cand_speed_val > sim_config['max_agent_speed']:
            final_vel = (cand['vel'] / cand_speed_val) * sim_config['max_agent_speed']
        else:
            final_vel = cand['vel']
        
        # Recalculate pos based on potentially capped vel
        final_pos = current_pos + final_vel * dt 
        final_candidates.append({'pos': final_pos, 'vel': final_vel, 'time_to_reach': dt})
    
    # If after all this, no candidates (e.g., agent is at dest and all moves are "stay"), ensure one exists.
    if not final_candidates:
         final_candidates.append({'pos': current_pos, 'vel': np.array([0,0]), 'time_to_reach': dt})


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
        'collision_pred_variances': [0.25, 0.25], 
        'max_agent_speed': 10.0, 
        'base_desired_speed': 5.0, 
        'collision_penalty_metric_factor': 10.0 
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

                util_dir = directional_alignment_utility(agent.pos, cand_pos, agent.dest, agent.current_heading_vector, params)
                
                # Simplified perception for speed alignment:
                # For this conceptual model, assume no specific leading agent is identified.
                # v_i_current for speed utility should be the speed resulting from the candidate action.
                is_horizon_empty_simple = True 
                leading_speed_simple = None 
                util_speed = speed_alignment_utility(agent.current_speed, cand_speed_val, leading_speed_simple, agent.desired_speed, params, is_horizon_empty_simple)

                util_dist = distance_reward_utility(cand_pos, agent.dest, agent.current_speed, params, sim_config)

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

# --- Example of how to run it once (for testing the simulation itself) ---
if __name__ == '__main__':
    # Order: S_theta, S_v, xi_i, S_d, gamma, w_x, w_y, w_c, w_ell, beta
    example_params = [
        0.6, 0.6, 2.55, 0.6, 1.75,
        2.75, 2.75, 2.75, 2.75, 1.05
    ]
    
    print("Running single simulation with example parameters...")
    metric = evaluate_simulation_model(example_params)
    print(f"Simulation finished. Performance Metric (Y): {metric}")


# In[3]:


import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import pandas as pd # For a nicer table output

# evaluate_simulation_model is defined in the cell above and is already in the namespace.

# Step 1: Define the problem.
# Parameters match the paper's utility formulation (Eq. 4-13): S_theta (Eq. 5),
# S_v & xi_i (Eq. 6), S_d & gamma (Eq. 7), w_c (Eq. 12), w_ell & beta (Eq. 13),
# plus distance weights w_x, w_y on d_eff (Eq. 7).
problem = {
    'num_vars': 10,
    'names': ['S_theta', 'S_v', 'xi_i', 'S_d', 'gamma',
              'w_x', 'w_y', 'w_c', 'w_ell', 'beta'],
    'bounds': [[0.2, 1.0],    # S_theta
               [0.2, 1.0],    # S_v
               [1.5, 3.5],    # xi_i
               [0.2, 1.0],    # S_d
               [0.5, 3.0],    # gamma
               [0.5, 5.0],    # w_x
               [0.5, 5.0],    # w_y
               [0.1, 20.0],   # w_c
               [5.0, 50],     # w_ell
               [0.1, 5.0]]    # beta
}

# Step 2: Generate samples
# For actual analysis, increase N_samples significantly (e.g., 1024, 2048, or your planned 2000).
# Using N_samples = 64 for a quick demonstration run.
N_samples = 64
param_values = sobol_sample.sample(problem, N_samples, calc_second_order=False)

print(f"Generated {param_values.shape[0]} parameter samples for GSA with updated ranges.")

# Step 3: Run the model for each parameter sample
Y_outputs = np.zeros(param_values.shape[0])

print(f"Running {param_values.shape[0]} model evaluations (this may take some time)...")
for i, X_sample_list in enumerate(param_values):
    if (i + 1) % 100 == 0 or i == 0 or i == param_values.shape[0] -1 : # Print progress
        print(f"GSA: Evaluating sample {i+1}/{param_values.shape[0]}")
    
    Y_outputs[i] = evaluate_simulation_model(X_sample_list)
print("Model evaluations complete.")

# Step 4: Perform Sobol analysis
Si = sobol.analyze(problem, Y_outputs, calc_second_order=False, print_to_console=False)

print("\n--- Global Sensitivity Analysis Results (with Updated Ranges) ---")

# Step 5: Print the results
print("\nSobol Indices (S1 and ST):")

results_data = []
for i, name in enumerate(problem['names']):
    results_data.append({
        "Parameter": name,
        "S1": Si['S1'][i],
        "S1_conf": Si['S1_conf'][i],
        "ST": Si['ST'][i],
        "ST_conf": Si['ST_conf'][i]
    })

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False, float_format="%.4f"))

print("\nInterpretation Notes:")
print("- S1 (First-order index): Measures the direct contribution of each parameter to the output variance.")
print("- ST (Total-order index): Measures the total contribution of each parameter, including interactions.")
print("- S1_conf / ST_conf: Confidence interval. Larger values indicate less certainty (consider increasing N_samples).")
print("- If ST is much larger than S1, it indicates strong interaction effects.")
print("- Parameters with very low ST values (close to 0) have little influence on the output variance for the chosen ranges.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




