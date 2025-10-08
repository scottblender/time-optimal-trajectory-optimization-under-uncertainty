# ml_script_lightgbm_replanning.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc
from filterpy.kalman import MerweScaledSigmaPoints

# --- ANALYSIS CONSTANT ---
DEVIATION_THRESHOLD_KM = 100.0

# --- HELPER AND PLOTTING FUNCTIONS FOR THRUST VECTOR ---
def compute_thrust_direction(mu, F, mee, lam):
    """
    Computes the optimal thrust direction vector from MEE state and costates.
    NOTE: Assumes 'mee' and 'lam' are 7-element vectors (MEE + mass).
    """
    p, f, g, h, k, L = mee[:-1]
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:-1]

    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL

    if np.isclose(w, 0, atol=1e-10):
        return np.full(3, np.nan)  # degenerate case

    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    C2 = 1 / w
    C3 = h * SinL - k * CosL

    A = np.array([
        [0, 2 * p * C2 * C1, 0],
        [C1 * SinL, C1 * C2 * ((w + 1) * CosL + f), -C1 * (g / w) * C3],
        [-C1 * CosL, C1 * C2 * ((w + 1) * SinL + g), C1 * (f / w) * C3],
        [0, 0, C1 * s * CosL * C2 / 2],
        [0, 0, C1 * s * SinL * C2 / 2],
        [0, 0, C1 * C2 * C3]
    ])
    mat = A.T @ lam_matrix
    norm = np.linalg.norm(mat)
    return (mat.flatten() / norm) if norm > 1e-9 else np.zeros(3)

def plot_thrust_profiles(t_eval, history_mc_states, sol_nom, mu, F_nom, window_type, level, t_start):
    """
    Plots the mean closed-loop thrust vector components against the nominal profiles.
    """
    # Calculate thrust history for the closed-loop (mean) trajectory
    thrust_history_closed_loop = []
    for states_at_t in history_mc_states:
        mean_state = np.mean(states_at_t, axis=0)
        mean_mee_mass = mean_state[:7]
        mean_lam = mean_state[7:]
        thrust_vector = compute_thrust_direction(mu, F_nom, mean_mee_mass, mean_lam)
        thrust_history_closed_loop.append(thrust_vector)
    thrust_history_closed_loop = np.array(thrust_history_closed_loop)
    
    # Calculate thrust history for the nominal trajectory
    thrust_history_nominal = []
    for i in range(len(t_eval)):
        nominal_mee_mass = sol_nom.y[:7, i]
        nominal_lam = sol_nom.y[7:, i]
        thrust_vector = compute_thrust_direction(mu, F_nom, nominal_mee_mass, nominal_lam)
        thrust_history_nominal.append(thrust_vector)
    thrust_history_nominal = np.array(thrust_history_nominal)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Thrust Control Vector Profiles ({window_type.upper()} Window)', fontsize=16)
    axes = axes.flatten()
    labels = ['$u_x$', '$u_y$', '$u_z$']

    for i in range(3):
        ax = axes[i]
        ax.plot(t_eval, thrust_history_nominal[:, i], color='black', linestyle='-', label='Nominal Thrust')
        ax.plot(t_eval, thrust_history_closed_loop[:, i], color='red', linestyle='--', label='Mean Closed-Loop Thrust')
        ax.set_title(f'Thrust Component {labels[i]}')
        ax.set_xlabel('Time [TU]')
        ax.set_ylabel('Thrust Component Value')
        ax.grid(True, linestyle=':')
        ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"thrust_profiles_{t_start:.3f}_{level}_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n  [Saved Plot] {fname}")


# --- MAIN SIMULATION FUNCTION ---
def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data, diag_mins, diag_maxs):
    # ... (setup is the same) ...
    print(f"\n--- Running Closed-Loop Sim for {window_type.upper()} WINDOW ---")
    
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    DU_km = 696340.0
    F_val = 0.98 * F_nom
    
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, lam_tr[idx_start]])
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    
    feature_cols = ['t','p','f','g','h','k','L','mass','c1','c2','c3','c4','c5','c6','c7',
                    'pos_error_score', 'vel_error_score', 'energy_error_score']

    P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/86400)**2)), [1e-3/(4000**2)]]))
    
    print("\n[INFO] Generating Monte Carlo samples for simulation...")
    num_samples = 50
    cartesian_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=num_samples)
    print(f"[INFO] Propagating {num_samples} Monte Carlo samples.")
    
    epsilon = 1e-9
    
    initial_mee_states = np.array([np.hstack([rv2mee(s[:3].reshape(1,3), s[3:6].reshape(1,3), mu).flatten(), s[6]]) for s in cartesian_samples])
    
    mean_mee_initial = np.mean(initial_mee_states, axis=0)
    devs_initial = initial_mee_states - mean_mee_initial
    P_mee_initial = np.einsum('ji,jk->ik', devs_initial, devs_initial) / (num_samples - 1)
    diag_vals_initial = np.diag(P_mee_initial)

    mc_states_current = []
    for j in tqdm(range(len(initial_mee_states)), desc="  -> Initializing samples"):
        state_s = initial_mee_states[j]
        r_s, v_s = mee2rv(*state_s[:6], mu)

        norm_r_s, norm_r0 = np.linalg.norm(r_s), np.linalg.norm(r0)
        pos_error = (norm_r_s - norm_r0) / (norm_r_s + norm_r0 + epsilon)
        norm_v_s, norm_v0 = np.linalg.norm(v_s), np.linalg.norm(v0)
        vel_error = (norm_v_s - norm_v0) / (norm_v_s + norm_v0 + epsilon)
        E_nom_0 = 0.5 * np.dot(v0, v0) - mu / norm_r0
        E_s_0 = 0.5 * np.dot(v_s, v_s) - mu / norm_r_s
        energy_error = (E_s_0 - E_nom_0) / (E_s_0 + E_nom_0 + epsilon)
        
        features = np.hstack([t_start_replan, state_s, diag_vals_initial, pos_error, vel_error, energy_error])
        lam_s = model.predict(pd.DataFrame([features], columns=feature_cols))[0]
        mc_states_current.append(np.hstack([state_s, lam_s]))
    
    mc_states_current = np.array(mc_states_current)
    history_mc_states = [mc_states_current]
    
    tqdm.write("Step |  Time (TU) | Deviation (km) | Ellipsoid Vol (km^3) | Action")
    tqdm.write("-" * 70)
    
    for i in tqdm(range(len(t_eval) - 1), desc="    -> Propagating"):
        # ... (propagation and replanning logic is the same) ...
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        next_states_list = []
        for s in mc_states_current:
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step])
            next_states_list.append(sol.y[:, -1])
        
        mc_states_current = np.array(next_states_list)
        
        mee_states_current = mc_states_current[:, :7]
        mean_mee_current = np.mean(mee_states_current, axis=0)
        devs_current = mee_states_current - mean_mee_current
        P_mee_current = np.einsum('ji,jk->ik', devs_current, devs_current) / (num_samples - 1)
        diag_vals_current = np.diag(P_mee_current)

        current_mc_positions = np.array([mee2rv(*s[:6], mu)[0] for s in mc_states_current])
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        
        mee_nom_current = sol_nom.y[:6, i+1]
        r_nom_current, v_nom_current = mee2rv(*mee_nom_current, mu)
        
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_current) * DU_km
        log_action = "---"
        
        cov_cartesian = np.cov(current_mc_positions.T)
        eigvals = np.maximum(np.linalg.eigvalsh(cov_cartesian), 0)
        volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        
        if deviation_km > DEVIATION_THRESHOLD_KM:
            log_action = "REPLAN"
            for j in range(len(mc_states_current)):
                sample_state = mc_states_current[j][:7]
                r_sample, v_sample = mee2rv(*sample_state[:6], mu)
                
                norm_r_sample, norm_r_nom_curr = np.linalg.norm(r_sample), np.linalg.norm(r_nom_current)
                pos_error = (norm_r_sample - norm_r_nom_curr) / (norm_r_sample + norm_r_nom_curr + epsilon)
                norm_v_sample, norm_v_nom_curr = np.linalg.norm(v_sample), np.linalg.norm(v_nom_current)
                vel_error = (norm_v_sample - norm_v_nom_curr) / (norm_v_sample + norm_v_nom_curr + epsilon)
                E_nom_curr = 0.5 * np.dot(v_nom_current, v_nom_current) - mu / norm_r_nom_curr
                E_sample_curr = 0.5 * np.dot(v_sample, v_sample) - mu / norm_r_sample
                energy_error = (E_sample_curr - E_nom_curr) / (E_sample_curr + E_nom_curr + epsilon)

                features = np.hstack([t_end_step, sample_state, diag_vals_current, pos_error, vel_error, energy_error])
                predicted_lam = model.predict(pd.DataFrame([features], columns=feature_cols))[0]
                
                mc_states_current[j][7:] = predicted_lam
        
        history_mc_states.append(mc_states_current)
        tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {volume_km3:20.2e} | {log_action}")
    
    # MODIFIED: Call the new thrust plotting function
    plot_thrust_profiles(t_eval, history_mc_states, sol_nom, mu, F_nom, window_type, "random1", t_start_replan)

# --- SCRIPT ENTRY POINT ---
def main():
    # ... (main function is unchanged) ...
    try:
        model_max = joblib.load("trained_model_max.pkl")
        model_min = joblib.load("trained_model_min.pkl")
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        diag_mins = np.load("diag_mins.npy")
        diag_maxs = np.load("diag_maxs.npy")
        width_file_path = "stride_1440min/bundle_segment_widths.txt"
        with open(width_file_path) as f: lines = f.readlines()[1:]
        times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input files are present.")
        return

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    max_idx = int(np.argmax(times_arr[:, 1])); t_start_max, t_end_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    min_idx_raw = int(np.argmin(times_arr[:, 1])); min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_min, t_end_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    run_replanning_simulation(model_min, t_start_min, t_end_min, 'min', data, diag_mins, diag_maxs)
    run_replanning_simulation(model_max, t_start_max, t_end_max, 'max', data, diag_mins, diag_maxs)
    
    
    print("\n[SUCCESS] All simulations complete.")


if __name__ == "__main__":
    main()