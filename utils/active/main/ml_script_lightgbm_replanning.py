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
CONTROL_SMOOTHING_ALPHA = 1.0

# --- NEW: PLOTTING FUNCTION FOR CONTROL PROFILES ---
def plot_control_profiles(t_eval, history_mc_states, sol_nom, window_type, level, t_start):
    """
    Plots the mean predicted costate profiles against the nominal costate profiles.
    """
    # Extract the mean costates from the simulation history
    # The costates are the last 7 elements of the state vector (index 7 onwards)
    mean_costates_history = np.array([np.mean(states_at_t[:, 7:], axis=0) for states_at_t in history_mc_states])
    
    # Extract the nominal costates from the nominal solution
    # The nominal costates are rows 7 to 13 of the solution object's state vector
    nominal_costates_history = sol_nom.y[7:, :].T

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f'Control Costate Profiles ({window_type.upper()} Window)', fontsize=16)
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]
        ax.plot(t_eval, nominal_costates_history[:, i], color='black', linestyle='-', label='Nominal Control')
        ax.plot(t_eval, mean_costates_history[:, i], color='red', linestyle='--', label='Mean Closed-Loop Control')
        ax.set_title(f'Costate $λ_{i+1}$ Profile')
        ax.set_xlabel('Time [TU]')
        ax.set_ylabel('Costate Value')
        ax.grid(True, linestyle=':')
        ax.legend()

    # Hide any unused subplots
    if len(axes) > 7:
        for i in range(7, len(axes)):
            fig.delaxes(axes[i])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"control_profiles_{t_start:.3f}_{level}_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n  [Saved Plot] {fname}")


# --- MAIN SIMULATION FUNCTION ---
def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data):
    # ... (simulation setup code is unchanged) ...
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
    
    print("\n[INFO] Generating initial sigma points for simulation...")
    mean_state_cart = np.hstack([r0, v0, m0_val])
    nsd = len(mean_state_cart)
    sigma_points_generator = MerweScaledSigmaPoints(n=nsd, alpha=1e-3, beta=2., kappa=3-nsd)
    cartesian_samples = sigma_points_generator.sigma_points(mean_state_cart, P_cart)
    print(f"[INFO] Propagating {len(cartesian_samples)} sigma points.")
    
    epsilon = 1e-9
    
    mc_states_current = []
    for s in cartesian_samples:
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        state_s = np.hstack([rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten(), m_s])
        mc_states_current.append(state_s)
    mc_states_current = np.array(mc_states_current)

    P_mee_initial = np.cov(mc_states_current, rowvar=False)
    diag_vals_initial = np.diag(P_mee_initial)

    temp_states_with_costates = []
    for j in tqdm(range(len(mc_states_current)), desc="  -> Initializing samples"):
        state_s = mc_states_current[j]
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
        
        temp_states_with_costates.append(np.hstack([state_s, lam_s]))
    mc_states_current = np.array(temp_states_with_costates)

    history_mc_states = [np.array(mc_states_current)]
    
    tqdm.write("Step |  Time (TU) | Deviation (km) | Action")
    tqdm.write("-" * 50)
    
    for i in tqdm(range(len(t_eval) - 1), desc="    -> Propagating"):
        # ... (propagation and replanning logic is unchanged) ...
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        states_at_next_step = [solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step]).y[:, -1] for s in mc_states_current]
        mc_states_current = np.array(states_at_next_step)
        
        P_mee_current = np.cov(mc_states_current[:, :7], rowvar=False)
        diag_vals_current = np.diag(P_mee_current)

        current_mc_positions = np.array([mee2rv(*s[:6], mu)[0] for s in mc_states_current])
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        
        mee_nom_current = sol_nom.y[:6, i+1]
        r_nom_current, v_nom_current = mee2rv(*mee_nom_current, mu)
        
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_current) * DU_km
        log_action = "---"
        
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
                
                old_lam = mc_states_current[j][7:]
                smoothed_lam = CONTROL_SMOOTHING_ALPHA * predicted_lam + (1 - CONTROL_SMOOTHING_ALPHA) * old_lam
                mc_states_current[j][7:] = smoothed_lam
        
        history_mc_states.append(np.array(mc_states_current))
        tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {log_action}")
    
    # MODIFIED: Call the new plotting function
    plot_control_profiles(t_eval, history_mc_states, sol_nom, window_type, "random1", t_start_replan)

# --- SCRIPT ENTRY POINT ---
def main():
    # ... (main function is unchanged) ...
    try:
        model_max = joblib.load("trained_model_max.pkl")
        model_min = joblib.load("trained_model_min.pkl")
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
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
    
    run_replanning_simulation(model_max, t_start_max, t_end_max, 'max', data)
    run_replanning_simulation(model_min, t_start_min, t_end_min, 'min', data)
    
    print("\n[SUCCESS] All simulations complete.")


if __name__ == "__main__":
    main()