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

# --- ANALYSIS CONSTANT ---
DEVIATION_THRESHOLD_KM = 500.0

# --- HELPER & PLOTTING FUNCTIONS ---

def compute_thrust_direction(mu, mee, lam):
    """Computes the optimal thrust direction vector from costates."""
    p, f, g, h, k, L = mee
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:-1]
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    if np.isclose(w, 0, atol=1e-10): return np.full(3, np.nan)
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
    return mat.flatten() / norm if norm > 0 else np.full(3, np.nan)

# In ml_script_lightgbm_replanning.py

def plot_full_replan_trajectory(r_nom, r_fval, r_mc_mean, replan_coords, replan_thrusts, window_type, level, t_start):
    """Plots full X-Y trajectories and marks replanning events with control quivers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(r_nom[:, 0], r_nom[:, 1], color='0.0', linestyle='-', linewidth=1.5, label='Nominal (F_nom)', alpha=0.6)
    ax.plot(r_fval[:, 0], r_fval[:, 1], color='0.5', linestyle='--', linewidth=1.2, label='Nominal (0.9*F_nom)', alpha=0.6)
    ax.plot(r_mc_mean[:, 0], r_mc_mean[:, 1], color='0.2', linestyle=':', linewidth=2.0, label='MC Mean (Closed-Loop)')
    
    ax.scatter(r_nom[0, 0], r_nom[0, 1], c='0.2', marker='o', s=100, alpha=0.7, label='Start', zorder=5)
    ax.scatter(r_nom[-1, 0], r_nom[-1, 1], c='0.0', marker='X', s=120, alpha=0.7, label='Nominal End', zorder=5)
    
    if replan_coords:
        replan_arr = np.array(replan_coords)
        thrust_arr = np.array(replan_thrusts)
        ax.quiver(replan_arr[:, 0], replan_arr[:, 1], thrust_arr[:, 0], thrust_arr[:, 1], 
                  color='red', alpha=0.4, width=0.005, label='Replanning Control')

    ax.set_xlabel('X [DU]'); ax.set_ylabel('Y [DU]')
    ax.legend(); ax.grid(True, linestyle=':')
    
    # --- START: MODIFICATION FOR PLOT SCALING ---
    
    # Calculate bounds from the nominal trajectory to keep the view focused
    x_min, x_max = np.min(r_nom[:, 0]), np.max(r_nom[:, 0])
    y_min, y_max = np.min(r_nom[:, 1]), np.max(r_nom[:, 1])

    # Add 10% padding to the limits
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Use this method to enforce an equal aspect ratio within the new limits
    ax.set_aspect('equal', adjustable='box')
    
    # --- END: MODIFICATION FOR PLOT SCALING ---

    inset_ax = fig.add_axes([0.65, 0.15, 0.25, 0.25])
    inset_ax.plot(r_nom[-4:, 0], r_nom[-4:, 1], color='black', linestyle='-', marker='o', markersize=3)
    inset_ax.plot(r_fval[-4:, 0], r_fval[-4:, 1], color='0.5', linestyle='--', marker='p', markersize=3)
    inset_ax.plot(r_mc_mean[-4:, 0], r_mc_mean[-4:, 1], color='0.2', linestyle=':', marker='*', markersize=4)
    inset_ax.scatter(r_nom[-1, 0], r_nom[-1, 1], c='black', marker='X', s=50, alpha=0.7)
    inset_ax.ticklabel_format(useOffset=False, style='plain')
    inset_ax.set_title("Final Timesteps", fontsize=9)
    inset_ax.grid(True, linestyle=':')

    fig.tight_layout()
    fname = f"full_replan_trajectory_{t_start:.3f}_{level}_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n  [Saved Plot] {fname}")

# --- MAIN SIMULATION FUNCTION ---

def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data, diag_mins, diag_maxs):
    """Runs a closed-loop Monte Carlo simulation with detailed logging."""
    print(f"\n--- Running Closed-Loop Sim for {window_type.upper()} WINDOW ---")
    
    feature_mins = np.load("feature_mins.npy")
    feature_maxs = np.load("feature_maxs.npy")

    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    DU_km = 696340.0
    F_val = 0.98 * F_nom
    
    rand_diags = {'random1': np.random.uniform(low=diag_mins, high=diag_maxs)}
    
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, lam_tr[idx_start]])
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    replan_r_nom, _ = mee2rv(*sol_nom.y[:6], mu)
    sol_fval = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    replan_r_fval, _ = mee2rv(*sol_fval.y[:6], mu)
    
    feature_cols = ['t','p','f','g','h','k','L','mass','c1','c2','c3','c4','c5','c6','c7', 'score']

    results = []
    for level, diag_vals in rand_diags.items():
        print(f"  Simulating uncertainty level: {level}...")
        
        P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/86400)**2)), [1e-3/(4000**2)]]))
        shared_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=1000)
        
        # --- Initial guidance: Predict for each sample individually ---
        mc_states_current = []
        print("[INFO] Performing initial guidance prediction for each MC sample...")
        for s in tqdm(shared_samples, desc="  -> Initializing samples"):
            r_s, v_s, m_s = s[:3], s[3:6], s[6]
            mee_s = rv2mee(r_s, v_s, mu)
            state_s = np.hstack([mee_s, m_s])
            
            # Initial score is near zero as we start on the nominal path
            dev_s = np.linalg.norm(r_s - r0) * DU_km
            score_s = min(1.0, dev_s / DEVIATION_THRESHOLD_KM)
            
            features_unclamped = np.hstack([t_start_replan, state_s, diag_vals, score_s])
            features = np.clip(features_unclamped, feature_mins, feature_maxs)
            
            lam_s = model.predict(pd.DataFrame([features], columns=feature_cols))[0]
            mc_states_current.append(np.hstack([state_s, lam_s]))

        history_mc_states = [np.array(mc_states_current)]
        replan_coords, replan_times, replan_thrusts = [], [], []
        
        tqdm.write("Step |  Time (TU) | Deviation (km) | Ellipsoid Vol (km^3) | Action")
        tqdm.write("-" * 70)
        
        for i in tqdm(range(len(t_eval) - 1), desc="    -> Propagating"):
            t_start_step, t_end_step = t_eval[i], t_eval[i+1]
            
            mc_states_to_propagate = [np.copy(s) for s in mc_states_current]
            mc_states_next_step = []

            for state in mc_states_to_propagate:
                sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], state, t_eval=[t_end_step])
                mc_states_next_step.append(sol.y[:, -1])
            
            mc_states_current = mc_states_next_step
            history_mc_states.append(np.array(mc_states_current))
            
            current_mc_positions = np.array([mee2rv(*s[:6], mu)[0].flatten() for s in mc_states_current])
            mean_mc_pos = np.mean(current_mc_positions, axis=0)
            deviation_km = np.linalg.norm(mean_mc_pos - replan_r_nom[i+1]) * DU_km
            
            # Calculate and log ellipsoid volume at each step
            cov_mc = np.cov(current_mc_positions.T)
            eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
            volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
            
            log_action = "---"
            if deviation_km > DEVIATION_THRESHOLD_KM:
                log_action = "REPLAN"
                if not replan_times: replan_times.append(t_end_step)
                
                replan_coords.append(mean_mc_pos)
                
                # For plotting, calculate the thrust for the mean state
                mean_current_mee = np.mean([s[:7] for s in mc_states_current], axis=0)
                mean_score = min(1.0, deviation_km / DEVIATION_THRESHOLD_KM)
                mean_features_unclamped = np.hstack([t_end_step, mean_current_mee, diag_vals, mean_score])
                mean_features = np.clip(mean_features_unclamped, feature_mins, feature_maxs)
                mean_lam = model.predict(pd.DataFrame([mean_features], columns=feature_cols))[0]
                replan_thrusts.append(compute_thrust_direction(mu, mean_current_mee[:6], mean_lam))

                # --- NEW LOGIC: Predict for each sample individually ---
                r_nom_current, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
                for j in range(len(mc_states_current)):
                    sample_state = mc_states_current[j][:7]
                    r_sample, _ = mee2rv(*sample_state[:6], mu)
                    sample_dev = np.linalg.norm(r_sample - r_nom_current) * DU_km
                    sample_score = min(1.0, sample_dev / DEVIATION_THRESHOLD_KM)
                    
                    sample_features_unclamped = np.hstack([t_end_step, sample_state, diag_vals, sample_score])
                    sample_features = np.clip(sample_features_unclamped, feature_mins, feature_maxs)
                    
                    predicted_lam = model.predict(pd.DataFrame([sample_features], columns=feature_cols))[0]
                    mc_states_current[j][7:] = predicted_lam
            
            tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {volume_km3:20.2e} | {log_action}")

        history_mc_r = [np.array([mee2rv(*s[:6], mu)[0].flatten() for s in states_at_t]) for states_at_t in history_mc_states]
        mc_mean_r = np.mean(np.array(history_mc_r), axis=1)
        
        thrust_nom = compute_thrust_direction(mu, nominal_state_start[:6], lam_tr[idx_start])
        final_dev_mc_km = np.linalg.norm(mc_mean_r[-1] - replan_r_nom[-1]) * DU_km
        
        plot_full_replan_trajectory(replan_r_nom, sol_fval.y[:3].T, mc_mean_r, replan_coords, replan_thrusts, window_type, level, t_start_replan)
        
    return [] # Results dictionary removed as Excel output is disabled

# --- SCRIPT ENTRY POINT ---

def main():
    """Main execution function to run simulations and save results."""
    try:
        model_max = joblib.load("trained_model_max.pkl")
        model_min = joblib.load("trained_model_min.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find model file: {e}. Please run the training script first.")
        return

    try:
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        width_file_path = "stride_1440min/bundle_segment_widths.txt"
        with open(width_file_path) as f: lines = f.readlines()[1:]
        times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
        diag_mins, diag_maxs = np.load("diag_mins.npy"), np.load("diag_maxs.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input files are present.")
        return
        
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max, t_end_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_min, t_end_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    all_results = []
    all_results.extend(run_replanning_simulation(model_max, t_start_max, t_end_max, 'max', data, diag_mins, diag_maxs))
    all_results.extend(run_replanning_simulation(model_min, t_start_min, t_end_min, 'min', data, diag_mins, diag_maxs))
    
    print("\n[SUCCESS] All simulations complete.")

if __name__ == "__main__":
    main()