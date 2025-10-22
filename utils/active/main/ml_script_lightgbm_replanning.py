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

# --- MISSION PARAMETERS & CONSTANTS ---
DU_km = 696340.0
DEVIATION_THRESHOLD_KM = 100.0
REPLAN_COOLDOWN_STEPS = 50
NUM_MC_SAMPLES = 1000

# ==============================================================================
# === PLOTTING FUNCTIONS =======================================================
# ==============================================================================

def plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, mu, DU_km, window_type):
    """Plots the final XYZ deviation comparison."""
    r_nom_hist, _ = mee2rv(*sol_nom.y[:6, :], mu)
    r_nom_perturbed_hist, _ = mee2rv(*sol_nom_perturbed.y[:6, :], mu)
    
    mean_states_optimal = np.mean(history_optimal, axis=1)
    r_optimal_hist, _ = mee2rv(*mean_states_optimal[:, :6].T, mu)

    mean_states_corrective = np.mean(history_corrective, axis=1)
    r_corrective_hist, _ = mee2rv(*mean_states_corrective[:, :6].T, mu)

    delta_r_optimal = (r_optimal_hist - r_nom_hist) * DU_km
    delta_r_corrective = (r_corrective_hist - r_nom_hist) * DU_km
    delta_r_perturbed = (r_nom_perturbed_hist - r_nom_hist) * DU_km

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Trajectory Deviation in XYZ Frame ({window_type.upper()} Window)', fontsize=16)
    labels = ['X Deviation [km]', 'Y Deviation [km]', 'Z Deviation [km]']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(t_eval, delta_r_perturbed[:, i], color='gray', linestyle='-', lw=2, label='Perturbed Nominal (0.98F)')
        ax.plot(t_eval, delta_r_optimal[:, i], color='blue', linestyle='--', lw=2, label='Optimal (Open-Loop)')
        ax.plot(t_eval, delta_r_corrective[:, i], color='red', linestyle=':', lw=2.5, label='Corrective (Closed-Loop)')
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle=':')
        ax.legend()
        
    axes[-1].set_xlabel('Time [TU]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"final_xyz_deviation_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n[Saved Plot] {fname}")

# ==============================================================================
# === CORE PROPAGATION AND CONTROL LOGIC =======================================
# ==============================================================================

def propagate_and_control(models, strategy, t_eval, data, sol_nom, initial_mc_states, window_type):
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    F_val = 0.98 * F_nom # Propagate with perturbed thrust
    mc_states_current = np.copy(initial_mc_states)

    feature_cols_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    
    feature_cols_corrective = feature_cols_optimal + [
        'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
        'delta_m', 't_go'
    ]

    # --- Initial open-loop "optimal" policy application (Control the Mean) ---
    tqdm.write(f"\n[INFO] Calculating initial control policy for {strategy.upper()} strategy...")
    initial_mee_states = mc_states_current[:, :7]
    mean_initial_state = np.mean(initial_mee_states, axis=0)
    initial_devs = initial_mee_states - mean_initial_state
    initial_cov = np.einsum('ji,jk->ik', initial_devs, initial_devs) / (len(initial_mee_states) - 1)
    initial_diag_vals = np.diag(initial_cov)
    
    model_optimal = models['optimal']
    features_opt = np.hstack([t_eval[0], mean_initial_state, initial_diag_vals])
    predicted_lam_optimal = model_optimal.predict(pd.DataFrame([features_opt], columns=feature_cols_optimal))[0]
    mc_states_current[:, 7:] = predicted_lam_optimal # Apply same control to all samples

    history_mc_states = [np.copy(mc_states_current)]
    steps_since_replan = REPLAN_COOLDOWN_STEPS
    
    tqdm.write(f"--- Propagating with {strategy.upper()} Strategy ---")
    tqdm.write(f"{'Step':<5} | {'Time (TU)':<12} | {'Deviation (km)':<18} | {'Ellipsoid Vol (km^3)':<22} | {'Action':<10}")
    tqdm.write("-" * 80)

    for i in tqdm(range(len(t_eval) - 1), desc=f"    -> Propagating ({strategy})", leave=False):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        next_states_list = [solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step]).y[:, -1] for s in mc_states_current]
        mc_states_current = np.array(next_states_list)
        
        current_mc_positions, _ = mee2rv(*mc_states_current[:, :6].T, mu)
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        r_nom_step, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_step) * DU_km
        
        cov_cartesian = np.cov(current_mc_positions.T)
        eigvals = np.maximum(np.linalg.eigvalsh(cov_cartesian), 0)
        volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        log_action = "---"

        if deviation_km > DEVIATION_THRESHOLD_KM and strategy == 'corrective' and steps_since_replan >= REPLAN_COOLDOWN_STEPS:
            log_action = "REPLAN"
            steps_since_replan = 0 # Reset cooldown
            
            mee_nom_full_curr = sol_nom.y[:7, i+1]
            lam_nom_curr = sol_nom.y[7:, i+1]
            r_nom_curr, v_nom_curr = mee2rv(*mee_nom_full_curr[:6], mu)
            
            # Control based on the mean of the distribution
            mee_states_current = mc_states_current[:, :7]
            mean_mee_state_curr = np.mean(mee_states_current, axis=0)
            devs_current = mee_states_current - mean_mee_state_curr
            diag_vals_current = np.diag(np.einsum('ji,jk->ik', devs_current, devs_current) / (len(mc_states_current) - 1))

            # --- On-the-fly Feature Engineering for Prediction (Corrected) ---
            r_mean_curr, v_mean_curr = mee2rv(*mean_mee_state_curr[:6], mu)
            delta_r = r_nom_curr - r_mean_curr
            delta_v = v_nom_curr - v_mean_curr
            delta_m = mee_nom_full_curr[-1] - mean_mee_state_curr[-1]
            t_go = t_eval[-1] - t_end_step

            features_corr = np.hstack([
                t_end_step, mean_mee_state_curr, diag_vals_current, 
                delta_r.flatten(), delta_v.flatten(), delta_m, t_go
            ])
            
            model_corrective = models['corrective']
            lam_correction = model_corrective.predict(pd.DataFrame([features_corr], columns=feature_cols_corrective))[0]
            
            # Apply same new control law to all samples
            mc_states_current[:, 7:] = lam_nom_curr + lam_correction if window_type == "min" else lam_nom_curr - lam_correction
        
        history_mc_states.append(np.copy(mc_states_current))
        tqdm.write(f"{i+1:<5} | {t_end_step:<12.2f} | {deviation_km:<18.2f} | {volume_km3:<22.2e} | {log_action:<10}")
        steps_since_replan += 1

    return np.array(history_mc_states)

# ==============================================================================
# === MAIN SIMULATION RUNNER ===================================================
# ==============================================================================

def run_comparison_simulation(models, t_start_replan, t_end_replan, window_type, data):
    print(f"\n{'='*25} Running Comparison for {window_type.upper()} WINDOW {'='*25}")
    
    mu, r_nom_hist, v_nom_hist = data["mu"], data["r_tr"], data["v_tr"]
    mass_nom_hist, lam_tr, t_vals = data["mass_tr"], data["lam_tr"], np.asarray(data["backTspan"][::-1])
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val, initial_lam = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start], lam_tr[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, initial_lam])
    
    print("[INFO] Propagating Nominal, Perturbed, and Controlled trajectories...")
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    sol_nom_perturbed = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], 0.98 * data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    g0 = 9.81/1000
    TU = np.sqrt(DU_km/g0)
    P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/TU)**2)), [1e-3/(4000**2)]]))
    cartesian_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=NUM_MC_SAMPLES)
    initial_mee_states = np.array([np.hstack([rv2mee(s[:3].reshape(1,3), s[3:6].reshape(1,3), mu).flatten(), s[6]]) for s in cartesian_samples])
    initial_mc_states = np.array([np.hstack([s, initial_lam]) for s in initial_mee_states])

    history_optimal = propagate_and_control(models, 'optimal', t_eval, data, sol_nom, initial_mc_states, window_type)
    history_corrective = propagate_and_control(models, 'corrective', t_eval, data, sol_nom, initial_mc_states, window_type)

    plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, data["mu"], DU_km, window_type)

    print(f"\n--- Performance Summary Table ({window_type.upper()} Window) ---")
    r_final_nom, _ = mee2rv(*sol_nom.y[:6, -1], mu)
    
    def get_final_stats(history):
        final_states = history[-1]
        final_pos, _ = mee2rv(*final_states[:, :6].T, mu)
        mean_final_pos = np.mean(final_pos, axis=0)
        cov = np.cov(final_pos.T)
        final_dev = np.linalg.norm(mean_final_pos - r_final_nom) * DU_km
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
        vol = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        return final_dev, vol

    final_dev_opt, vol_opt = get_final_stats(history_optimal)
    final_dev_corr, vol_corr = get_final_stats(history_corrective)
    
    r_final_pert, _ = mee2rv(*sol_nom_perturbed.y[:6, -1], mu)
    final_dev_pert = np.linalg.norm(r_final_pert - r_final_nom) * DU_km

    summary_data = {
        "Strategy": ["Optimal (Open-Loop)", "Corrective (Closed-Loop)", "Perturbed Nominal (0.98F)"],
        "Final Deviation (km)": [f"{final_dev_opt:.2f}", f"{final_dev_corr:.2f}", f"{final_dev_pert:.2f}"],
        "Final Ellipsoid Volume (km^3)": [f"{vol_opt:.2e}", f"{vol_corr:.2e}", "N/A"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

# ==============================================================================
# === MAIN EXECUTION BLOCK =====================================================
# ==============================================================================
def main():
    try:
        models_max = { 'optimal': joblib.load("trained_model_max_optimal.pkl"), 'corrective': joblib.load("trained_model_max_corrective.pkl") }
        models_min = { 'optimal': joblib.load("trained_model_min_optimal.pkl"), 'corrective': joblib.load("trained_model_min_corrective.pkl") }
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        with open("stride_1440min/bundle_segment_widths.txt") as f:
            lines = f.readlines()[1:]
            times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input files are present.")
        return

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    
    # --- CORRECTED: Robust logic for finding min/max indices ---
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max, t_end_replan_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    
    search_end_idx = int(0.75 * len(times_arr))
    search_array = times_arr[:search_end_idx, 1]
    min_idx = np.argmin(search_array)
    t_start_min, t_end_replan_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data)
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data)
    
    print("\n[SUCCESS] All comparison simulations complete.")

if __name__ == "__main__":
    main()
