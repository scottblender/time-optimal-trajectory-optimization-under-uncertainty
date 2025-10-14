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

# --- MISSION PARAMETERS (USER DEFINED) ---
DU_km = 696340.0  # Sun radius in km
g0_s = 9.81 / 1000 # g0 in km/s^2
TU_SECONDS = np.sqrt(DU_km / g0_s)

# --- ANALYSIS CONSTANTS ---
DEVIATION_THRESHOLD_KM = 100.0
REPLAN_COOLDOWN_STEPS = 50 # Cooldown in steps before another replan is allowed

### NEW PLOTTING FUNCTION FOR FINAL ANALYSIS IN XYZ FRAME ###
def plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, mu, DU_km, window_type, uncertainty_level):
    """
    Plots the final XYZ deviation comparison including the perturbed nominal case.
    """
    # 1. Get Cartesian histories for all trajectories
    r_nom_hist, _ = mee2rv(*sol_nom.y[:6, :], mu)
    r_nom_perturbed_hist, _ = mee2rv(*sol_nom_perturbed.y[:6, :], mu)
    
    mean_states_optimal = np.mean(history_optimal, axis=1)
    r_optimal_hist, _ = mee2rv(*mean_states_optimal[:, :6].T, mu)

    mean_states_corrective = np.mean(history_corrective, axis=1)
    r_corrective_hist, _ = mee2rv(*mean_states_corrective[:, :6].T, mu)

    # 2. Calculate deviation vectors in the inertial XYZ frame relative to the true nominal
    delta_r_optimal = (r_optimal_hist - r_nom_hist) * DU_km
    delta_r_corrective = (r_corrective_hist - r_nom_hist) * DU_km
    delta_r_perturbed = (r_nom_perturbed_hist - r_nom_hist) * DU_km

    # 3. Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Trajectory Deviation in XYZ Frame ({window_type.upper()} Window, {uncertainty_level} Uncertainty)', fontsize=16)
    
    labels = ['X Deviation [km]', 'Y Deviation [km]', 'Z Deviation [km]']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(t_eval, delta_r_perturbed[:, i], 'g-', lw=2, label='Perturbed Nominal (0.98F)')
        ax.plot(t_eval, delta_r_optimal[:, i], 'b--', lw=2, label='Optimal (Open-Loop)')
        ax.plot(t_eval, delta_r_corrective[:, i], 'r:', lw=2, label='Corrective (Closed-Loop)')
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle=':')
        ax.legend()
        
    axes[-1].set_xlabel('Time [TU]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"final_xyz_deviation_{window_type}_{uncertainty_level}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n[Saved Plot] {fname}")


def propagate_and_control(models, strategy, t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type):
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    F_val = 0.98 * F_nom
    t_eval = sol_nom.t

    mc_states_current = np.copy(initial_mc_states)

    feature_cols_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    feature_cols_corrective = feature_cols_optimal + ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz', 'delta_m']

    tqdm.write(f"\n[INFO] Calculating initial SUB-NOMINAL OPTIMAL control policy for each sample in {strategy.upper()} strategy...")
    initial_mee_states = mc_states_current[:, :7]
    initial_devs = initial_mee_states - np.mean(initial_mee_states, axis=0)
    initial_diag_vals = np.diag(np.cov(initial_devs.T))

    model_optimal = models['optimal']
    for j in range(len(mc_states_current)):
        sample_state_mee_mass = mc_states_current[j][:7]
        features = np.hstack([t_start_replan, sample_state_mee_mass, initial_diag_vals])
        predicted_lam = model_optimal.predict(pd.DataFrame([features], columns=feature_cols_optimal))[0]
        mc_states_current[j][7:] = predicted_lam

    history_mc_states = [np.copy(mc_states_current)]
    steps_since_replan = REPLAN_COOLDOWN_STEPS

    tqdm.write("\n" + f"--- Propagating with {strategy.upper()} Strategy ---")

    for i in tqdm(range(len(t_eval) - 1), desc=f"    -> Propagating ({strategy})", leave=False):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]

        next_states_list = []
        for s in mc_states_current:
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step])
            next_states_list.append(sol.y[:, -1])
        mc_states_current = np.array(next_states_list)

        mean_mc_pos, _ = mee2rv(*np.mean(mc_states_current, axis=0)[:6], mu)
        r_nom_step, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_step) * DU_km

        if deviation_km > DEVIATION_THRESHOLD_KM and strategy == 'corrective' and steps_since_replan >= REPLAN_COOLDOWN_STEPS:
            steps_since_replan = 0
            
            mee_nom_full_curr = sol_nom.y[:7, i+1]
            lam_nom_curr = sol_nom.y[7:, i+1]
            r_nom_curr, v_nom_curr = mee2rv(*mee_nom_full_curr[:6], mu)
            m_nom_curr = mee_nom_full_curr[6]
            
            mee_states_current = mc_states_current[:, :7]
            diag_vals_current = np.diag(np.cov(mee_states_current.T))

            for j in range(len(mc_states_current)):
                sample_state_mee_mass = mc_states_current[j][:7]
                model_corrective = models['corrective']

                r_sample, v_sample = mee2rv(*sample_state_mee_mass[:6], mu)
                delta_r = r_nom_curr - r_sample
                delta_v = v_nom_curr - v_sample
                delta_m = m_nom_curr - sample_state_mee_mass[6]
                
                features_corr = np.hstack([t_end_step, sample_state_mee_mass, diag_vals_current, delta_r.flatten(), delta_v.flatten(), delta_m])
                lam_correction = model_corrective.predict(pd.DataFrame([features_corr], columns=feature_cols_corrective))[0]
                
                if window_type == 'min':
                    mc_states_current[j][7:] = lam_nom_curr + lam_correction
                else:
                    mc_states_current[j][7:] = lam_nom_curr - lam_correction
        
        history_mc_states.append(np.copy(mc_states_current))
        steps_since_replan += 1

    return np.array(history_mc_states)

def run_comparison_simulation(models, t_start_replan, t_end_replan, window_type, data):
    print(f"\n{'='*25} Running Comparison for {window_type.upper()} WINDOW {'='*25}")
    
    mu = data["mu"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val, initial_lam = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start], lam_tr[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_mee_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val])
    nominal_state_start = np.hstack([nominal_mee_start, initial_lam])
    
    # Propagate baseline trajectories once
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    sol_nom_perturbed = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], 0.98 * data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    # --- DEFINE UNCERTAINTY SCENARIOS ---
    try:
        base_cov_diag = np.loadtxt(f"cov_diag_{window_type}.txt")
        # Ensure positivity for covariance matrix
        base_cov_diag[base_cov_diag < 1e-12] = 1e-12
    except IOError:
        print(f"[ERROR] cov_diag_{window_type}.txt not found. Please run training script first.")
        return
        
    uncertainty_scenarios = {
        "Original": np.diag(base_cov_diag),
        "Scaled_2x": np.diag(2 * base_cov_diag)
    }
    num_samples = 1000

    # --- LOOP THROUGH UNCERTAINTY SCENARIOS ---
    for level, P_mee in uncertainty_scenarios.items():
        print(f"\n--- Testing Uncertainty Level: {level} ---")
        
        # Generate MC samples in MEE space for this uncertainty level
        initial_mee_states = np.random.multivariate_normal(nominal_mee_start, P_mee, size=num_samples)
        initial_mc_states = np.array([np.hstack([s, initial_lam]) for s in initial_mee_states])

        history_optimal = propagate_and_control(models, 'optimal', t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type)
        history_corrective = propagate_and_control(models, 'corrective', t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type)
        
        # --- Generate Plots ---
        plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, data["mu"], DU_km, window_type, level)
        
        # --- Generate Tabular Report ---
        print(f"\n--- Performance Summary Table ({window_type.upper()} Window, {level} Uncertainty) ---")
        r_final_nom, _ = mee2rv(*sol_nom.y[:6, -1], mu)

        # Optimal stats
        final_states_opt = history_optimal[-1]
        final_pos_opt = np.array([mee2rv(*s[:6], mu)[0] for s in final_states_opt])
        mean_final_pos_opt = np.mean(final_pos_opt, axis=0)
        final_dev_opt = np.linalg.norm(mean_final_pos_opt - r_final_nom) * DU_km
        cov_opt = np.cov(final_pos_opt.T)
        eigvals_opt = np.maximum(np.linalg.eigvalsh(cov_opt), 0)
        vol_opt = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals_opt)) * (DU_km**3)

        # Corrective stats
        final_states_corr = history_corrective[-1]
        final_pos_corr = np.array([mee2rv(*s[:6], mu)[0] for s in final_states_corr])
        mean_final_pos_corr = np.mean(final_pos_corr, axis=0)
        final_dev_corr = np.linalg.norm(mean_final_pos_corr - r_final_nom) * DU_km
        cov_corr = np.cov(final_pos_corr.T)
        eigvals_corr = np.maximum(np.linalg.eigvalsh(cov_corr), 0)
        vol_corr = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals_corr)) * (DU_km**3)
        
        # Perturbed Nominal stats
        r_final_pert, _ = mee2rv(*sol_nom_perturbed.y[:6, -1], mu)
        final_dev_pert = np.linalg.norm(r_final_pert - r_final_nom) * DU_km

        # Create and print table
        summary_data = {
            "Strategy": ["Optimal (Open-Loop)", "Corrective (Closed-Loop)", "Perturbed Nominal (0.98F)"],
            "Final Deviation (km)": [f"{final_dev_opt:.2f}", f"{final_dev_corr:.2f}", f"{final_dev_pert:.2f}"],
            "Final Ellipsoid Volume (km^3)": [f"{vol_opt:.2e}", f"{vol_corr:.2e}", "N/A"]
        }
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

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
    num_times = len(time_vals)

    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max = time_vals[max(0, max_idx - 1)]
    t_end_replan_max = time_vals[min(num_times - 1, max_idx + 1)]

    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == num_times - 1 else min_idx_raw
    t_start_min = time_vals[max(0, min_idx - 1)]
    t_end_replan_min = time_vals[min(num_times - 1, min_idx + 1)]
    
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data)
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data)
    
    print("\n[SUCCESS] All comparison simulations complete.")

if __name__ == "__main__":
    main()