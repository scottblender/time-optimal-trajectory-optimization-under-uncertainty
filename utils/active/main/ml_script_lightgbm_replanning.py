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
NUM_MC_SAMPLES = 1000

# ==============================================================================
# === HELPER AND PLOTTING FUNCTIONS ============================================
# ==============================================================================

def compute_thrust_direction_vectorized(mu, mee_array, lam_array):
    """Computes optimal thrust direction from MEE states and costates."""
    p, f, g, h, k, L = [mee_array[:, i] for i in range(6)]
    lam_matrix = lam_array
    
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    C2 = 1 / w
    C3 = h * SinL - k * CosL

    A = np.zeros((mee_array.shape[0], 6, 3))
    A[:, 0, 1] = 2 * p * C2 * C1
    A[:, 1, 0] = C1 * SinL
    A[:, 1, 1] = C1 * C2 * ((w + 1) * CosL + f)
    A[:, 1, 2] = -C1 * (g / w) * C3
    A[:, 2, 0] = -C1 * CosL
    A[:, 2, 1] = C1 * C2 * ((w + 1) * SinL + g)
    A[:, 2, 2] = C1 * (f / w) * C3
    A[:, 3, 2] = C1 * s * CosL * C2 / 2
    A[:, 4, 2] = C1 * s * SinL * C2 / 2
    A[:, 5, 2] = C1 * C2 * C3
    
    mat = np.einsum('nij,nj->ni', A.transpose(0, 2, 1), lam_matrix[:,:6])
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    
    u_vecs = np.zeros_like(mat)
    non_zero_norm_mask = norm.flatten() > 1e-9
    u_vecs[non_zero_norm_mask] = mat[non_zero_norm_mask] / norm[non_zero_norm_mask]
    return u_vecs

def plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, mu, DU_km, window_type, uncertainty_level):
    """
    Plots the final XYZ deviation comparison in grayscale, including the perturbed nominal case.
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

    # 3. Create the plot with a grayscale palette
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Trajectory Deviation in XYZ Frame ({window_type.upper()} Window, {uncertainty_level} Uncertainty)', fontsize=16)
    
    labels = ['X Deviation [km]', 'Y Deviation [km]', 'Z Deviation [km]']
    
    for i in range(3):
        ax = axes[i]
        # UPDATED: Changed to grayscale colors with distinct line styles
        ax.plot(t_eval, delta_r_perturbed[:, i], color='#888888', linestyle='-', lw=2, label='Perturbed Nominal (0.98F)') # Medium gray, solid
        ax.plot(t_eval, delta_r_optimal[:, i], color='#444444', linestyle='--', lw=2, label='Optimal (Open-Loop)')    # Dark gray, dashed
        ax.plot(t_eval, delta_r_corrective[:, i], color='black', linestyle=':', lw=2, label='Corrective (Closed-Loop)')  # Black, dotted
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle=':')
        ax.legend()
        
    axes[-1].set_xlabel('Time [TU]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"final_xyz_deviation_{window_type}_{uncertainty_level}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n[Saved Plot] {fname}")

def generate_thrust_report(initial_thrusts, replanned_thrusts_log):
    """Generates a summary table for thrust vector predictions."""
    thrust_labels = ['u_x', 'u_y', 'u_z']
    initial_mean = np.mean(initial_thrusts, axis=0)
    initial_std = np.std(initial_thrusts, axis=0)
    
    report_data = {
        "Event": ["Initial Prediction"] * 3, "Component": thrust_labels,
        "Mean": initial_mean.tolist(), "Std Dev": initial_std.tolist()
    }
    
    if replanned_thrusts_log:
        first_replan = replanned_thrusts_log[0]
        replan_time = first_replan['time']
        replan_thrusts = np.array(first_replan['new_thrust_list'])
        replan_mean = np.mean(replan_thrusts, axis=0)
        replan_std = np.std(replan_thrusts, axis=0)
        
        report_data["Event"].extend([f"First Replan @ {replan_time:.2f} TU"] * 3)
        report_data["Component"].extend(thrust_labels)
        report_data["Mean"].extend(replan_mean.tolist())
        report_data["Std Dev"].extend(replan_std.tolist())

    report_df = pd.DataFrame(report_data)
    print(f"\n--- Thrust Vector Prediction Summary ---")
    print(report_df.to_string(index=False, float_format="%.4e"))

# ==============================================================================
# === CORE PROPAGATION AND CONTROL LOGIC =======================================
# ==============================================================================

def propagate_and_control(models, strategy, t_eval, data, sol_nom, initial_mc_states, window_type):
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    F_val = 0.98 * F_nom
    mc_states_current = np.copy(initial_mc_states)

    feature_cols_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    feature_cols_corrective = feature_cols_optimal + ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz', 'delta_m']

    tqdm.write(f"\n[INFO] Calculating initial SUB-NOMINAL OPTIMAL control policy...")
    initial_mee_states = mc_states_current[:, :7]
    initial_diag_vals = np.diag(np.cov(initial_mee_states.T))
    
    initial_costates_list = []
    model_optimal = models['optimal']
    for j in range(len(mc_states_current)):
        sample_state_mee_mass = mc_states_current[j][:7]
        features = np.hstack([t_eval[0], sample_state_mee_mass, initial_diag_vals])
        predicted_lam = model_optimal.predict(pd.DataFrame([features], columns=feature_cols_optimal))[0]
        mc_states_current[j][7:] = predicted_lam
        initial_costates_list.append(predicted_lam)
    
    initial_thrusts = compute_thrust_direction_vectorized(mu, initial_mee_states, np.array(initial_costates_list))

    history_mc_states = [np.copy(mc_states_current)]
    replanned_thrusts_log = []
    steps_since_replan = REPLAN_COOLDOWN_STEPS
    
    tqdm.write(f"--- Propagating with {strategy.upper()} Strategy ---")
    for i in tqdm(range(len(t_eval) - 1), desc=f"    -> Propagating ({strategy})", leave=False):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        mc_states_current = np.array([solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step]).y[:, -1] for s in mc_states_current])

        mean_mc_pos, _ = mee2rv(*np.mean(mc_states_current, axis=0)[:6], mu)
        r_nom_step, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_step) * DU_km

        if deviation_km > DEVIATION_THRESHOLD_KM and strategy == 'corrective' and steps_since_replan >= REPLAN_COOLDOWN_STEPS:
            steps_since_replan = 0
            mee_nom_full_curr = sol_nom.y[:7, i+1]
            lam_nom_curr, r_nom_curr, v_nom_curr = sol_nom.y[7:, i+1], *mee2rv(*mee_nom_full_curr[:6], mu)
            m_nom_curr = mee_nom_full_curr[6]
            mee_states_current = mc_states_current[:, :7]
            diag_vals_current = np.diag(np.cov(mee_states_current.T))

            first_replan = not replanned_thrusts_log
            if first_replan: replan_event = {'time': t_end_step, 'new_thrust_list': []}

            new_costates_list = []
            for j in range(len(mc_states_current)):
                sample_state_mee_mass = mc_states_current[j][:7]
                r_sample, v_sample = mee2rv(*sample_state_mee_mass[:6], mu)
                delta_r, delta_v = r_nom_curr - r_sample, v_nom_curr - v_sample
                delta_m = m_nom_curr - sample_state_mee_mass[6]
                features_corr = np.hstack([t_end_step, sample_state_mee_mass, diag_vals_current, delta_r.flatten(), delta_v.flatten(), delta_m])
                lam_correction = models['corrective'].predict(pd.DataFrame([features_corr], columns=feature_cols_corrective))[0]
                new_lam = lam_nom_curr + lam_correction if window_type == 'min' else lam_nom_curr - lam_correction
                mc_states_current[j][7:] = new_lam
                new_costates_list.append(new_lam)

            if first_replan:
                new_thrusts = compute_thrust_direction_vectorized(mu, mee_states_current, np.array(new_costates_list))
                replan_event['new_thrust_list'].extend(new_thrusts)
                replanned_thrusts_log.append(replan_event)
        
        history_mc_states.append(np.copy(mc_states_current))
        steps_since_replan += 1

    return np.array(history_mc_states), initial_thrusts, replanned_thrusts_log

# ==============================================================================
# === MAIN SIMULATION RUNNER ===================================================
# ==============================================================================

def run_comparison_simulation(models, t_start_replan, t_end_replan, window_type, data):
    print(f"\n{'='*25} Running Comparison for {window_type.upper()} WINDOW {'='*25}")
    
    mu, r_nom_hist, v_nom_hist = data["mu"], data["r_tr"], data["v_tr"]
    mass_nom_hist, lam_tr, t_vals = data["mass_tr"], data["lam_tr"], np.asarray(data["backTspan"][::-1])
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val, initial_lam = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start], lam_tr[idx_start]
    
    propagation_tu = t_end_replan - t_start_replan
    propagation_years = (propagation_tu * TU_SECONDS) / (365.25 * 24 * 3600)
    initial_velocity_kms = np.linalg.norm(v0) * DU_km / TU_SECONDS
    print(f"[INFO] Propagation Time: {propagation_tu:.2f} TU ({propagation_years:.2f} years)")
    print(f"[INFO] Initial Velocity: {initial_velocity_kms:.2f} km/s")

    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_mee_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val])
    nominal_state_start = np.hstack([nominal_mee_start, initial_lam])
    
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    sol_nom_perturbed = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], 0.98 * data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    # --- NEW: Generate Uncertainty from Physical Covariances ---
    VU_kms = DU_km / TU_SECONDS
    P_pos_km2 = np.eye(3) * 0.01
    P_vel_kms2 = np.eye(3) * 1e-10
    P_mass_kg2 = np.array([[1e-3]])
    P_pos = P_pos_km2 / (DU_km**2)
    P_vel = P_vel_kms2 / (VU_kms**2)
    P_mass = P_mass_kg2 / (4000**2) # Assuming nominal mass of 4000 kg for scaling

    P_rv = np.block([[P_pos, np.zeros((3,3))], [np.zeros((3,3)), P_vel]])
    rv_samples = np.random.multivariate_normal(np.hstack([r0, v0]), P_rv, size=NUM_MC_SAMPLES)
    mass_samples = np.random.normal(m0_val, np.sqrt(P_mass[0,0]), size=NUM_MC_SAMPLES)
    
    mee_samples = rv2mee(rv_samples[:, :3], rv_samples[:, 3:], mu)
    mee_mass_samples = np.hstack([mee_samples, mass_samples.reshape(-1, 1)])
    P_mee_base = np.cov(mee_mass_samples.T)

    uncertainty_scenarios = {"Original": P_mee_base, "Scaled_2x": 2 * P_mee_base}
    
    for level, P_mee in uncertainty_scenarios.items():
        print(f"\n--- Testing Uncertainty Level: {level} ---")
        initial_mee_states = np.random.multivariate_normal(nominal_mee_start, P_mee, size=NUM_MC_SAMPLES)
        initial_mc_states = np.array([np.hstack([s, initial_lam]) for s in initial_mee_states])

        t_start_check, t_end_check = t_eval[0], t_eval[1]
        next_states_check = np.array([solve_ivp(lambda t, x: odefunc(t, x, data["mu"], 0.98*data["F"], data["c"], data["m0"], data["g0"]), [t_start_check, t_end_check], s, t_eval=[t_end_check]).y[:, -1] for s in initial_mc_states])
        P_step1 = np.cov(next_states_check[:, :7].T)
        if np.trace(P_step1) > 2 * np.trace(P_mee):
            print(f"[WARNING] Covariance trace increased by >100% in one step. Dynamics may be unstable.")
            print(f"          Initial Trace: {np.trace(P_mee):.2e}, Propagated Trace: {np.trace(P_step1):.2e}")

        history_optimal, _, _ = propagate_and_control(models, 'optimal', t_eval, data, sol_nom, initial_mc_states, window_type)
        history_corrective, initial_thrusts, replan_log = propagate_and_control(models, 'corrective', t_eval, data, sol_nom, initial_mc_states, window_type)
        
        plot_final_deviations_xyz(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, data["mu"], DU_km, window_type, level)
        generate_thrust_report(initial_thrusts, replan_log)

        print(f"\n--- Performance Summary Table ({window_type.upper()} Window, {level} Uncertainty) ---")
        r_final_nom, _ = mee2rv(*sol_nom.y[:6, -1], mu)
        final_states_opt = history_optimal[-1]
        final_pos_opt = np.array([mee2rv(*s[:6], mu)[0] for s in final_states_opt])
        mean_final_pos_opt, cov_opt = np.mean(final_pos_opt, axis=0), np.cov(final_pos_opt.T)
        final_dev_opt = np.linalg.norm(mean_final_pos_opt - r_final_nom) * DU_km
        eigvals_opt = np.maximum(np.linalg.eigvalsh(cov_opt), 0)
        vol_opt = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals_opt)) * (DU_km**3)

        final_states_corr = history_corrective[-1]
        final_pos_corr = np.array([mee2rv(*s[:6], mu)[0] for s in final_states_corr])
        mean_final_pos_corr, cov_corr = np.mean(final_pos_corr, axis=0), np.cov(final_pos_corr.T)
        final_dev_corr = np.linalg.norm(mean_final_pos_corr - r_final_nom) * DU_km
        eigvals_corr = np.maximum(np.linalg.eigvalsh(cov_corr), 0)
        vol_corr = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals_corr)) * (DU_km**3)
        
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
    time_vals, num_times = times_arr[:, 0], len(times_arr)
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max, t_end_replan_max = time_vals[max(0, max_idx - 1)], time_vals[min(num_times - 1, max_idx + 1)]
    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == num_times - 1 else min_idx_raw
    t_start_min, t_end_replan_min = time_vals[max(0, min_idx - 1)], time_vals[min(num_times - 1, min_idx + 1)]
    
    # --- UPDATED ORDER ---
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data)
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data)
    
    print("\n[SUCCESS] All comparison simulations complete.")

if __name__ == "__main__":
    main()