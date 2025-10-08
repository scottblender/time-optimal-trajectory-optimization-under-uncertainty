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

# --- HELPER AND PLOTTING FUNCTIONS FOR THRUST VECTOR ---
def compute_thrust_direction(mu, F, mee, lam):
    p, f, g, h, k, L = mee[:-1]
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
    return (mat.flatten() / norm) if norm > 1e-9 else np.zeros(3)

def plot_comparison_profiles(t_eval, history_optimal, history_corrective, sol_nom, mu, F_nom, window_type):
    def get_thrust_history(history_states, is_nominal=False):
        thrust_history = []
        for i in range(len(history_states)):
            states_at_t = history_states[i]
            if is_nominal:
                nominal_state = states_at_t
                mean_mee_mass = nominal_state[:7]
                mean_lam = nominal_state[7:]
            else:
                mean_state = np.mean(states_at_t, axis=0)
                mean_mee_mass = mean_state[:7]
                mean_lam = mean_state[7:]
            thrust_vector = compute_thrust_direction(mu, F_nom, mean_mee_mass, mean_lam)
            thrust_history.append(thrust_vector)
        return np.array(thrust_history)

    thrust_history_nominal = get_thrust_history(sol_nom.y.T, is_nominal=True)
    thrust_history_optimal = get_thrust_history(history_optimal)
    thrust_history_corrective = get_thrust_history(history_corrective)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Thrust Control Strategy Comparison ({window_type.upper()} Window)', fontsize=16)
    labels = ['$u_x$', '$u_y$', '$u_z$']

    for i in range(3):
        ax = axes[i]
        ax.plot(t_eval, thrust_history_nominal[:, i], color='gray', linestyle='-.', lw=1.5, label='Original Nominal')
        ax.plot(t_eval, thrust_history_optimal[:, i], color='blue', linestyle='--', lw=2, label='Optimal (Open-Loop)')
        ax.plot(t_eval, thrust_history_corrective[:, i], color='red', linestyle=':', lw=2, label='Corrective (Closed-Loop)')
        ax.set_title(f'Thrust Component {labels[i]}')
        ax.set_ylabel('Thrust Value')
        ax.grid(True, linestyle=':')
        ax.legend()
    axes[-1].set_xlabel('Time [TU]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fname = f"thrust_comparison_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n[Saved Plot] {fname}")

def propagate_and_control(models, strategy, t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type):
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    F_val = 0.98 * F_nom
    t_eval = sol_nom.t
    
    mc_states_current = np.copy(initial_mc_states)
    
    feature_cols_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    feature_cols_corrective = feature_cols_optimal + ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz']

    # --- INITIAL REPLAN AT T=0 ---
    tqdm.write(f"\n[INFO] Calculating initial SUB-NOMINAL OPTIMAL control policy for {strategy.upper()} strategy...")
    initial_mee_states = mc_states_current[:, :7]
    initial_devs = initial_mee_states - np.mean(initial_mee_states, axis=0)
    initial_diag_vals = np.diag(np.einsum('ji,jk->ik', initial_devs, initial_devs) / (len(initial_mee_states) - 1))
    
    model_optimal = models['optimal']
    for j in range(len(mc_states_current)):
        sample_state_mee_mass = mc_states_current[j][:7]
        features = np.hstack([t_start_replan, sample_state_mee_mass, initial_diag_vals])
        predicted_lam = model_optimal.predict(pd.DataFrame([features], columns=feature_cols_optimal))[0]
        mc_states_current[j][7:] = predicted_lam

    history_mc_states = [np.copy(mc_states_current)]
    
    ### MODIFICATION HERE: ADD COOLDOWN COUNTER ###
    # Initialize counter to allow a replan on the first possible step
    steps_since_replan = REPLAN_COOLDOWN_STEPS 

    tqdm.write("\n" + f"--- Propagating with {strategy.upper()} Strategy ---")
    tqdm.write("Step |  Time (TU) | Deviation (km) | Ellipsoid Vol (km^3) | Action")
    tqdm.write("-" * 70)

    for i in tqdm(range(len(t_eval) - 1), desc=f"    -> Propagating ({strategy})"):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        next_states_list = []
        for s in mc_states_current:
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step])
            next_states_list.append(sol.y[:, -1])
        mc_states_current = np.array(next_states_list)
        
        current_mc_positions = np.array([mee2rv(*s[:6], mu)[0] for s in mc_states_current])
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        r_nom_step, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_step) * DU_km
        
        cov_cartesian = np.cov(current_mc_positions.T)
        eigvals = np.maximum(np.linalg.eigvalsh(cov_cartesian), 0)
        volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        log_action = "---"
        
        ### MODIFICATION HERE: CHECK COOLDOWN ###
        if deviation_km > DEVIATION_THRESHOLD_KM and strategy == 'corrective' and steps_since_replan >= REPLAN_COOLDOWN_STEPS:
            log_action = "REPLAN"
            steps_since_replan = 0 # Reset cooldown counter
            
            mee_nom_full_curr = sol_nom.y[:7, i+1]
            lam_nom_curr = sol_nom.y[7:, i+1]
            r_nom_curr, v_nom_curr = mee2rv(*mee_nom_full_curr[:6], mu)
            
            thrust_nominal = compute_thrust_direction(mu, F_nom, mee_nom_full_curr, lam_nom_curr)
            
            mee_states_current = mc_states_current[:, :7]
            devs_current = mee_states_current - np.mean(mee_states_current, axis=0)
            diag_vals_current = np.diag(np.einsum('ji,jk->ik', devs_current, devs_current) / (len(mc_states_current) - 1))

            for j in range(len(mc_states_current)):
                lam_before_replan = np.copy(mc_states_current[j][7:])
                sample_state_mee_mass = mc_states_current[j][:7]
                
                model_corrective = models['corrective']

                r_sample, v_sample = mee2rv(*sample_state_mee_mass[:6], mu)
                delta_r = r_nom_curr - r_sample
                delta_v = v_nom_curr - v_sample
                features_corr = np.hstack([t_end_step, sample_state_mee_mass, diag_vals_current, delta_r.flatten(), delta_v.flatten()])
                lam_correction = model_corrective.predict(pd.DataFrame([features_corr], columns=feature_cols_corrective))[0]
                
                if window_type == 'min':
                    mc_states_current[j][7:] = lam_nom_curr + lam_correction
                else: 
                    mc_states_current[j][7:] = lam_nom_curr - lam_correction

                if j == 0:
                    lam_after_replan = mc_states_current[j][7:]
                    thrust_before = compute_thrust_direction(mu, F_nom, sample_state_mee_mass, lam_before_replan)
                    thrust_after = compute_thrust_direction(mu, F_nom, sample_state_mee_mass, lam_after_replan)
                    
                    tqdm.write("    [REPLAN DETAIL] Thrust Change:")
                    tqdm.write(f"      Nominal Thrust: {np.array2string(thrust_nominal, precision=4, floatmode='fixed')}")
                    tqdm.write(f"      Thrust Before:  {np.array2string(thrust_before, precision=4, floatmode='fixed')}")
                    tqdm.write(f"      Thrust After:   {np.array2string(thrust_after, precision=4, floatmode='fixed')}")
        
        history_mc_states.append(np.copy(mc_states_current))
        tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {volume_km3:20.2e} | {log_action}")
        
        ### MODIFICATION HERE: INCREMENT COOLDOWN COUNTER ###
        steps_since_replan += 1

    return np.array(history_mc_states)

def run_comparison_simulation(models, t_start_replan, t_end_replan, window_type, data):
    print(f"\n--- Running Comparison for {window_type.upper()} WINDOW ---")
    
    mu = data["mu"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val, initial_lam = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start], lam_tr[idx_start]
    
    v0_kms = v0 * DU_km / TU_SECONDS
    print("-" * 60)
    print(f"[VELOCITY INFO] Initial Velocity for {window_type.upper()} Window (at t={t_start_replan:.2f}):")
    print(f"  > In canonical units (DU/TU): {v0}")
    print(f"  > In km/s (using TU={TU_SECONDS:.2f}s): {v0_kms}")
    print(f"  > Speed: {np.linalg.norm(v0_kms):.2f} km/s")
    print("-" * 60)
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, initial_lam])
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)

    P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/86400)**2)), [1e-3/(4000**2)]]))
    num_samples = 1000
    cartesian_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=num_samples)
    initial_mee_states = np.array([np.hstack([rv2mee(s[:3].reshape(1,3), s[3:6].reshape(1,3), mu).flatten(), s[6]]) for s in cartesian_samples])
    initial_mc_states = np.array([np.hstack([s, initial_lam]) for s in initial_mee_states])

    history_optimal = propagate_and_control(models, 'optimal', t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type)
    history_corrective = propagate_and_control(models, 'corrective', t_start_replan, t_end_replan, data, sol_nom, initial_mc_states, window_type)

    plot_comparison_profiles(t_eval, history_optimal, history_corrective, sol_nom, data["mu"], data["F"], window_type)

def main():
    try:
        models_max = {
            'optimal': joblib.load("trained_model_max_optimal.pkl"),
            'corrective': joblib.load("trained_model_max_corrective.pkl")
        }
        models_min = {
            'optimal': joblib.load("trained_model_min_optimal.pkl"),
            'corrective': joblib.load("trained_model_min_corrective.pkl")
        }
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        with open("stride_1440min/bundle_segment_widths.txt") as f:
            lines = f.readlines()[1:]
            times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input files are present.")
        return

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    max_idx = int(np.argmax(times_arr[:, 1])); t_start_max, t_end_replan_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    min_idx_raw = int(np.argmin(times_arr[:, 1])); min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_min, t_end_replan_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data)
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data)
    
    print("\n[SUCCESS] All comparison simulations complete.")

if __name__ == "__main__":
    main()