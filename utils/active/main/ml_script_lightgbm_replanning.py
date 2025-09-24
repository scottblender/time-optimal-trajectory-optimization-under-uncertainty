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

# --- ANALYSIS CONSTANT ---
DEVIATION_THRESHOLD_KM = 100.0

# --- HELPER & PLOTTING FUNCTIONS ---
replan_history = []
def plot_full_replan_trajectory(r_nom, r_fval, r_mc_mean, window_type, level, t_start):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_nom[:, 0], r_nom[:, 1], color='0.0', linestyle='-', linewidth=1.5, label='Nominal (F_nom)', alpha=0.6)
    ax.plot(r_fval[:, 0], r_fval[:, 1], color='0.5', linestyle='--', linewidth=1.2, label='Nominal (F_val)', alpha=0.6)
    ax.plot(r_mc_mean[:, 0], r_mc_mean[:, 1], color='0.2', linestyle=':', linewidth=2.0, label='MC Mean (Closed-Loop)')
    ax.scatter(r_nom[0, 0], r_nom[0, 1], c='0.2', marker='o', s=100, alpha=0.7, label='Start', zorder=5)
    ax.scatter(r_nom[-1, 0], r_nom[-1, 1], c='0.0', marker='X', s=120, alpha=0.7, label='Nominal End', zorder=5)
    replan_points = np.array([p for p in replan_history if p is not None])
    if len(replan_points) > 0:
        ax.scatter(replan_points[:, 0], replan_points[:, 1], c='magenta', marker='*', s=60, alpha=0.8, label='Replan Event')
    ax.set_xlabel('X [DU]'); ax.set_ylabel('Y [DU]'); ax.grid(True, linestyle=':'); ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box'); fig.tight_layout()
    fname = f"full_replan_trajectory_{t_start:.3f}_{level}_{window_type}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname)); plt.close()
    print(f"\n  [Saved Plot] {fname}")

# --- MAIN SIMULATION FUNCTION ---
def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data):
    global replan_history
    replan_history = []
    
    print(f"\n--- Running Closed-Loop Sim for {window_type.upper()} WINDOW ---")
    
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    DU_km = 696340.0
    F_val = 0.98 * F_nom
    alpha, beta = 1.0, 0.1

    diag_mins = np.array([9.098224e+01, 7.082445e-04, 6.788599e-04, 1.376023e-08, 2.346605e-08, 5.885859e-08, 1.000000e-04])
    diag_maxs = np.array([1.977489e+03, 6.013501e-03, 5.173225e-03, 4.284912e-04, 1.023625e-03, 6.818500e+00, 1.000000e-04])
    diag_vals = diag_maxs if window_type == 'max' else diag_mins
    
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, lam_tr[idx_start]])
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    sol_fval = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    
    feature_cols = ['t','p','f','g','h','k','L','mass','c1','c2','c3','c4','c5','c6','c7','weighted_error']

    P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/86400)**2)), [1e-3/(4000**2)]]))
    shared_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=1000)
    
    mc_states_current = []
    for s in tqdm(shared_samples, desc="  -> Initializing samples"):
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        state_s = np.hstack([rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten(), m_s])
        delta_r, delta_v = r_s - r0, v_s - v0
        weighted_error = (alpha * np.sum(delta_r**2)) + (beta * np.sum(delta_v**2))
        features = np.hstack([t_start_replan, state_s, diag_vals, weighted_error])
        lam_s = model.predict(pd.DataFrame([features], columns=feature_cols))[0]
        mc_states_current.append(np.hstack([state_s, lam_s]))

    history_mc_states = [np.array(mc_states_current)]
    
    ## --- RESTORED DIAGNOSTIC PRINTOUT ---
    tqdm.write("Step |  Time (TU) | Deviation (km) | Ellipsoid Vol (km^3) | Action")
    tqdm.write("-" * 70)
    ## --- END RESTORED ---
    
    for i in tqdm(range(len(t_eval) - 1), desc="    -> Propagating"):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        mc_states_next_step = [solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], s, t_eval=[t_end_step]).y[:, -1] for s in mc_states_current]
        mc_states_current = mc_states_next_step
        
        current_mc_positions = np.array([mee2rv(*s[:6], mu)[0] for s in mc_states_current])
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        
        mee_nom_current = sol_nom.y[:6, i+1]
        r_nom_current, v_nom_current = mee2rv(*mee_nom_current, mu)
        
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_current) * DU_km
        
        ## --- RESTORED DIAGNOSTIC PRINTOUT ---
        cov_mc = np.cov(current_mc_positions.T)
        eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
        volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        log_action = "---"
        ## --- END RESTORED ---
        
        replan_point = None
        if deviation_km > DEVIATION_THRESHOLD_KM:
            log_action = "REPLAN" # Restored
            replan_point = mean_mc_pos[:2]
            for j in range(len(mc_states_current)):
                sample_state = mc_states_current[j][:7]
                r_sample, v_sample = mee2rv(*sample_state[:6], mu)
                delta_r, delta_v = r_sample - r_nom_current, v_sample - v_nom_current
                weighted_error = (alpha * np.sum(delta_r**2)) + (beta * np.sum(delta_v**2))
                features = np.hstack([t_end_step, sample_state, diag_vals, weighted_error])
                predicted_lam = model.predict(pd.DataFrame([features], columns=feature_cols))[0]
                mc_states_current[j][7:] = predicted_lam
        
        replan_history.append(replan_point)
        history_mc_states.append(np.array(mc_states_current))
        
        ## --- RESTORED DIAGNOSTIC PRINTOUT ---
        tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {volume_km3:20.2e} | {log_action}")
        ## --- END RESTORED ---

    r_nom_segment, _ = mee2rv(*sol_nom.y[:6,:].T, mu)
    r_fval_segment, _ = mee2rv(*sol_fval.y[:6,:].T, mu)
    history_mc_r = [np.array([mee2rv(*s[:6], mu)[0] for s in states_at_t]) for states_at_t in history_mc_states]
    mc_mean_r = np.mean(np.array(history_mc_r), axis=1)
    
    plot_full_replan_trajectory(r_nom_segment, r_fval_segment, mc_mean_r, window_type, "random1", t_start_replan)

# --- SCRIPT ENTRY POINT ---
def main():
    model_max = joblib.load("trained_model_max.pkl")
    model_min = joblib.load("trained_model_min.pkl")
    data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
    with open("stride_1440min/bundle_segment_widths.txt") as f: lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    max_idx = int(np.argmax(times_arr[:, 1])); t_start_max, t_end_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    min_idx_raw = int(np.argmin(times_arr[:, 1])); min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_min, t_end_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    run_replanning_simulation(model_min, t_start_min, t_end_min, 'min', data)
    run_replanning_simulation(model_max, t_start_max, t_end_max, 'max', data)
    
    print("\n[SUCCESS] All simulations complete.")

if __name__ == "__main__":
    main()