# ml_script_tcn_replanning.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import torch
from torch import nn

# === Setup ===
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cpu")

# === Local imports ===
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

# === TCN Model Definition (must match training script) ===
class TCN_MANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super().__init__()
        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            input_size = hidden_size
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.controller = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_size, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.controller(x)
        return x.transpose(1, 2)

# === Constants & Hyperparameters (from training script) ===
SEQ_LENGTH = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 4
INPUT_SIZE = 19 # t, 7 MEEs, 7 cov diags, score, 3 delta_r
OUTPUT_SIZE = 7  # 7 costates
DEVIATION_THRESHOLD_KM = 100.0

# --- HELPER & PLOTTING FUNCTIONS (from LightGBM script) ---
def compute_thrust_direction(mu, mee, lam):
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

def plot_full_replan_trajectory(r_nom, r_fval, r_mc_mean, initial_quiver, first_replan_quiver, window_type, level, t_start):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_nom[:, 0], r_nom[:, 1], color='0.0', linestyle='-', linewidth=1.5, label='Nominal (F_nom)', alpha=0.6)
    ax.plot(r_fval[:, 0], r_fval[:, 1], color='0.5', linestyle='--', linewidth=1.2, label='Nominal (0.9*F_nom)', alpha=0.6)
    ax.plot(r_mc_mean[:, 0], r_mc_mean[:, 1], color='0.2', linestyle=':', linewidth=2.0, label='MC Mean (Closed-Loop)')
    ax.scatter(r_nom[0, 0], r_nom[0, 1], c='0.2', marker='o', s=100, alpha=0.7, label='Start', zorder=5)
    ax.scatter(r_nom[-1, 0], r_nom[-1, 1], c='0.0', marker='X', s=120, alpha=0.7, label='Nominal End', zorder=5)
    if initial_quiver:
        pos, thrust = initial_quiver
        ax.quiver(pos[0], pos[1], thrust[0], thrust[1], color='blue', alpha=0.9, scale=15, width=0.008, label='Initial Control')
    if first_replan_quiver:
        pos, thrust = first_replan_quiver
        ax.quiver(pos[0], pos[1], thrust[0], thrust[1], color='magenta', alpha=0.9, scale=15, width=0.008, label='First Replan Control')
    ax.set_xlabel('X [DU]'); ax.set_ylabel('Y [DU]')
    ax.grid(True, linestyle=':'); ax.legend(loc='upper right')
    x_min, x_max = np.min(r_nom[:, 0]), np.max(r_nom[:, 0]); y_min, y_max = np.min(r_nom[:, 1]), np.max(r_nom[:, 1])
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_pad, x_max + x_pad); ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fname = f"full_replan_trajectory_{t_start:.3f}_{level}_{window_type}_tcn.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    print(f"\n  [Saved Plot] {fname}")

# === MAIN SIMULATION FUNCTION (adapted for TCN) ===
def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data, scaler_X, scaler_y):
    print(f"\n--- Running TCN Closed-Loop Sim for {window_type.upper()} WINDOW ---")
    
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom_hist, v_nom_hist, mass_nom_hist, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    DU_km = 696340.0
    F_val = 0.98 * F_nom
    
    # Use fixed covariance diagonals for min/max cases
    diag_mins = np.array([9.098224e+01, 7.082445e-04, 6.788599e-04, 1.376023e-08, 2.346605e-08, 5.885859e-08, 1.000000e-04])
    diag_maxs = np.array([1.977489e+03, 6.013501e-03, 5.173225e-03, 4.284912e-04, 1.023625e-03, 6.818500e+00, 1.000000e-04])
    diag_vals = diag_maxs if window_type == 'max' else diag_mins
    
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, lam_tr[idx_start]])
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)
    replan_r_nom, _ = mee2rv(*sol_nom.y[:6], mu)
    sol_fval = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval)

    P_cart = np.diag(np.concatenate([np.diag(np.eye(3)*0.01/(DU_km**2)), np.diag(np.eye(3)*1e-10/((DU_km/86400)**2)), [1e-3/(4000**2)]]))
    shared_samples = np.random.multivariate_normal(np.hstack([r0, v0, m0_val]), P_cart, size=1000)
    
    mc_particles = [] # Will store dicts: {'state': array, 'history': deque}

    print("[INFO] Initializing MC particles and histories...")
    for s in tqdm(shared_samples, desc="  -> Initializing samples"):
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        mee_s = rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten()
        state_s = np.hstack([mee_s, m_s])
        delta_r_s = r_s - r0
        score_s = np.linalg.norm(delta_r_s)
        
        # Create initial feature vector and history
        initial_feature = np.hstack([t_start_replan, state_s, diag_vals, score_s, delta_r_s])
        history = deque([initial_feature] * SEQ_LENGTH, maxlen=SEQ_LENGTH)
        
        # Initial prediction
        x_seq_scaled = scaler_X.transform(np.array(history))
        x_tensor = torch.tensor(x_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            lam_scaled = model(x_tensor)[:, -1, :].cpu().numpy()
        lam = scaler_y.inverse_transform(lam_scaled)[0]
        
        mc_particles.append({'state': np.hstack([state_s, lam]), 'history': history})

    history_mc_states = [[p['state'] for p in mc_particles]]
    first_replan_quiver_data, initial_quiver_data = None, None # Placeholder
    
    tqdm.write("Step |  Time (TU) | Deviation (km) | Action")
    tqdm.write("-" * 50)
    
    for i in tqdm(range(len(t_eval) - 1), desc="    -> Propagating"):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        for p in mc_particles:
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_step, t_end_step], p['state'], t_eval=[t_end_step])
            p['state'] = sol.y[:, -1]
        
        current_mc_positions = np.array([mee2rv(*p['state'][:6], mu)[0].flatten() for p in mc_particles])
        mean_mc_pos = np.mean(current_mc_positions, axis=0)
        deviation_km = np.linalg.norm(mean_mc_pos - replan_r_nom[i+1]) * DU_km
        log_action = "---"

        if deviation_km > DEVIATION_THRESHOLD_KM:
            log_action = "REPLAN"
            mee_nom_current = sol_nom.y[:6, i+1]
            r_nom_current, _ = mee2rv(*mee_nom_current, mu)

            for p in mc_particles:
                # Update history with the latest state
                current_mee_state = p['state'][:7]
                r_sample, _ = mee2rv(*current_mee_state[:6], mu)
                delta_r = r_sample - r_nom_current
                score = np.linalg.norm(delta_r)
                new_feature = np.hstack([t_end_step, current_mee_state, diag_vals, score, delta_r])
                p['history'].append(new_feature)

                # Re-predict costates with updated history
                x_seq_scaled = scaler_X.transform(np.array(p['history']))
                x_tensor = torch.tensor(x_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    lam_scaled = model(x_tensor)[:, -1, :].cpu().numpy()
                new_lam = scaler_y.inverse_transform(lam_scaled)[0]
                p['state'][7:] = new_lam # Update costates
        
        history_mc_states.append([p['state'] for p in mc_particles])
        tqdm.write(f"{i+1:4d} | {t_end_step:10.2f} | {deviation_km:14.2f} | {log_action}")

    history_mc_r = [np.array([mee2rv(*s[:6], mu)[0].flatten() for s in states_at_t]) for states_at_t in history_mc_states]
    mc_mean_r = np.mean(np.array(history_mc_r), axis=1)
    
    plot_full_replan_trajectory(replan_r_nom, sol_fval.y[:3].T, mc_mean_r, initial_quiver_data, first_replan_quiver_data, window_type, "random1", t_start_replan)

# --- SCRIPT ENTRY POINT ---
def main():
    # Load models
    model_max = TCN_MANN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(device)
    model_max.load_state_dict(torch.load("trained_model_tcn_max.pt", map_location=device))
    model_max.eval()

    model_min = TCN_MANN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(device)
    model_min.load_state_dict(torch.load("trained_model_tcn_min.pt", map_location=device))
    model_min.eval()
    
    # Load scalers and data
    scaler_X = joblib.load("scaler_tcn_X.pkl")
    scaler_y = joblib.load("scaler_tcn_y.pkl")
    data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
    
    with open("stride_1440min/bundle_segment_widths.txt") as f:
        lines = f.readlines()[1:]
        times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max, t_end_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_min, t_end_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]
    
    run_replanning_simulation(model_min, t_start_min, t_end_min, 'min', data, scaler_X, scaler_y)
    run_replanning_simulation(model_max, t_start_max, t_end_max, 'max', data, scaler_X, scaler_y)
    
    print("\n[SUCCESS] All TCN simulations complete.")

if __name__ == "__main__":
    main()