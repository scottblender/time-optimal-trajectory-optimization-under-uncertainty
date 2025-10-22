import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
import odefunc

# === Constants ===
OUTPUT_FILENAME = "processed_training_data.pkl"

# === Load segment times from width file ===
print("[INFO] Identifying training time windows...")
with open("stride_1440min/bundle_segment_widths.txt") as f:
    lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    time_vals = times_arr[:, 0]

    search_end_idx = int(0.75 * len(times_arr))
    search_array = times_arr[:search_end_idx, 1]
    min_idx = np.argmin(search_array)
    max_idx = int(np.argmax(times_arr[:, 1]))

    n = 50
    t_train_neighbors_max = time_vals[max(0, max_idx - n): max_idx + n + 1]
    t_train_neighbors_min = time_vals[max(0, min_idx - n): min_idx + n + 1]

# === ROBUST DATA LOADING (DEFINITIVE FIX) ===
X_train_max_list, y_train_max_list = [], []
X_train_min_list, y_train_min_list = [], []
batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))

# 1. Round the 101 target times to a set precision to eliminate floating-point ambiguity.
precision = 6
rounded_targets_max = np.unique(np.round(t_train_neighbors_max, precision))
rounded_targets_min = np.unique(np.round(t_train_neighbors_min, precision))

for file in tqdm(batch_files, desc="[INFO] Loading raw batches"):
    d = joblib.load(file)
    Xb, yb = d["X"], d["y"]
    
    # 2. Round the time column of the batch data to the same precision.
    rounded_batch_times = np.round(Xb[:, 0], precision)

    # 3. Use np.isin for an efficient, EXACT match between the rounded arrays.
    mask_max = np.isin(rounded_batch_times, rounded_targets_max)
    if np.any(mask_max):
        X_train_max_list.append(Xb[mask_max])
        y_train_max_list.append(yb[mask_max])
        
    mask_min = np.isin(rounded_batch_times, rounded_targets_min)
    if np.any(mask_min):
        X_train_min_list.append(Xb[mask_min])
        y_train_min_list.append(yb[mask_min])
# === END OF FIX ===

X_train_max = np.vstack(X_train_max_list)
y_train_max = np.vstack(y_train_max_list)
X_train_min = np.vstack(X_train_min_list)
y_train_min = np.vstack(y_train_min_list)

def process_data(X_train, y_train):
    X_full, y_full = X_train, y_train
    df_X = pd.DataFrame(X_full, columns=['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx'])
    df_y = pd.DataFrame(y_full)
    df_X["orig_index"] = np.arange(len(df_X))
    df_dedup = df_X.groupby(['t', 'sigma_idx', 'bundle_idx'], sort=False).tail(1).sort_values("orig_index")
    y_dedup = df_y.iloc[df_dedup.index].to_numpy()
    return df_dedup, y_dedup, df_y

df_dedup_max, y_max_dedup, df_y_max = process_data(X_train_max, y_train_max)
df_dedup_min, y_min_dedup, df_y_min = process_data(X_train_min, y_train_min)

# === Nominal Trajectory Propagation and Feature Engineering ===
nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
mu, F_nom, c_nom, m0_nom, g0_nom = nominal_data["mu"], nominal_data["F"], nominal_data["c"], nominal_data["m0"], nominal_data["g0"]
r_tr, v_tr, mass_tr, lam_tr, forwardTspan = nominal_data["r_tr"], nominal_data["v_tr"], nominal_data["mass_tr"], nominal_data["lam_tr"], nominal_data["backTspan"][::-1]

def get_initial_state(t_start, forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu):
    start_idx = np.argmin(np.abs(forwardTspan - t_start))
    r0, v0 = r_tr[start_idx], v_tr[start_idx]
    mee0 = rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten()
    m0_prop, lam0_prop = mass_tr[start_idx], lam_tr[start_idx]
    return np.concatenate([mee0, [m0_prop], lam0_prop])

def propagate_and_get_nominal_states(t_eval_points, x0, mu, F, c, m0, g0):
    sol = solve_ivp(odefunc.odefunc, (t_eval_points[0], t_eval_points[-1]), x0, args=(mu, F, c, m0, g0), t_eval=sorted(t_eval_points), dense_output=True, rtol=1e-6, atol=1e-9)
    r_prop, v_prop = mee2rv(*sol.y[:6, :], mu)
    return {t: (r, v, lam, m) for t, r, v, lam, m in zip(sol.t, r_prop, v_prop, sol.y[7:, :].T, sol.y[6, :].T)}

def engineer_corrective_features(df, df_y, nominal_state_lookup, mu, forwardTspan, target_unique_times):
    # --- CORRECTED: Removed magnitude columns from initialization ---
    for col in ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz', 'delta_m', 't_go']:
        df[col] = 0.0
    
    df_y_corr = pd.DataFrame(np.zeros_like(df_y), index=df_y.index)
    
    unique_times = target_unique_times
    t_final = forwardTspan[-1]
    
    for t in tqdm(unique_times, desc="[INFO] Engineering corrective features"):
        time_key = min(nominal_state_lookup.keys(), key=lambda k: abs(k-t))
        group_df = df[np.isclose(df['t'], t)]
        
        if group_df.empty:
            continue
            
        group_indices = group_df.index
        r_nom_target, v_nom_target, lam_nom_target, m_nom_target = nominal_state_lookup[time_key]
        
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, v_samples = mee2rv(*mees.T, mu)
        delta_r = r_nom_target - r_samples
        delta_v = v_nom_target - v_samples
        df.loc[group_indices, ['delta_rx', 'delta_ry', 'delta_rz']] = delta_r
        df.loc[group_indices, ['delta_vx', 'delta_vy', 'delta_vz']] = delta_v
        
        m_samples = group_df['mass'].values
        df.loc[group_indices, 'delta_m'] = m_nom_target - m_samples
        df.loc[group_indices, 't_go'] = t_final - t
        
        y_optimal_samples = df_y.loc[group_indices].values
        df_y_corr.loc[group_indices] = y_optimal_samples - lam_nom_target
        
    return df, df_y_corr.to_numpy()

# --- Process MAX data ---
print("\n[INFO] Processing MAX segment data...")
x0_max = get_initial_state(t_train_neighbors_max[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_max = propagate_and_get_nominal_states(t_train_neighbors_max, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_max, y_max_corr = engineer_corrective_features(df_dedup_max, df_y_max.iloc[df_dedup_max.index], nominal_states_max, mu, forwardTspan, t_train_neighbors_max)

# --- Process MIN data ---
print("\n[INFO] Processing MIN data...")
x0_min = get_initial_state(t_train_neighbors_min[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_min = propagate_and_get_nominal_states(t_train_neighbors_min, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_min, y_min_corr = engineer_corrective_features(df_dedup_min, df_y_min.iloc[df_dedup_min.index], nominal_states_min, mu, forwardTspan, t_train_neighbors_min)

# --- Save processed data to file ---
print(f"\n[INFO] Saving all processed data to {OUTPUT_FILENAME}...")
processed_data = {
    "max_data": {"df": df_dedup_max, "y_opt": y_max_dedup, "y_corr": y_max_corr},
    "min_data": {"df": df_dedup_min, "y_opt": y_min_dedup, "y_corr": y_min_corr},
}
joblib.dump(processed_data, OUTPUT_FILENAME)

print("\n[SUCCESS] Dataset generation complete.")