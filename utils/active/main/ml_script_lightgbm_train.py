# ml_script_lightgbm_train.py

import os
import glob
import joblib
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# New imports for trajectory propagation and scoring
from scipy.integrate import solve_ivp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
import odefunc

# === Constants ===
mu = 27.8996
DU_km = 696340.0  # Sun radius in km
P_pos_km2  = np.eye(3) * 0.01
P_pos = P_pos_km2 / DU_km **2

# === Load segment times from width file ===
with open("stride_1440min/bundle_segment_widths.txt") as f:
    lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    time_vals = times_arr[:, 0]

    max_idx = int(np.argmax(times_arr[:, 1]))
    min_idx = int(np.argmin(times_arr[:, 1])) - 1
    if min_idx == len(times_arr) - 1:
        sorted_indices = np.argsort(times_arr[:, 1])
        min_idx = sorted_indices[1]
    n = 50
    t_max_neighbors_eval = time_vals[max(0, max_idx - 1): max_idx + 2]
    t_min_neighbors_eval = time_vals[max(0, min_idx - 1): min_idx + 2]
    
    # Define separate training windows for max and min segments
    t_train_neighbors_max = time_vals[max(0, max_idx - n): max_idx + n + 1]
    t_train_neighbors_min = time_vals[max(0, min_idx - n): min_idx + n + 1]

# === Init containers ===
X_eval_max, y_eval_max, X_eval_min, y_eval_min = [], [], [], []
X_train_max, y_train_max = [], []
X_train_min, y_train_min = [], []
Wm, Wc = None, None

start = time.time()
batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))

for file in tqdm(batch_files, desc="[INFO] Loading batches"):
    d = joblib.load(file)
    Xb, yb = d["X"], d["y"]
    if Wm is None: Wm, Wc = d["Wm"], d["Wc"]

    for t in t_max_neighbors_eval:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_eval_max.append(Xb[idx]); y_eval_max.append(yb[idx])
    for t in t_min_neighbors_eval:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_eval_min.append(Xb[idx]); y_eval_min.append(yb[idx])
    for t in t_train_neighbors_max:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_train_max.append(Xb[idx]); y_train_max.append(yb[idx])
    for t in t_train_neighbors_min:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_train_min.append(Xb[idx]); y_train_min.append(yb[idx])

# === Stack the evaluation data
X_max_eval = np.vstack(X_eval_max)
y_max_eval = np.vstack(y_eval_max)
X_min_eval = np.vstack(X_eval_min)
y_min_eval = np.vstack(y_eval_min)

# --- PROCESSING FOR MAX MODEL ---
print("\n" + "="*20 + " PROCESSING MAX DATA " + "="*20)
X_full_max = np.vstack(X_train_max)
y_full_max = np.vstack(y_train_max)
df_X_max = pd.DataFrame(X_full_max)
df_y_max = pd.DataFrame(y_full_max)
df_X_max.columns = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
df_X_max["orig_index"] = np.arange(len(df_X_max))
group_cols = ['t', 'sigma_idx', 'bundle_idx']
df_dedup_max = df_X_max.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
y_max_dedup = df_y_max.iloc[df_dedup_max.index].to_numpy()

# --- PROCESSING FOR MIN MODEL ---
print("\n" + "="*20 + " PROCESSING MIN DATA " + "="*20)
X_full_min = np.vstack(X_train_min)
y_full_min = np.vstack(y_train_min)
df_X_min = pd.DataFrame(X_full_min)
df_y_min = pd.DataFrame(y_full_min)
df_X_min.columns = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
df_X_min["orig_index"] = np.arange(len(df_X_min))
group_cols = ['t', 'sigma_idx', 'bundle_idx']
df_dedup_min = df_X_min.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
y_min_dedup = df_y_min.iloc[df_dedup_min.index].to_numpy()

# ==============================================================================
# === NEW: NOMINAL TRAJECTORY PROPAGATION AND SCORING ==========================
# ==============================================================================
print("\n[INFO] Loading nominal trajectory data for scoring...")
nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
r_tr, v_tr, mass_tr, lam_tr = nominal_data["r_tr"], nominal_data["v_tr"], nominal_data["mass_tr"], nominal_data["lam_tr"]
forwardTspan = nominal_data["backTspan"][::-1]
F_nom, c_nom, m0_nom, g0_nom = nominal_data["F"], nominal_data["c"], nominal_data["m0"], nominal_data["g0"]

def get_initial_state(t_start, forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu):
    start_idx = np.argmin(np.abs(forwardTspan - t_start))
    r0, v0 = r_tr[start_idx], v_tr[start_idx]
    mee0 = rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten()
    m0_prop, lam0_prop = mass_tr[start_idx], lam_tr[start_idx]
    return np.concatenate([mee0, [m0_prop], lam0_prop])

def propagate_and_get_nominal_state(t_eval_points, x0, mu, F, c, m0, g0):
    t_span = (t_eval_points[0], t_eval_points[-1])
    sol = solve_ivp(
        odefunc.odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    propagated_mees = sol.y[:6, :].T
    r_propagated, v_propagated = mee2rv(*propagated_mees.T, mu)
    return {t: (r, v) for t, r, v in zip(sol.t, r_propagated, v_propagated)}

def add_weighted_error_feature(df, nominal_state_lookup, mu):
    # --- Tunable Weights ---
    # alpha prioritizes position error, beta prioritizes velocity error.
    alpha = 1.0
    beta = 0.1
    
    df['weighted_error'] = 0.0
    unique_times = df['t'].unique()
    grouped = df.groupby('t')
    
    for t in tqdm(unique_times, desc="[INFO] Calculating weighted error feature"):
        time_key = min(nominal_state_lookup.keys(), key=lambda k: abs(k-t))
        group_df = grouped.get_group(t)
        r_nom, v_nom = nominal_state_lookup[time_key]
        
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, v_samples = mee2rv(*mees.T, mu)
        
        delta_r = r_samples - r_nom
        delta_v = v_samples - v_nom
        
        dr_norm_sq = np.sum(delta_r**2, axis=1)
        dv_norm_sq = np.sum(delta_v**2, axis=1)
        
        weighted_error = (alpha * dr_norm_sq) + (beta * dv_norm_sq)
        df.loc[group_df.index, 'weighted_error'] = weighted_error
        
    return df

# --- Propagate and Score MAX data ---
print("\n[INFO] Propagating nominal trajectory for MAX segment training window...")
t_train_max_unique = np.unique(np.round(df_dedup_max['t'], 6))
x0_max = get_initial_state(t_train_max_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_state_max = propagate_and_get_nominal_state(t_train_max_unique, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_max = add_weighted_error_feature(df_dedup_max, nominal_state_max, mu)

# --- Propagate and Score MIN data ---
print("\n[INFO] Propagating nominal trajectory for MIN segment training window...")
t_train_min_unique = np.unique(np.round(df_dedup_min['t'], 6))
x0_min = get_initial_state(t_train_min_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_state_min = propagate_and_get_nominal_state(t_train_min_unique, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_min = add_weighted_error_feature(df_dedup_min, nominal_state_min, mu)

# ==============================================================================
# === REBUILD CLEANED DATASETS =================================================
# ==============================================================================
feature_cols = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass',
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                'weighted_error']
bookkeeping_cols = ['bundle_idx', 'sigma_idx']

# --- Rebuild MAX data ---
X_max_cleaned = df_dedup_max[feature_cols + bookkeeping_cols].to_numpy()
y_max_cleaned = y_max_dedup
is_sigma0_max = X_max_cleaned[:, -2] == 0
is_zero_cov_max = np.all(np.isclose(X_max_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_max = is_sigma0_max & is_zero_cov_max
X_max_final = X_max_cleaned[~is_appended_sigma0_max]
y_max_final = y_max_cleaned[~is_appended_sigma0_max]
print(f"[FILTER] Final MAX training data shape: {X_max_final.shape}")

# --- Rebuild MIN data ---
X_min_cleaned = df_dedup_min[feature_cols + bookkeeping_cols].to_numpy()
y_min_cleaned = y_min_dedup
is_sigma0_min = X_min_cleaned[:, -2] == 0
is_zero_cov_min = np.all(np.isclose(X_min_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_min = is_sigma0_min & is_zero_cov_min
X_min_final = X_min_cleaned[~is_appended_sigma0_min]
y_min_final = y_min_cleaned[~is_appended_sigma0_min]
print(f"[FILTER] Final MIN training data shape: {X_min_final.shape}")

# === Train models ===
base_model_params = {'n_estimators': 1500, 'learning_rate': 0.005, 'max_depth': 15, 'min_child_samples': 10, 'force_row_wise': True, 'random_state': 42, 'verbose': 1}

# --- Train MAX model ---
print("\n[INFO] Training LightGBM model for MAX segment...")
model_max = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_max.fit(X_max_final[:,:-2], y_max_final)

# --- Train MIN model ---
print("\n[INFO] Training LightGBM model for MIN segment...")
model_min = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_min.fit(X_min_final[:,:-2], y_min_final)

# === Save everything ===
joblib.dump(model_max, "trained_model_max.pkl")
joblib.dump(model_min, "trained_model_min.pkl")
joblib.dump({"X": X_max_eval, "y": y_max_eval}, "segment_max.pkl")
joblib.dump({"X": X_min_eval, "y": y_min_eval}, "segment_min.pkl")
print("\n[INFO] Models and segment data saved.")