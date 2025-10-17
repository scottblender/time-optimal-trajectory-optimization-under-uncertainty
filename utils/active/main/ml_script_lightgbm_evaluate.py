# ml_script_lightgbm_train.py

import os
import glob
import joblib
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# New imports for trajectory propagation and scoring
from scipy.integrate import solve_ivp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

# === Constants ===
mu = 27.8996

# === Load segment times from width file ===
with open("stride_1440min/bundle_segment_widths.txt") as f:
    lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    time_vals = times_arr[:, 0]

    max_idx = int(np.argmax(times_arr[:, 1]))
    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    n = 50
    t_train_neighbors_max = time_vals[max(0, max_idx - n): max_idx + n + 1]
    t_train_neighbors_min = time_vals[max(0, min_idx - n): min_idx + n + 1]

# === Init containers ===
X_train_max, y_train_max = [], []
X_train_min, y_train_min = [], []

start = time.time()
batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))

for file in tqdm(batch_files, desc="[INFO] Loading batches"):
    d = joblib.load(file)
    Xb, yb = d["X"], d["y"]

    for t in t_train_neighbors_max:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_train_max.append(Xb[idx]); y_train_max.append(yb[idx])
    for t in t_train_neighbors_min:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx): X_train_min.append(Xb[idx]); y_train_min.append(yb[idx])

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
# === SAVE MEAN COVARIANCE DIAGONALS ===========================================
# ==============================================================================
cov_cols = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
mean_cov_diag_max = df_dedup_max[cov_cols].mean().to_numpy()
np.savetxt("cov_diag_max.txt", mean_cov_diag_max)
print(f"\n[INFO] Saved mean covariance diagonal for MAX window to cov_diag_max.txt")

mean_cov_diag_min = df_dedup_min[cov_cols].mean().to_numpy()
np.savetxt("cov_diag_min.txt", mean_cov_diag_min)
print(f"[INFO] Saved mean covariance diagonal for MIN window to cov_diag_min.txt")

# ==============================================================================
# === SAVE PROCESSED SEGMENT DATA FOR EVALUATION ============
# ==============================================================================
x_cols_to_save = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']

# Select only the desired columns before converting to numpy and saving
data_to_save_max = {"X": df_dedup_max[x_cols_to_save].to_numpy(), "y": y_max_dedup}
data_to_save_min = {"X": df_dedup_min[x_cols_to_save].to_numpy(), "y": y_min_dedup}

# Save the data using joblib
joblib.dump(data_to_save_max, "segment_max.pkl")
joblib.dump(data_to_save_min, "segment_min.pkl")

print("\n[INFO] Saved processed MAX segment data to segment_max.pkl")
print("[INFO] Saved processed MIN segment data to segment_min.pkl")

# ==============================================================================
# === NOMINAL TRAJECTORY PROPAGATION AND FEATURE ENGINEERING ===================
# ==============================================================================
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

def propagate_and_get_nominal_states(t_eval_points, x0, mu, F, c, m0, g0):
    t_span = (t_eval_points[0], t_eval_points[-1])
    sol = solve_ivp(
        odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    r_propagated, v_propagated = mee2rv(*sol.y[:6, :], mu)
    mass_propagated = sol.y[6, :].T
    lam_propagated = sol.y[7:, :].T
    return {t: (r, v, lam, m) for t, r, v, lam, m in zip(sol.t, r_propagated, v_propagated, lam_propagated, mass_propagated)}

### FEATURE ENGINEERING FOR CORRECTIVE MODEL ONLY ###
def engineer_corrective_features(df, df_y, nominal_state_lookup, mu):
    # This function now ONLY creates features and targets for the corrective model.
    df['delta_rx'], df['delta_ry'], df['delta_rz'] = 0.0, 0.0, 0.0
    df['delta_vx'], df['delta_vy'], df['delta_vz'] = 0.0, 0.0, 0.0
    # REMOVED: df['delta_m'] and other new features
    
    df_y_corr = pd.DataFrame(np.zeros_like(df_y), index=df_y.index)
    unique_times = df['t'].unique()
    
    for t in tqdm(unique_times, desc="[INFO] Engineering features for corrective model"):
        time_key = min(nominal_state_lookup.keys(), key=lambda k: abs(k-t))
        group_df = df[df['t'] == t]
        group_indices = group_df.index
        
        r_nom_target, v_nom_target, lam_nom_target, m_nom_target = nominal_state_lookup[time_key]
        
        # Cartesian deviations
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, v_samples = mee2rv(*mees.T, mu)
        delta_r = r_nom_target - r_samples
        delta_v = v_nom_target - v_samples
        df.loc[group_indices, ['delta_rx', 'delta_ry', 'delta_rz']] = delta_r
        df.loc[group_indices, ['delta_vx', 'delta_vy', 'delta_vz']] = delta_v
        
        # REMOVED: Mass deviation calculation and other new feature calculations
        
        # Target (costate correction)
        y_optimal_samples = df_y.loc[group_indices].values
        df_y_corr.loc[group_indices] = y_optimal_samples - lam_nom_target
        
    return df, df_y_corr.to_numpy()

# --- Propagate and Engineer Features for MAX data ---
print("\n[INFO] Propagating and engineering features for MAX segment...")
t_train_max_unique = np.unique(np.round(df_dedup_max['t'], 6))
x0_max = get_initial_state(t_train_max_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_max = propagate_and_get_nominal_states(t_train_max_unique, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_max, y_max_corr = engineer_corrective_features(df_dedup_max, df_y_max.iloc[df_dedup_max.index], nominal_states_max, mu)

# --- Propagate and Engineer Features for MIN data ---
print("\n[INFO] Propagating and engineering features for MIN segment...")
t_train_min_unique = np.unique(np.round(df_dedup_min['t'], 6))
x0_min = get_initial_state(t_train_min_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_min = propagate_and_get_nominal_states(t_train_min_unique, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_min, y_min_corr = engineer_corrective_features(df_dedup_min, df_y_min.iloc[df_dedup_min.index], nominal_states_min, mu)


# ==============================================================================
# === BUILD AND TRAIN MODELS ===================================================
# ==============================================================================
base_model_params = {'n_estimators': 1500, 'learning_rate': 0.005, 'max_depth': 15, 'min_child_samples': 10, 'force_row_wise': True, 'random_state': 42, 'verbose': -1}

# --- Define Feature Sets ---
features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
# REVERTED: Corrective features list now only includes delta_r and delta_v
features_corrective = features_optimal + ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz']

# --- Process and Train MAX models ---
print("\n" + "="*20 + " TRAINING MAX MODELS " + "="*20)
is_sigma0_max = df_dedup_max['sigma_idx'] == 0
is_zero_cov_max = np.all(np.isclose(df_dedup_max[cov_cols].values, 0.0), axis=1)
mask_max = ~(is_sigma0_max & is_zero_cov_max)

# Data for Optimal Model (MAX)
X_max_optimal = df_dedup_max.loc[mask_max, features_optimal].to_numpy()
y_max_optimal = y_max_dedup[mask_max]
print(f"[INFO] Training Optimal MAX model with shape: {X_max_optimal.shape}")
model_max_optimal = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_max_optimal.fit(X_max_optimal, y_max_optimal)
joblib.dump(model_max_optimal, "trained_model_max_optimal.pkl")

# Data for Corrective Model (MAX)
X_max_corrective = df_dedup_max.loc[mask_max, features_corrective].to_numpy()
y_max_corrective = y_max_corr[mask_max]
print(f"[INFO] Training Corrective MAX model with shape: {X_max_corrective.shape}")
model_max_corrective = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_max_corrective.fit(X_max_corrective, y_max_corrective)
joblib.dump(model_max_corrective, "trained_model_max_corrective.pkl")

# --- Process and Train MIN models ---
print("\n" + "="*20 + " TRAINING MIN MODELS " + "="*20)
is_sigma0_min = df_dedup_min['sigma_idx'] == 0
is_zero_cov_min = np.all(np.isclose(df_dedup_min[cov_cols].values, 0.0), axis=1)
mask_min = ~(is_sigma0_min & is_zero_cov_min)

# Data for Optimal Model (MIN)
X_min_optimal = df_dedup_min.loc[mask_min, features_optimal].to_numpy()
y_min_optimal = y_min_dedup[mask_min]
print(f"[INFO] Training Optimal MIN model with shape: {X_min_optimal.shape}")
model_min_optimal = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_min_optimal.fit(X_min_optimal, y_min_optimal)
joblib.dump(model_min_optimal, "trained_model_min_optimal.pkl")

# Data for Corrective Model (MIN)
X_min_corrective = df_dedup_min.loc[mask_min, features_corrective].to_numpy()
y_min_corrective = y_min_corr[mask_min]
print(f"[INFO] Training Corrective MIN model with shape: {X_min_corrective.shape}")
model_min_corrective = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_min_corrective.fit(X_min_corrective, y_min_corrective)
joblib.dump(model_min_corrective, "trained_model_min_corrective.pkl")

print("\n[SUCCESS] All comparison models have been trained and saved.")