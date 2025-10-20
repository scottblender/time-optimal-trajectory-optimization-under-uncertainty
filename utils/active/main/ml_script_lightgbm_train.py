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
from scipy.integrate import solve_ivp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
import odefunc

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

batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))

for file in tqdm(batch_files, desc="[INFO] Loading batches"):
    d = joblib.load(file)
    Xb, yb = d["X"], d["y"]

    for t in t_train_neighbors_max:
        idx = np.isclose(Xb[:, 0], t)
        if np.any(idx): X_train_max.append(Xb[idx]); y_train_max.append(yb[idx])
    for t in t_train_neighbors_min:
        idx = np.isclose(Xb[:, 0], t)
        if np.any(idx): X_train_min.append(Xb[idx]); y_train_min.append(yb[idx])

def process_data(X_train, y_train):
    """Helper to process raw loaded data into deduplicated dataframes."""
    X_full = np.vstack(X_train)
    y_full = np.vstack(y_train)
    df_X = pd.DataFrame(X_full)
    df_y = pd.DataFrame(y_full)
    df_X.columns = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
    df_X["orig_index"] = np.arange(len(df_X))
    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup = df_X.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
    y_dedup = df_y.iloc[df_dedup.index].to_numpy()
    return df_dedup, y_dedup, df_y

# --- PROCESSING FOR MAX & MIN MODELS ---
print("\n" + "="*20 + " PROCESSING MAX DATA " + "="*20)
df_dedup_max, y_max_dedup, df_y_max = process_data(X_train_max, y_train_max)
print("\n" + "="*20 + " PROCESSING MIN DATA " + "="*20)
df_dedup_min, y_min_dedup, df_y_min = process_data(X_train_min, y_train_min)

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
        odefunc.odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    r_propagated, v_propagated = mee2rv(*sol.y[:6, :], mu)
    lam_propagated = sol.y[7:, :].T
    mass_propagated = sol.y[6, :].T
    return {t: (r, v, lam, m) for t, r, v, lam, m in zip(sol.t, r_propagated, v_propagated, lam_propagated, mass_propagated)}

def engineer_corrective_features(df, df_y, nominal_state_lookup, mu, forwardTspan):
    """Creates features and targets for the corrective model with the enhanced feature set."""
    # Add columns for all new features
    for col in ['delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
                'delta_m', 't_go', 'pos_dev_mag', 'vel_dev_mag']:
        df[col] = 0.0
    
    df_y_corr = pd.DataFrame(np.zeros_like(df_y), index=df_y.index)
    unique_times = df['t'].unique()
    t_final = forwardTspan[-1]
    
    for t in tqdm(unique_times, desc="[INFO] Engineering corrective features"):
        time_key = min(nominal_state_lookup.keys(), key=lambda k: abs(k-t))
        group_df = df[df['t'] == t]
        group_indices = group_df.index
        
        r_nom_target, v_nom_target, lam_nom_target, m_nom_target = nominal_state_lookup[time_key]
        
        # Calculate position and velocity deviations
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, v_samples = mee2rv(*mees.T, mu)
        delta_r = r_nom_target - r_samples
        delta_v = v_nom_target - v_samples
        df.loc[group_indices, ['delta_rx', 'delta_ry', 'delta_rz']] = delta_r
        df.loc[group_indices, ['delta_vx', 'delta_vy', 'delta_vz']] = delta_v
        
        # --- NEW FEATURE CALCULATIONS ---
        # Mass deviation
        m_samples = group_df['mass'].values
        df.loc[group_indices, 'delta_m'] = m_nom_target - m_samples
        # Time-to-go
        df.loc[group_indices, 't_go'] = t_final - t
        # Deviation magnitudes
        df.loc[group_indices, 'pos_dev_mag'] = np.linalg.norm(delta_r, axis=1)
        df.loc[group_indices, 'vel_dev_mag'] = np.linalg.norm(delta_v, axis=1)
        
        # Calculate the target variable: delta_lambda = lambda_optimal - lambda_nominal
        y_optimal_samples = df_y.loc[group_indices].values
        df_y_corr.loc[group_indices] = y_optimal_samples - lam_nom_target
        
    return df, df_y_corr.to_numpy()

# --- Propagate and Engineer Features for MAX data ---
print("\n[INFO] Propagating and engineering features for MAX segment...")
t_train_max_unique = np.unique(np.round(df_dedup_max['t'], 6))
x0_max = get_initial_state(t_train_max_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_max = propagate_and_get_nominal_states(t_train_max_unique, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_max, y_max_corr = engineer_corrective_features(df_dedup_max, df_y_max.iloc[df_dedup_max.index], nominal_states_max, mu, forwardTspan)

# --- Propagate and Engineer Features for MIN data ---
print("\n[INFO] Propagating and engineering features for MIN segment...")
t_train_min_unique = np.unique(np.round(df_dedup_min['t'], 6))
x0_min = get_initial_state(t_train_min_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_states_min = propagate_and_get_nominal_states(t_train_min_unique, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_min, y_min_corr = engineer_corrective_features(df_dedup_min, df_y_min.iloc[df_dedup_min.index], nominal_states_min, mu, forwardTspan)

# ==============================================================================
# === BUILD AND TRAIN MODELS ===================================================
# ==============================================================================
base_model_params = {
    'n_estimators': 2000,          
    'learning_rate': 0.005,
    'max_depth': 10,              
    'min_child_samples': 25,       
    'reg_lambda': 0.1,             
    'force_row_wise': True,
    'random_state': 42,
    'verbose': -1
}

# --- Define Feature Sets ---
features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
features_corrective = features_optimal + [
    'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
    'delta_m', 't_go', 'pos_dev_mag', 'vel_dev_mag'
]

def train_and_save_models(df_dedup, y_dedup, y_corr, model_name_prefix):
    """Helper to handle the final filtering, training, and saving for a model set."""
    is_sigma0 = df_dedup['sigma_idx'] == 0
    is_zero_cov = np.all(np.isclose(df_dedup[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']].values, 0.0), axis=1)
    mask = ~(is_sigma0 & is_zero_cov)

    # Train and save Optimal Model
    X_optimal = df_dedup.loc[mask, features_optimal].to_numpy()
    y_optimal = y_dedup[mask]
    print(f"[INFO] Training Optimal {model_name_prefix.upper()} model with shape: {X_optimal.shape}")
    model_optimal = MultiOutputRegressor(LGBMRegressor(**base_model_params))
    model_optimal.fit(X_optimal, y_optimal)
    joblib.dump(model_optimal, f"trained_model_{model_name_prefix}_optimal.pkl")

    # Train and save Corrective Model
    X_corrective = df_dedup.loc[mask, features_corrective].to_numpy()
    y_corrective = y_corr[mask]
    print(f"[INFO] Training Corrective {model_name_prefix.upper()} model with shape: {X_corrective.shape}")
    model_corrective = MultiOutputRegressor(LGBMRegressor(**base_model_params))
    model_corrective.fit(X_corrective, y_corrective)
    joblib.dump(model_corrective, f"trained_model_{model_name_prefix}_corrective.pkl")

# --- Process and Train MAX & MIN models ---
print("\n" + "="*20 + " TRAINING MAX MODELS " + "="*20)
train_and_save_models(df_dedup_max, y_max_dedup, y_max_corr, "max")
print("\n" + "="*20 + " TRAINING MIN MODELS " + "="*20)
train_and_save_models(df_dedup_min, y_min_dedup, y_min_corr, "min")

print("\n[SUCCESS] All comparison models have been trained and saved.")