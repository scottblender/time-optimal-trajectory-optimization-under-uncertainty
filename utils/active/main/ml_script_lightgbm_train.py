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

    # print(f"[INFO] Max segment times (for eval): {t_max_neighbors_eval}")
    # print(f"[INFO] Min segment times (for eval): {t_min_neighbors_eval}")
    # print(f"[INFO] Training times for MAX (t_max +/- {n}): {t_train_neighbors_max}")
    # print(f"[INFO] Training times for MIN (t_min +/- {n}): {t_train_neighbors_min}")


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

    # === Covariance check for each bundle
    bundle_ids = np.unique(Xb[:, -2]).astype(int)
    for b_idx in bundle_ids:
        Xb_bundle = Xb[Xb[:, -2] == b_idx]
        t0 = np.min(Xb_bundle[:, 0])
        X_t0 = Xb_bundle[np.isclose(Xb_bundle[:, 0], t0)]
        sigmas = []
        for s_idx in range(15):
            matches = X_t0[X_t0[:, -1] == s_idx]
            if len(matches) == 0: continue
            sigmas.append(matches[-1])
        if len(sigmas) != 15:
            continue
        r_all = []
        for row in sigmas:
            mee = row[1:8]
            r, _ = mee2rv(*[np.array([val]) for val in mee[:6]], mu)
            r_all.append(r.flatten())
        r_all = np.array(r_all)
        mean_r = np.sum(Wm[:, None] * r_all, axis=0)
        cov_r = np.einsum("i,ij,ik->jk", Wc, r_all - mean_r, r_all - mean_r)
        frob = np.linalg.norm(cov_r - P_pos)
        print(f"[VERIFY] bundle {int(b_idx)} @ t={t0:.3f}: ||cov - P_pos||_F = {frob:.3e}")
        if frob < 1e-5:
            print("         ✅ Position covariance matches expected P_init.")
        else:
            print("         ⚠️  Covariance deviates from P_init.")

    for t in t_max_neighbors_eval:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_eval_max.append(Xb[idx])
            y_eval_max.append(yb[idx])

    for t in t_min_neighbors_eval:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_eval_min.append(Xb[idx])
            y_eval_min.append(yb[idx])
    
    for t in t_train_neighbors_max:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_train_max.append(Xb[idx])
            y_train_max.append(yb[idx])
            
    for t in t_train_neighbors_min:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_train_min.append(Xb[idx])
            y_train_min.append(yb[idx])


# === Stack the evaluation data
X_max_eval = np.vstack(X_eval_max)
y_max_eval = np.vstack(y_eval_max)
X_min_eval = np.vstack(X_eval_min)
y_min_eval = np.vstack(y_eval_min)


# --- PROCESSING FOR MAX MODEL ---
print("\n" + "="*20 + " PROCESSING MAX DATA " + "="*20)
X_full_max = np.vstack(X_train_max)
y_full_max = np.vstack(y_train_max)

# === Deduplicate σ-point rows at t₀ (keep last per group)
df_X_max = pd.DataFrame(X_full_max)
df_y_max = pd.DataFrame(y_full_max)
df_X_max.columns = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass',
                    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                    'bundle_idx', 'sigma_idx']
df_X_max["orig_index"] = np.arange(len(df_X_max))
group_cols = ['t', 'sigma_idx', 'bundle_idx']
df_dedup_max = df_X_max.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")

# --- PROCESSING FOR MIN MODEL ---
print("\n" + "="*20 + " PROCESSING MIN DATA " + "="*20)
X_full_min = np.vstack(X_train_min)
y_full_min = np.vstack(y_train_min)

# === Deduplicate σ-point rows at t₀ (keep last per group)
df_X_min = pd.DataFrame(X_full_min)
df_y_min = pd.DataFrame(y_full_min)
df_X_min.columns = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass',
                    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                    'bundle_idx', 'sigma_idx']
df_X_min["orig_index"] = np.arange(len(df_X_min))
group_cols = ['t', 'sigma_idx', 'bundle_idx']
df_dedup_min = df_X_min.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")


# ==============================================================================
# === NEW: NOMINAL TRAJECTORY PROPAGATION AND SCORING ==========================
# ==============================================================================

print("\n[INFO] Loading nominal trajectory data for scoring...")
nominal_data_file = "stride_1440min/bundle_data_1440min.pkl"
nominal_data = joblib.load(nominal_data_file)
r_tr = nominal_data["r_tr"]
v_tr = nominal_data["v_tr"]
mass_tr = nominal_data["mass_tr"]
lam_tr = nominal_data["lam_tr"]
backTspan = nominal_data["backTspan"]
forwardTspan = backTspan[::-1]
F_nom = nominal_data["F"]
c_nom = nominal_data["c"]
m0_nom = nominal_data["m0"]
g0_nom = nominal_data["g0"]

def get_initial_state(t_start, forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu):
    """Finds the full 14-element state vector for the nominal trajectory at a given time."""
    start_idx = np.argmin(np.abs(forwardTspan - t_start))
    r0, v0 = r_tr[start_idx], v_tr[start_idx]
    mee0 = rv2mee(r0, v0, mu)
    m0_prop = mass_tr[start_idx]
    lam0_prop = lam_tr[start_idx]
    x0 = np.concatenate([mee0, [m0_prop], lam0_prop])
    return x0

def propagate_and_get_positions(t_eval_points, x0, mu, F, c, m0, g0):
    """Propagates the nominal trajectory and returns positions at evaluation times."""
    t_span = (t_eval_points[0], t_eval_points[-1])
    sol = solve_ivp(
        odefunc.odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    propagated_mees = sol.y[:6, :].T
    r_propagated, _ = mee2rv(
        propagated_mees[:, 0], propagated_mees[:, 1], propagated_mees[:, 2],
        propagated_mees[:, 3], propagated_mees[:, 4], propagated_mees[:, 5], mu
    )
    return {t: r for t, r in zip(sol.t, r_propagated)}

def add_deviation_features(df, nominal_pos_lookup, mu):
    """
    Calculates distance from nominal and the deviation vector components,
    and adds them as features to the DataFrame.
    """
    # Initialize new columns
    df['score'] = 0.0
    df['delta_r_x'] = 0.0
    df['delta_r_y'] = 0.0
    df['delta_r_z'] = 0.0
    
    unique_times = df['t'].unique()
    grouped = df.groupby('t')
    
    for t in tqdm(unique_times, desc="[INFO] Calculating deviation features"):
        time_key = min(nominal_pos_lookup.keys(), key=lambda k: abs(k-t))
        if abs(time_key - t) > 1e-3:
            print(f"[WARN] Time {t:.4f} not in nominal solution (closest is {time_key:.4f}). Skipping.")
            continue
        
        group_df = grouped.get_group(t)
        r_nom = nominal_pos_lookup[time_key]
        
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, _ = mee2rv(mees[:, 0], mees[:, 1], mees[:, 2],
                              mees[:, 3], mees[:, 4], mees[:, 5], mu)
        
        # Calculate the deviation vector and scalar distance
        delta_r = r_samples - r_nom
        distances = np.linalg.norm(delta_r, axis=1)
        
        # Normalize distance for the 'score' feature
        max_dist = np.max(distances)
        scores = distances / max_dist if max_dist > 1e-9 else np.zeros_like(distances)

        # Update the main DataFrame using the group's index
        df.loc[group_df.index, 'score'] = scores
        df.loc[group_df.index, 'delta_r_x'] = delta_r[:, 0]
        df.loc[group_df.index, 'delta_r_y'] = delta_r[:, 1]
        df.loc[group_df.index, 'delta_r_z'] = delta_r[:, 2]
        
    return df

# --- Propagate and Score MAX data ---
print("\n[INFO] Propagating nominal trajectory for MAX segment training window...")
t_train_max_unique = np.unique(np.round(df_dedup_max['t'], 6))
x0_max = get_initial_state(t_train_max_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_pos_max = propagate_and_get_positions(t_train_max_unique, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_max = add_deviation_features(df_dedup_max, nominal_pos_max, mu)

# --- Propagate and Score MIN data ---
print("\n[INFO] Propagating nominal trajectory for MIN segment training window...")
t_train_min_unique = np.unique(np.round(df_dedup_min['t'], 6))
x0_min = get_initial_state(t_train_min_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
nominal_pos_min = propagate_and_get_positions(t_train_min_unique, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
df_dedup_min = add_deviation_features(df_dedup_min, nominal_pos_min, mu)


# ==============================================================================
# === REBUILD CLEANED DATASETS WITH NEW SCORE FEATURE ==========================
# ==============================================================================

# --- Rebuild MAX data ---
feature_cols = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass',
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                'score', 'delta_r_x', 'delta_r_y', 'delta_r_z']
bookkeeping_cols = ['bundle_idx', 'sigma_idx']
X_max_cleaned = df_dedup_max[feature_cols + bookkeeping_cols].to_numpy()
y_max_cleaned = df_y_max.iloc[df_dedup_max.index].to_numpy()
print(f"[INFO] Rebuilt MAX data with score. Shape: {X_max_cleaned.shape}")

# === Remove appended sigma₀ rows (used only for eval, not training)
is_sigma0_max = X_max_cleaned[:, -2] == 0 # bundle_idx is second to last
is_zero_cov_max = np.all(np.isclose(X_max_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_max = is_sigma0_max & is_zero_cov_max
print(f"[FILTER] Removing {np.sum(is_appended_sigma0_max)} appended sigma₀ rows from X_max_cleaned...")
X_max_cleaned = X_max_cleaned[~is_appended_sigma0_max]
y_max_cleaned = y_max_cleaned[~is_appended_sigma0_max]


# --- Rebuild MIN data ---
X_min_cleaned = df_dedup_min[feature_cols + bookkeeping_cols].to_numpy()
y_min_cleaned = df_y_min.iloc[df_dedup_min.index].to_numpy()
print(f"[INFO] Rebuilt MIN data with score. Shape: {X_min_cleaned.shape}")

# === Remove appended sigma₀ rows (used only for eval, not training)
is_sigma0_min = X_min_cleaned[:, -2] == 0 # bundle_idx is second to last
is_zero_cov_min = np.all(np.isclose(X_min_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_min = is_sigma0_min & is_zero_cov_min
print(f"[FILTER] Removing {np.sum(is_appended_sigma0_min)} appended sigma₀ rows from X_min_cleaned...")
X_min_cleaned = X_min_cleaned[~is_appended_sigma0_min]
y_min_cleaned = y_min_cleaned[~is_appended_sigma0_min]


# === Calculate and save min/max covariance values for replanning script
# Note: Using the combined cleaned data to get a global min/max for the covariance
combined_cleaned_X = np.vstack([X_max_cleaned, X_min_cleaned])
non_zero_cov_rows = combined_cleaned_X[~np.all(np.isclose(combined_cleaned_X[:, 8:15], 0.0, atol=1e-12), axis=1)]
if non_zero_cov_rows.size > 0:
    diag_mins = np.min(non_zero_cov_rows[:, 8:15], axis=0)
    diag_maxs = np.max(non_zero_cov_rows[:, 8:15], axis=0)
    np.save("diag_mins.npy", diag_mins)
    np.save("diag_maxs.npy", diag_maxs)
    print(f"\n[INFO] Saved diag_mins.npy: {diag_mins}")
    print(f"[INFO] Saved diag_maxs.npy: {diag_maxs}")
else:
    print("\n[WARN] No non-zero covariance rows found to calculate min/max diagonals.")

# === Bundle presence check
expected_total_bundles = 50
for name, X_seg in [("segment_max", X_max_eval), ("segment_min", X_min_eval)]:
    bundles = np.unique(X_seg[:, -2]).astype(int)
    missing = set(range(expected_total_bundles)) - set(bundles)
    if missing:
        raise ValueError(f"[ERROR] {name} is missing bundle indices: {sorted(missing)}")

print("[SUCCESS] Both segment_max and segment_min include all 50 bundles.")

# === Calculate and save feature bounds for clamping ===
print("\n[INFO] Calculating and saving feature bounds for replanning...")
combined_training_X = np.vstack([X_max_cleaned, X_min_cleaned])

# Select only the feature columns the model is trained on
training_features = combined_training_X[:, :-2] 

feature_mins = np.min(training_features, axis=0)
feature_maxs = np.max(training_features, axis=0)

np.save("feature_mins.npy", feature_mins)
np.save("feature_maxs.npy", feature_maxs)
print("[INFO] Saved feature_mins.npy and feature_maxs.npy.")

# === Train models ===
base_model_params = {
    'n_estimators': 1500,
    'learning_rate': 0.005,
    'max_depth': 15,
    'min_child_samples': 10,
    'force_row_wise': True,
    'random_state': 42,
    'verbose': 1
}

# --- Train MAX model ---
print("\n[INFO] Training LightGBM model for MAX segment...")
model_max = MultiOutputRegressor(LGBMRegressor(**base_model_params))
# The slice [:, :-2] now correctly selects all features including the new 'score'
model_max.fit(X_max_cleaned[:,:-2], y_max_cleaned)

# --- Train MIN model ---
print("\n[INFO] Training LightGBM model for MIN segment...")
model_min = MultiOutputRegressor(LGBMRegressor(**base_model_params))
model_min.fit(X_min_cleaned[:,:-2], y_min_cleaned)


# === Save everything ===
joblib.dump(model_max, "trained_model_max.pkl")
joblib.dump(model_min, "trained_model_min.pkl")
joblib.dump(Wm, "Wm.pkl")
joblib.dump(Wc, "Wc.pkl")
joblib.dump({"X": X_max_eval, "y": y_max_eval}, "segment_max.pkl")
joblib.dump({"X": X_min_eval, "y": y_eval_min}, "segment_min.pkl")

print("\n[INFO] Models and segment data saved.")
print(f"[INFO] Elapsed time: {time.time() - start:.2f} sec")