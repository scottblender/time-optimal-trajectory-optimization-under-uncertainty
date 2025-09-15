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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv

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
    
    # --- MODIFICATION START ---
    # Define separate training windows for max and min segments
    t_train_neighbors_max = time_vals[max(0, max_idx - n): max_idx + n + 1]
    t_train_neighbors_min = time_vals[max(0, min_idx - n): min_idx + n + 1]

    print(f"[INFO] Max segment times (for eval): {t_max_neighbors_eval}")
    print(f"[INFO] Min segment times (for eval): {t_min_neighbors_eval}")
    print(f"[INFO] Training times for MAX (t_max +/- {n}): {t_train_neighbors_max}")
    print(f"[INFO] Training times for MIN (t_min +/- {n}): {t_train_neighbors_min}")
    # --- MODIFICATION END ---


# === Init containers ===
X_eval_max, y_eval_max, X_eval_min, y_eval_min = [], [], [], []
# --- MODIFICATION START ---
# Create separate training lists for max and min data
X_train_max, y_train_max = [], []
X_train_min, y_train_min = [], []
# --- MODIFICATION END ---
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
            # print(f"[WARN] Skipping covariance check: expected 15, got {len(sigmas)}")
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
    
    # --- MODIFICATION START ---
    # Populate the training data for the MAX model
    for t in t_train_neighbors_max:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_train_max.append(Xb[idx])
            y_train_max.append(yb[idx])
            
    # Populate the training data for the MIN model
    for t in t_train_neighbors_min:
        idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
        if np.any(idx):
            X_train_min.append(Xb[idx])
            y_train_min.append(yb[idx])
    # --- MODIFICATION END ---


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
X_max_cleaned = df_dedup_max.drop(columns=["orig_index"]).to_numpy()
y_max_cleaned = df_y_max.iloc[df_dedup_max["orig_index"].values].to_numpy()
print(f"[INFO] Deduplicated MAX rows: from {len(X_full_max)} -> {len(X_max_cleaned)}")

# === Remove appended sigma₀ rows (used only for eval, not training)
is_sigma0_max = X_max_cleaned[:, -1] == 0
is_zero_cov_max = np.all(np.isclose(X_max_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_max = is_sigma0_max & is_zero_cov_max
print(f"[FILTER] Removing {np.sum(is_appended_sigma0_max)} appended sigma₀ rows from X_full_max...")
X_max_cleaned = X_max_cleaned[~is_appended_sigma0_max]
y_max_cleaned = y_max_cleaned[~is_appended_sigma0_max]


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
X_min_cleaned = df_dedup_min.drop(columns=["orig_index"]).to_numpy()
y_min_cleaned = df_y_min.iloc[df_dedup_min["orig_index"].values].to_numpy()
print(f"[INFO] Deduplicated MIN rows: from {len(X_full_min)} -> {len(X_min_cleaned)}")

# === Remove appended sigma₀ rows (used only for eval, not training)
is_sigma0_min = X_min_cleaned[:, -1] == 0
is_zero_cov_min = np.all(np.isclose(X_min_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0_min = is_sigma0_min & is_zero_cov_min
print(f"[FILTER] Removing {np.sum(is_appended_sigma0_min)} appended sigma₀ rows from X_full_min...")
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
joblib.dump({"X": X_min_eval, "y": y_min_eval}, "segment_min.pkl")

print("\n[INFO] Models and segment data saved.")
print(f"[INFO] Elapsed time: {time.time() - start:.2f} sec")