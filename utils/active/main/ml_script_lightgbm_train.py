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
P_pos = np.eye(3) * 0.01

# === Load segment times from width file ===
with open("stride_4000min/bundle_segment_widths.txt") as f:
    lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    time_vals = times_arr[:, 0]

    max_idx = int(np.argmax(times_arr[:, 1]))
    min_idx = int(np.argmin(times_arr[:, 1])) - 1
    if min_idx == len(times_arr) - 1:
        sorted_indices = np.argsort(times_arr[:, 1])
        min_idx = sorted_indices[1]

    t_max_neighbors = time_vals[max(0, max_idx - 1): max_idx + 2]
    t_min_neighbors = time_vals[max(0, min_idx - 1): min_idx + 2]

    print(f"[INFO] Max segment times: {t_max_neighbors}")
    print(f"[INFO] Min segment times: {t_min_neighbors}")

# === Init containers ===
X_all, y_all = [], []
X_max, y_max, X_min, y_min = [], [], [], []
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
            print(f"[WARN] Skipping covariance check: expected 15, got {len(sigmas)}")
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

    X_all.append(Xb)
    y_all.append(yb)

    for t_extract, X_list, y_list, label in [
        (t_max_neighbors, X_max, y_max, "max"),
        (t_min_neighbors, X_min, y_min, "min")
    ]:
        matched_any = False
        for t in t_extract:
            idx = np.round(Xb[:, 0], 6) == np.round(t, 6)
            if np.any(idx):
                X_list.append(Xb[idx])
                y_list.append(yb[idx])
                matched_any = True
        if not matched_any:
            print(f"[WARN] No data in {file} for {label} times {t_extract}")

# === Stack all data
X_full = np.vstack(X_all)
y_full = np.vstack(y_all)
X_max = np.vstack(X_max)
y_max = np.vstack(y_max)
X_min = np.vstack(X_min)
y_min = np.vstack(y_min)

# === Deduplicate σ-point rows at t₀ (keep last per group)
df_X = pd.DataFrame(X_full)
df_y = pd.DataFrame(y_full)
df_X.columns = ['t', 'p', 'f', 'g', 'h', 'L', 'mass', 'dummy1',
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                'bundle_idx', 'sigma_idx']
df_X["orig_index"] = np.arange(len(df_X))
group_cols = ['t', 'sigma_idx', 'bundle_idx']
df_dedup = df_X.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
X_full_cleaned = df_dedup.drop(columns=["orig_index"]).to_numpy()
y_full_cleaned = df_y.iloc[df_dedup["orig_index"].values].to_numpy()
print(f"[INFO] Deduplicated rows: from {len(X_full)} → {len(X_full_cleaned)}")

# === Remove appended sigma₀ rows (used only for eval, not training)
is_sigma0 = X_full_cleaned[:, -1] == 0
is_zero_cov = np.all(np.isclose(X_full_cleaned[:, 8:15], 0.0, atol=1e-12), axis=1)
is_appended_sigma0 = is_sigma0 & is_zero_cov
print(f"[FILTER] Removing {np.sum(is_appended_sigma0)} appended σ₀ rows from X_full...")
X_full_cleaned = X_full_cleaned[~is_appended_sigma0]
y_full_cleaned = y_full_cleaned[~is_appended_sigma0]

# === Time check
print("\n[CHECK] Unique times in X_full:")
unique_times_full = np.unique(X_full_cleaned[:, 0])
print(f"  Found {len(unique_times_full)} unique times:")
print(f"  [{unique_times_full[0]:.6f} ... {unique_times_full[-1]:.6f}]")

print("\n[CHECK] Unique times in X_max:")
print(f"  {np.unique(X_max[:, 0])}")

print("\n[CHECK] Unique times in X_min:")
print(f"  {np.unique(X_min[:, 0])}")

# === Bundle presence check
expected_total_bundles = 50
for name, X_seg in [("segment_max", X_max), ("segment_min", X_min)]:
    bundles = np.unique(X_seg[:, -2]).astype(int)
    missing = set(range(expected_total_bundles)) - set(bundles)
    if missing:
        raise ValueError(f"[ERROR] {name} is missing bundle indices: {sorted(missing)}")

print("[SUCCESS] Both segment_max and segment_min include all 50 bundles.")

# === Normalize features
print("[INFO] Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full_cleaned[:, :-2])

# === Train model
print("[INFO] Training LightGBM model...")
base_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    min_child_samples=20,
    force_row_wise=True,
    random_state=42,
    verbose=1
)
model = MultiOutputRegressor(base_model)
model.fit(X_scaled, y_full_cleaned)

# === Save everything
joblib.dump(model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(Wm, "Wm.pkl")
joblib.dump(Wc, "Wc.pkl")
joblib.dump({"X": X_max, "y": y_max}, "segment_max.pkl")
joblib.dump({"X": X_min, "y": y_min}, "segment_min.pkl")

print("[INFO] Model and segment data saved.")
print(f"[INFO] Elapsed time: {time.time() - start:.2f} sec")
