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

# === Initialize containers ===
X_all, y_all = [], []
X_max, y_max, X_min, y_min = [], [], [], []
Wm, Wc = None, None

start = time.time()
batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))

# === Load and filter data with progress bar ===
for file in tqdm(batch_files, desc="[INFO] Loading batches"):
    d = joblib.load(file)
    Xb, yb = d["X"], d["y"]
    if Wm is None: Wm, Wc = d["Wm"], d["Wc"]
    X_all.append(Xb)
    y_all.append(yb)

    # Optimized segment filtering using vectorized OR mask
    segment_pairs = {
        "max": list(zip(t_max_neighbors[:-1], t_max_neighbors[1:])),
        "min": list(zip(t_min_neighbors[:-1], t_min_neighbors[1:]))
    }

    for label, pairs in segment_pairs.items():
        bounds = np.array(pairs)
        mask = np.zeros(len(Xb), dtype=bool)
        for t0, t1 in bounds:
            mask |= (Xb[:, 0] >= t0 - 1e-6) & (Xb[:, 0] <= t1 + 1e-6)
        if np.any(mask):
            if label == "max":
                X_max.append(Xb[mask])
                y_max.append(yb[mask])
            else:
                X_min.append(Xb[mask])
                y_min.append(yb[mask])
        else:
            print(f"[WARN] No data in {file} for {label} segments")

# === Check for empty segment data ===
if not X_max or not y_max:
    raise ValueError("[ERROR] No data found for max segment times.")
if not X_min or not y_min:
    raise ValueError("[ERROR] No data found for min segment times.")

# === Stack all data ===
X_full = np.vstack(X_all)
y_full = np.vstack(y_all)
X_max = np.vstack(X_max)
y_max = np.vstack(y_max)
X_min = np.vstack(X_min)
y_min = np.vstack(y_min)

# === Deduplicate σ-point rows at t₀ (keep last per (t, σ_idx, bundle_idx))
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
unique_times_max = np.unique(X_max[:, 0])
print(f"  Found {len(unique_times_max)} unique times:")
print(f"  {unique_times_max}")

print("\n[CHECK] Unique times in X_min:")
unique_times_min = np.unique(X_min[:, 0])
print(f"  Found {len(unique_times_min)} unique times:")
print(f"  {unique_times_min}")

# === Bundle presence check
expected_total_bundles = 50
bundles_max = np.unique(X_max[:, -2]).astype(int)
missing_max = set(range(expected_total_bundles)) - set(bundles_max)
if missing_max:
    raise ValueError(f"[ERROR] segment_max is missing bundle indices: {sorted(missing_max)}")

bundles_min = np.unique(X_min[:, -2]).astype(int)
missing_min = set(range(expected_total_bundles)) - set(bundles_min)
if missing_min:
    raise ValueError(f"[ERROR] segment_min is missing bundle indices: {sorted(missing_min)}")

print("[SUCCESS] Both segment_max and segment_min include all 50 bundles.")

# === Normalize features
print("[INFO] Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full_cleaned[:, :-2])  # Exclude bundle_idx, sigma_idx

# === Train model
print("[INFO] Training LightGBM model...")
base_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    min_data_in_leaf=20,
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
