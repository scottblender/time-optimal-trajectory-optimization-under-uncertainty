import os
import glob
import joblib
import numpy as np
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

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

    for t_extract, X_list, y_list, label in [
        (t_max_neighbors, X_max, y_max, "max"),
        (t_min_neighbors, X_min, y_min, "min")
    ]:
        matched_any = False
        for t in t_extract:
            idx = np.isclose(Xb[:, 0], t)
            if np.any(idx):
                X_list.append(Xb[idx])
                y_list.append(yb[idx])
                matched_any = True
        if not matched_any:
            print(f"[WARN] No data in {file} for {label} times {t_extract}")

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

# === Verify all 50 bundles are present before training ===
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

# === Train Random Forest model ===
print("[INFO] Training Random Forest model...")
base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(base_model)
model.fit(X_full[:, :-2], y_full)

# === Save model and data ===
joblib.dump(model, "trained_model.pkl")
joblib.dump(Wm, "Wm.pkl")
joblib.dump(Wc, "Wc.pkl")
joblib.dump({"X": X_max, "y": y_max}, "segment_max.pkl")
joblib.dump({"X": X_min, "y": y_min}, "segment_min.pkl")

print("[INFO] Model and segment data saved.")
print(f"[INFO] Elapsed time: {time.time() - start:.2f} sec")
