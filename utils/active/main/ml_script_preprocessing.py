import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

def clean_batch(Xb, yb):
    df_X = pd.DataFrame(Xb)
    df_y = pd.DataFrame(yb)
    df_X.columns = ['t', 'p', 'f', 'g', 'h', 'L', 'mass',
                    'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                    'bundle_idx', 'sigma_idx']
    df_X["orig_index"] = np.arange(len(df_X))
    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup = df_X.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
    X_clean = df_dedup.drop(columns=["orig_index"]).to_numpy()
    y_clean = df_y.iloc[df_dedup["orig_index"].values].to_numpy()

    is_sigma0 = X_clean[:, -1] == 0
    is_zero_cov = np.all(np.isclose(X_clean[:, 8:15], 0.0, atol=1e-12), axis=1)
    is_appended_sigma0 = is_sigma0 & is_zero_cov
    X_clean = X_clean[~is_appended_sigma0]
    y_clean = y_clean[~is_appended_sigma0]
    return X_clean, y_clean

def stream_rows(file):
    d = joblib.load(file, mmap_mode='r')
    Xb, yb, meta = d["X"], d["y"], d["meta"]
    for i in range(len(Xb)):
        bundle = int(meta[i, 0])
        sigma = int(meta[i, 1])
        key = f"bundle_{bundle:02d}_sigma_{sigma:02d}"
        with open(os.path.join(TMP_DIR, f"{key}_X.npy"), "ab") as fx, \
             open(os.path.join(TMP_DIR, f"{key}_y.npy"), "ab") as fy:
            np.save(fx, Xb[i])
            np.save(fy, yb[i])

def finalize_sequence(key):
    X_path = os.path.join(TMP_DIR, f"{key}_X.npy")
    y_path = os.path.join(TMP_DIR, f"{key}_y.npy")
    X_stream, y_stream = [], []
    with open(X_path, "rb") as fx, open(y_path, "rb") as fy:
        while True:
            try:
                X_stream.append(np.load(fx))
                y_stream.append(np.load(fy))
            except ValueError:
                break
    X_arr = np.stack(X_stream)
    y_arr = np.stack(y_stream)
    sort_idx = np.argsort(X_arr[:, 0])
    X_sorted = X_arr[sort_idx]
    y_sorted = y_arr[sort_idx]
    out_path = os.path.join(SEQ_DIR, f"{key}.pkl")
    joblib.dump({"X": X_sorted, "y": y_sorted}, out_path, compress=0)
    tqdm.write(f"[WRITE] {key} → {out_path}")

if __name__ == "__main__":
    RAW_DIR = "baseline_stride_1/batch_*"
    OUT_DIR = "baseline_stride_1_cleaned"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[PASS 1] Fitting StandardScaler on all cleaned data...")
    scaler_data = []
    batch_files = sorted(glob.glob(RAW_DIR))

    for batch_dir in tqdm(batch_files, desc="[SCALER] Collecting stats"):
        file = os.path.join(batch_dir, "data.pkl")
        if not os.path.exists(file):
            print(f"[SKIP] Missing {file}")
            continue
        d = joblib.load(file)
        Xb, yb = d["X"], d["y"]
        Xb_clean, _ = clean_batch(Xb, yb)
        scaler_data.append(Xb_clean[:, :-2])  # exclude bundle_idx and sigma_idx

    X_all = np.vstack(scaler_data)
    scaler = StandardScaler()
    scaler.fit(X_all)
    joblib.dump(scaler, "scaler_tcn.pkl")
    print("[DONE] Scaler fitted and saved to scaler_tcn.pkl")

    print("[PASS 2] Cleaning + scaling + saving batches...")
    for batch_dir in tqdm(batch_files, desc="[PROCESS] Saving cleaned batches"):
        file = os.path.join(batch_dir, "data.pkl")
        if not os.path.exists(file):
            print(f"[SKIP] Missing {file}")
            continue
        d = joblib.load(file)
        Xb, yb = d["X"], d["y"]
        Xb_clean, yb_clean = clean_batch(Xb, yb)
        Xb_scaled = scaler.transform(Xb_clean[:, :-2])
        bundle_sigma = Xb_clean[:, -2:]

        batch_id = os.path.basename(os.path.dirname(file))
        out_dir = os.path.join(OUT_DIR, batch_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.pkl")
        joblib.dump({"X": Xb_scaled, "y": yb_clean, "meta": bundle_sigma}, out_path, compress=0)
        print(f"[SAVE] Batch {batch_id} → {out_path}")

    print("[COMPLETE] All batches processed and saved.")

