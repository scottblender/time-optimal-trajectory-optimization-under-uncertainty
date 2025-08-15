import os
import glob
import joblib
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

CLEANED_DIR = "baseline_stride_1_cleaned"
SEQ_PATH = "bundle_sigma_sequences.h5"

def extract_rows(file):
    result = []
    d = joblib.load(file, mmap_mode="r")
    Xb, yb, meta = d["X"], d["y"], d["meta"]

    for i in range(len(Xb)):
        bundle = int(meta[i, 0])
        sigma = int(meta[i, 1])
        key = f"bundle_{bundle:02d}_sigma_{sigma:02d}"
        result.append((key, Xb[i], yb[i]))
    return result

def main():
    os.makedirs(os.path.dirname(SEQ_PATH) or ".", exist_ok=True)

    batch_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "batch_*/data.pkl")))
    print(f"[INFO] Found {len(batch_files)} cleaned batch files.")

    print("[STEP 1] Extracting all rows (in parallel)...")
    all_rows = []
    with Pool(min(cpu_count(), 8)) as pool:
        for result in tqdm(pool.imap_unordered(extract_rows, batch_files), total=len(batch_files)):
            all_rows.extend(result)

    print(f"[INFO] Total rows collected: {len(all_rows):,}")

    print(f"[STEP 2] Writing sorted sequences to {SEQ_PATH} using HDF5...")
    from collections import defaultdict
    grouped = defaultdict(lambda: {"X": [], "y": []})

    for key, x, y in all_rows:
        grouped[key]["X"].append(x)
        grouped[key]["y"].append(y)

    with h5py.File(SEQ_PATH, "w") as h5f:
        for key, data in tqdm(grouped.items(), desc="[SAVE]"):
            X_arr = np.stack(data["X"])
            y_arr = np.stack(data["y"])
            sort_idx = np.argsort(X_arr[:, 0])
            X_sorted = X_arr[sort_idx]
            y_sorted = y_arr[sort_idx]

            g = h5f.create_group(key)
            g.create_dataset("X", data=X_sorted, compression="gzip", compression_opts=3)
            g.create_dataset("y", data=y_sorted, compression="gzip", compression_opts=3)

    print("[DONE] HDF5 sequence file written.")

if __name__ == "__main__":
    freeze_support()
    main()
