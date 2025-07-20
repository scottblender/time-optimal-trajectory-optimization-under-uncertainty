import os
import joblib
import numpy as np

def analyze_covariance_ranges(base_dir="baseline_stride_1"):
    covariances = []

    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname, "data.pkl")
        if os.path.isfile(fpath):
            print(f"[INFO] Loading {fpath}")
            data = joblib.load(fpath)
            X = data["X"]

            # Filter only rows with nonzero covariance features (skip appended sigma₀)
            valid_rows = ~np.all(np.abs(X[:, 8:15]) < 1e-12, axis=1)
            covariances.append(X[valid_rows, 8:15])  # shape: (N, 7)

    if not covariances:
        print("[ERROR] No valid data found.")
        return

    all_cov = np.vstack(covariances)  # shape: (Total rows, 7)
    min_vals = np.min(all_cov, axis=0)
    max_vals = np.max(all_cov, axis=0)

    labels = ['p', 'f', 'g', 'h', 'L', 'mass', 'extra']
    print("\n=== Covariance Diagonal Range in MEE + mass ===")
    for i, (min_v, max_v) in enumerate(zip(min_vals, max_vals)):
        print(f"{labels[i]:>5}: min = {min_v:.6e}, max = {max_v:.6e}")

if __name__ == "__main__":
    analyze_covariance_ranges()
