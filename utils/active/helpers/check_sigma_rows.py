import numpy as np
import joblib
from mee2rv import mee2rv
from numpy.linalg import inv

# === Load segment data and weights ===
data = joblib.load("segment_max.pkl")  # or "segment_min.pkl"
X = data["X"]
Wm = joblib.load("Wm.pkl")
Wc = joblib.load("Wc.pkl")

# === CONFIG ===
mu = 27.8996
t0 = np.unique(X[:, 0])[0]  # segment start time
bundle_idx = int(np.unique(X[:, -2])[0])

# === Get all sigma points at t0 ===
X_b_all = X[(X[:, 0] == t0) & (X[:, -2] == bundle_idx)]
sigma_indices = np.unique(X_b_all[:, -1])
assert len(sigma_indices) == 15, f"[ERROR] Expected 15 unique sigma indices, got {len(sigma_indices)}"

# Select only the first row per sigma_idx
X_b = np.array([X_b_all[X_b_all[:, -1] == s][1] for s in sigma_indices])

# === Convert all sigma points to Cartesian positions ===
r_all = []
for row in X_b:
    mee = row[1:8]
    r, _ = mee2rv(*[np.array([val]) for val in mee[:6]], mu)
    r_all.append(r.flatten())
r_all = np.array(r_all)

# === Compute weighted mean and covariance ===
mean_r = np.sum(Wm[:, None] * r_all, axis=0)
deviations = r_all - mean_r[None, :]
cov_r = np.einsum("i,ij,ik->jk", Wc, deviations, deviations)
inv_cov_r = inv(cov_r + 1e-10 * np.eye(3))

print(f"[INFO] Weighted position covariance:\n{cov_r}")
print(f"[INFO] Weighted mean r: {mean_r}")

# === Find σ₀ rows (multiple rows with sigma_idx == 0) ===
candidates = X_b[X_b[:, -1] == 0]
print(f"\n[DEBUG] Found {len(candidates)} σ₀ candidates at t = {t0:.6f}")

best_idx = -1
best_score = float("inf")

for i, row in enumerate(candidates):
    mee = row[1:8]
    r, _ = mee2rv(*[np.array([val]) for val in mee[:6]], mu)
    r = r.flatten()
    maha = np.sqrt((r - mean_r) @ inv_cov_r @ (r - mean_r))
    print(f"  Row {i}: Mahalanobis distance to mean = {maha:.6f}")
    if maha < best_score:
        best_score = maha
        best_idx = i

print(f"\n[SELECTED] Row {best_idx} is the best σ₀ match.")
print("Selected row:", candidates[best_idx])
