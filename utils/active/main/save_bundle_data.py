import joblib
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params

# === Compute Nominal and Bundle Trajectory Params ===
mu_s = 132712 * 10**6 * 1e9
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

num_bundles = 100
time_resolution_minutes = 20

r_tr, v_tr, mass_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, mass_bundles, backTspan = compute_bundle_trajectory_params.compute_bundle_trajectory_params(
    p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU, num_bundles, time_resolution_minutes
)

# === Save All Bundle Data ===
data = {
    "r_tr": r_tr,
    "v_tr": v_tr,
    "mass_tr": mass_tr,
    "S_bundles": S_bundles,
    "r_bundles": r_bundles,
    "v_bundles": v_bundles,
    "new_lam_bundles": new_lam_bundles,
    "mass_bundles": mass_bundles,
    "backTspan": backTspan,
    "mu": mu, "F": F, "c": c, "m0": m0, "g0": g0
}
joblib.dump(data, "bundle_data.pkl")
print("Saved nominal bundle data to bundle_data.pkl")

# === Export initial_bundles_all.csv ===
backTspan_reversed = backTspan[::-1]
combined_data = []

for bundle_index in range(num_bundles - 25):  # match original logic
    bundle_r = r_bundles[::-1, :, bundle_index]
    bundle_v = v_bundles[::-1, :, bundle_index]
    bundle_m = mass_bundles[::-1, bundle_index]
    bundle_lam = new_lam_bundles[::-1, :, bundle_index]

    for t_idx in range(len(backTspan_reversed)):
        row = [
            backTspan_reversed[t_idx],
            *bundle_r[t_idx], *bundle_v[t_idx],
            bundle_m[t_idx],
            *bundle_lam[t_idx],
            bundle_index
        ]
        combined_data.append(row)

df = pd.DataFrame(combined_data, columns=[
    "time", "x", "y", "z", "vx", "vy", "vz", "mass",
    "lam0", "lam1", "lam2", "lam3", "lam4", "lam5", "lam6",
    "bundle_index"
])
df.to_csv("initial_bundles_all.csv", index=False)
print("Saved initial bundle states to initial_bundles_all.csv")
