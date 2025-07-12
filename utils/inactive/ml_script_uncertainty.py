import os
import glob
import joblib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
bundle_idx = 32

def final_position_deviation(r_true, r_pred):
    return np.linalg.norm(r_true[-1] - r_pred[-1])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def propagate_sigma0_truth(X_bundle, y_bundle):
    mask = X_bundle[:, -1] == 0
    X_sigma = X_bundle[mask]
    y_sigma = y_bundle[mask]
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma = y_sigma[sort_idx]

    r_true_all, v_true_all = [], []
    lam_used = y_sigma[0]

    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        seg_times = np.linspace(t0, t1, 5)
        state = X_sigma[i, 1:8]
        S = np.concatenate([state, lam_used])
        Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                       [t0, t1], S, t_eval=seg_times)
        r_xyz, v_xyz = mee2rv.mee2rv(*Sf.y.T[:, :6].T, mu)
        r_true_all.append(r_xyz)
        v_true_all.append(v_xyz)
        lam_used = Sf.y[7:, -1]

    return np.vstack(r_true_all), np.vstack(v_true_all)

def evaluate_single_sigma_against_sigma0(X_bundle, sigma_idx, y_pred_bundle, y_true_bundle, r_ref, v_ref):
    mask = X_bundle[:, -1] == sigma_idx
    X_sigma = X_bundle[mask]
    y_sigma_pred = y_pred_bundle[mask]
    y_sigma_true = y_true_bundle[mask]
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]
    y_sigma_true = y_sigma_true[sort_idx]

    r_pred_all, v_pred_all = [], []
    control_errors = []
    lam_used_pred = y_sigma_pred[0]
    lam_used_true = y_sigma_true[0]

    print(f"\n=== Sigma {sigma_idx} ===")
    print(f"Cosine similarity: {cosine_similarity(lam_used_pred, lam_used_true):.6f}")

    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        seg_times = np.linspace(t0, t1, 5)
        state = X_sigma[i, 1:8]
        S_pred = np.concatenate([state, lam_used_pred])
        Sf_pred = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                            [t0, t1], S_pred, t_eval=seg_times)
        r_xyz, v_xyz = mee2rv.mee2rv(*Sf_pred.y.T[:, :6].T, mu)
        r_pred_all.append(r_xyz)
        v_pred_all.append(v_xyz)
        control_errors.append(np.mean((lam_used_pred - lam_used_true) ** 2))
        lam_used_pred = Sf_pred.y[7:, -1]

    r_pred_all = np.vstack(r_pred_all)
    v_pred_all = np.vstack(v_pred_all)
    r_ref = r_ref[:len(r_pred_all)]
    v_ref = v_ref[:len(v_pred_all)]

    ctrl_mse_over_time = np.mean((y_sigma_pred - y_sigma_true) ** 2)
    mse_r = np.mean((r_ref - r_pred_all) ** 2, axis=0)
    mse_v = np.mean((v_ref - v_pred_all) ** 2, axis=0)
    final_dev = final_position_deviation(r_ref, r_pred_all)

    return {
        "sigma": sigma_idx,
        "x mse": mse_r[0],
        "y mse": mse_r[1],
        "z mse": mse_r[2],
        "vx mse": mse_v[0],
        "vy mse": mse_v[1],
        "vz mse": mse_v[2],
        "control mse (initial)": np.mean(control_errors),
        "control mse (full profile)": ctrl_mse_over_time,
        "cosine similarity": cosine_similarity(y_sigma_pred[0], y_sigma_true[0]),
        "final position deviation": final_dev
    }

def evaluate_uncertainty_distribution(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    residuals = y_pred_train - y_train
    rf_std = np.std(residuals, axis=0)
    print(f"\n--- {label.upper()} ---")
    print("Control prediction residual std dev:")
    for i, std in enumerate(rf_std):
        print(f"  λ{i+1}: {std:.6f}")

    test_mse_all = mean_squared_error(y_test, rf.predict(X_test))

    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = rf.predict(X_bundle[:, :-2])
    sigma_indices = np.unique(X_bundle[:, -1].astype(int))
    r_ref, v_ref = propagate_sigma0_truth(X_bundle, y_bundle)

    results = []
    for sigma_idx in sigma_indices:
        if sigma_idx == 0:
            continue
        metrics = evaluate_single_sigma_against_sigma0(X_bundle, sigma_idx, y_pred_bundle, y_bundle, r_ref, v_ref)
        metrics["distribution"] = label
        metrics["model mse"] = test_mse_all
        results.append(metrics)

    if results:
        mean_row = pd.DataFrame(results).drop(columns=["sigma"]).mean(numeric_only=True).to_dict()
        mean_row["sigma"] = "ALL"
        mean_row["distribution"] = label
        results.append(mean_row)

    return results

# === Main Evaluation Loop ===
all_metrics = []
uncertainty_files = sorted(glob.glob("data_bundle_32_uncertainty_*.pkl"))

for file in uncertainty_files:
    label = file.split("_uncertainty_")[1].replace(".pkl", "")
    data = joblib.load(file)
    metrics = evaluate_uncertainty_distribution(data['X'], data['y'], label)
    all_metrics.extend(metrics)

df = pd.DataFrame(all_metrics)
df = df[[ 
    "distribution", "sigma", "model mse",
    "control mse (initial)", "control mse (full profile)",
    "cosine similarity",
    "x mse", "y mse", "z mse", "vx mse", "vy mse", "vz mse",
    "final position deviation"
]]
df.sort_values(by=["distribution", "sigma"], inplace=True)
df.to_csv("per_sigma_metrics_vs_uncertainty.csv", index=False)
print("Saved: per_sigma_metrics_vs_uncertainty.csv")
