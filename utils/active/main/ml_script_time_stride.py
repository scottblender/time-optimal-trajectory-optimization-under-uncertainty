import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from scipy.integrate import solve_ivp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
bundle_idx = 32

def final_position_deviation(r_true, r_pred):
    return np.linalg.norm(r_true[-1] - r_pred[-1])

def evaluate_single_sigma(X_bundle, sigma_idx, y_pred_bundle, y_true_bundle, trajectories, backTspan, stride):
    mask = X_bundle[:, -1] == sigma_idx
    X_sigma = X_bundle[mask]
    y_sigma_pred = y_pred_bundle[mask]
    y_sigma_true = y_true_bundle[mask]
    
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]
    y_sigma_true = y_sigma_true[sort_idx]

    r_actual = trajectories[0][0][sigma_idx][:, :3]
    v_actual = trajectories[0][0][sigma_idx][:, 3:6]

    r_pred_all, v_pred_all = [], []
    control_errors = []

    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        seg_times = np.linspace(t0, t1, 5)
        mee = X_sigma[i, 1:8]
        ctrl_pred = y_sigma_pred[i]
        ctrl_true = y_sigma_true[i]
        S = np.concatenate([mee, ctrl_pred])
        Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                       [t0, t1], S, t_eval=seg_times)

        r_xyz, v_xyz = mee2rv.mee2rv(Sf.y.T[:, 0], Sf.y.T[:, 1], Sf.y.T[:, 2],
                                     Sf.y.T[:, 3], Sf.y.T[:, 4], Sf.y.T[:, 5], mu)
        r_pred_all.append(r_xyz)
        v_pred_all.append(v_xyz)

        propagated_controls = Sf.y.T[:, 7:]
        true_controls = np.tile(ctrl_true, (propagated_controls.shape[0], 1))
        control_error = np.mean((propagated_controls - true_controls) ** 2)
        control_errors.append(control_error)

    r_pred_all = np.vstack(r_pred_all)
    v_pred_all = np.vstack(v_pred_all)
    r_pred_interp = r_pred_all[:r_actual.shape[0]]
    v_pred_interp = v_pred_all[:v_actual.shape[0]]

    mse_r = np.mean((r_actual - r_pred_interp) ** 2, axis=0)
    mse_v = np.mean((v_actual - v_pred_interp) ** 2, axis=0)
    final_dev = final_position_deviation(r_actual, r_pred_interp)
    control_mse = np.mean(control_errors)

    return {
        "sigma": sigma_idx,
        "x mse": mse_r[0],
        "y mse": mse_r[1],
        "z mse": mse_r[2],
        "vx mse": mse_v[0],
        "vy mse": mse_v[1],
        "vz mse": mse_v[2],
        "control mse": control_mse,
        "final position deviation": final_dev
    }

def evaluate_per_sigma_point(X, y, trajectories, backTspan, stride):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    test_mse_all = mean_squared_error(y_test, rf.predict(X_test))
    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = rf.predict(X_bundle[:, :-2])
    sigma_indices = np.unique(X_bundle[:, -1].astype(int))

    metrics_list = []
    for sigma_idx in sigma_indices:
        metrics = evaluate_single_sigma(X_bundle, sigma_idx, y_pred_bundle, y_bundle, trajectories, backTspan, stride)
        metrics["time stride"] = stride
        metrics["model mse"] = test_mse_all
        metrics_list.append(metrics)

    return metrics_list

# === Main Loop ===
results = []
stride_files = sorted(glob.glob("data_bundle_32_stride_*.pkl"))

for file in stride_files:
    stride = int(file.split("_stride_")[1].split(".")[0])
    data = joblib.load(file)
    metrics = evaluate_per_sigma_point(data['X'], data['y'], data['trajectories'], data['X'][:, 0], stride)
    results.extend(metrics)

# === Save results ===
df = pd.DataFrame(results)

# Ensure all desired columns are present
df = df[[
    "time stride", "sigma", "model mse", "control mse",
    "x mse", "y mse", "z mse",
    "vx mse", "vy mse", "vz mse",
    "final position deviation"
]]

df.sort_values(by=["time stride", "sigma"], inplace=True)
df.to_csv("per_sigma_metrics_vs_stride.csv", index=False)
print("Saved: per_sigma_metrics_vs_stride.csv")
