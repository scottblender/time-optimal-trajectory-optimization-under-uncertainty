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

mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
bundle_idx = 32

def final_position_deviation(r_true, r_pred):
    return np.linalg.norm(r_true[-1] - r_pred[-1])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def evaluate_single_sigma_against_sigma0(X_bundle, sigma_idx, y_pred_bundle, y_true_bundle, backTspan, stride, r_ref, v_ref):
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

    ctrl_pred = y_sigma_pred[0]
    ctrl_true = y_sigma_true[0]
    cosine_dir = cosine_similarity(ctrl_pred, ctrl_true)

    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        seg_times = np.linspace(t0, t1, 5)
        mee = X_sigma[i, 1:8]

        if i == 0:
            print(f"\n=== Time Stride {stride} | Sigma {sigma_idx} ===")
            print("Predicted Initial Control (lambda):")
            print(np.array2string(ctrl_pred, formatter={'float_kind': lambda x: f'{x: .5f}'}))
            print("Actual Initial Control (lambda):")
            print(np.array2string(ctrl_true, formatter={'float_kind': lambda x: f'{x: .5f}'}))
            print(f"Cosine Similarity (Pred vs. True): {cosine_dir:.6f}")

            r0_pred, _ = mee2rv.mee2rv(
                np.array([mee[0]]), np.array([mee[1]]), np.array([mee[2]]),
                np.array([mee[3]]), np.array([mee[4]]), np.array([mee[5]]), mu
            )
            print("Initial Predicted Position [x y z]:")
            print(np.array2string(r0_pred[0], formatter={'float_kind': lambda x: f'{x: .5f}'}))
            print("Initial Sigma-0 Reference Position [x y z]:")
            print(np.array2string(r_ref[0], formatter={'float_kind': lambda x: f'{x: .5f}'}))

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

    r_pred_interp = r_pred_all[:r_ref.shape[0]]
    v_pred_interp = v_pred_all[:v_ref.shape[0]]

    print("Final Predicted Position [x y z]:")
    print(np.array2string(r_pred_interp[-1], formatter={'float_kind': lambda x: f'{x: .5f}'}))
    print("Final Sigma-0 Reference Position [x y z]:")
    print(np.array2string(r_ref[-1], formatter={'float_kind': lambda x: f'{x: .5f}'}))

    mse_r = np.mean((r_ref - r_pred_interp) ** 2, axis=0)
    mse_v = np.mean((v_ref - v_pred_interp) ** 2, axis=0)
    final_dev = final_position_deviation(r_ref, r_pred_interp)
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
        "cosine similarity": cosine_dir,
        "final position deviation": final_dev
    }

def evaluate_per_sigma_point(X, y, trajectories, backTspan, stride):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    residuals = y_pred_train - y_train
    rf_std = np.std(residuals, axis=0)
    print("\n=== Empirical Std Deviation of Control Prediction Residuals ===")
    for i, std in enumerate(rf_std):
        print(f"Control {i+1}: {std:.6f}")

    test_mse_all = mean_squared_error(y_test, rf.predict(X_test))
    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = rf.predict(X_bundle[:, :-2])
    sigma_indices = np.unique(X_bundle[:, -1].astype(int))

    print(f"\n=== Propagation Time Span ===")
    print(f"Start Time: {X_bundle[:, 0].min():.5f}")
    print(f"End Time:   {X_bundle[:, 0].max():.5f}")
    print(f"Duration:   {(X_bundle[:, 0].max() - X_bundle[:, 0].min()):.5f} (nondimensional)")

    r_ref = trajectories[0][0][0][:, :3]
    v_ref = trajectories[0][0][0][:, 3:6]

    metrics_list = []
    for sigma_idx in sigma_indices:
        if sigma_idx == 0:
            continue
        metrics = evaluate_single_sigma_against_sigma0(
            X_bundle, sigma_idx, y_pred_bundle, y_bundle,
            backTspan, stride, r_ref, v_ref
        )
        metrics["time stride"] = stride
        metrics["model mse"] = test_mse_all
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)
    if not df_metrics.empty:
        mean_row = df_metrics.drop(columns=["sigma"]).mean(numeric_only=True).to_dict()
        mean_row["sigma"] = "ALL"
        mean_row["time stride"] = stride
        metrics_list.append(mean_row)

    return metrics_list

# === Main Loop ===
results = []
stride_files = sorted(glob.glob("data_bundle_32_stride_*.pkl"))

for file in stride_files:
    stride = int(file.split("_stride_")[1].split(".")[0])
    data = joblib.load(file)
    metrics = evaluate_per_sigma_point(data['X'], data['y'], data['trajectories'], data['X'][:, 0], stride)
    results.extend(metrics)

df = pd.DataFrame(results)
df = df[[ "time stride", "sigma", "model mse", "control mse", "cosine similarity",
          "x mse", "y mse", "z mse", "vx mse", "vy mse", "vz mse",
          "final position deviation" ]]
df.sort_values(by=["time stride", "sigma"], inplace=True)
df.to_csv("per_sigma_metrics_vs_stride.csv", index=False)
print("Saved: per_sigma_metrics_vs_stride.csv")