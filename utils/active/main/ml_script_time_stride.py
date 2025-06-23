import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import odefunc

# Constants
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
bundle_idx = 32

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def monte_carlo_propagation(state_init, control_profile, segments, num_samples=10, std=1e-5):
    r_hist, v_hist = [], []
    for _ in range(num_samples):
        state = state_init.copy()
        state[:6] += np.random.normal(0, std, size=6)
        r_traj, v_traj = [], []

        for i, (t0, t1) in enumerate(segments):
            lam = control_profile[i]
            S = np.concatenate([state, lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S, t_eval=t_eval)
            r, v = mee2rv.mee2rv(*Sf.y[:6], mu)
            r_traj.append(r)
            v_traj.append(v)
            state = Sf.y[:7, -1]
        r_hist.append(np.vstack(r_traj))
        v_hist.append(np.vstack(v_traj))
    return np.array(r_hist), np.array(v_hist)

def evaluate_model_with_sigma0_alignment(X, y, stride, Wm, Wc):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X[:, :-2])

    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = rf.predict(X_bundle[:, :-2])
    time_vals = np.unique(X_bundle[:, 0])
    segments = list(zip(time_vals[:-1], time_vals[1:]))

    r_pred_stack, control_profile, init_state = [], [], None
    v_pred_all = []

    sigma_indices = np.unique(X_bundle[:, -1]).astype(int)
    sigma_indices.sort()

    for sigma_idx in sigma_indices:
        r_sigma, v_sigma = [], []
        for i, (t0, t1) in enumerate(segments):
            row = X_bundle[(X_bundle[:, 0] == t0) & (X_bundle[:, -1] == sigma_idx)][0]
            lam = y_pred_bundle[(X_bundle[:, 0] == t0) & (X_bundle[:, -1] == sigma_idx)][0]
            S = np.concatenate([row[1:8], lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S, t_eval=t_eval)
            r, v = mee2rv.mee2rv(*Sf.y[:6], mu)
            r_sigma.append(r)
            v_sigma.append(v)
            if i == 0 and sigma_idx == 0:
                init_state = row[1:8]
            if sigma_idx == 0:
                control_profile.append(lam)
        r_pred_stack.append(np.vstack(r_sigma))
        if sigma_idx == 0:
            v_pred_all = np.vstack(v_sigma)

    r_pred_stack = np.array(r_pred_stack)  # (15, T, 3)
    r_pred_all = r_pred_stack[0]

    mean_pred = np.sum(Wm[:, None, None] * r_pred_stack, axis=0)
    deviations = r_pred_stack - mean_pred[None, :, :]
    P_full = np.einsum("i,ijk,ijl->jkl", Wc, deviations, deviations)
    P_diag = np.array([np.diag(np.diag(P)) for P in P_full])
    inv_cov = inv(P_diag[-1])

    r_mc, v_mc = monte_carlo_propagation(init_state, control_profile, segments)
    r_mc_final = r_mc[:, -1, :]
    r_mc_mean_final = np.mean(r_mc_final, axis=0)

    final_time = time_vals[-1]
    sigma0_rows = X_bundle[(X_bundle[:, 0] == final_time) & (X_bundle[:, -1] == 0)]
    appended = next((row for row in sigma0_rows if np.allclose(row[8:15], 0)), None)
    if appended is None:
        raise ValueError(f"Appended sigma 0 not found at t = {final_time:.6f}")

    r_ref, v_ref = mee2rv.mee2rv(*appended[1:7].reshape(6, 1), mu)
    r_ref = r_ref.flatten()
    v_ref = v_ref.flatten()

    mahalanobis_pred = mahalanobis(r_pred_all[-1], r_ref, inv_cov)
    mahalanobis_mc_mean = mahalanobis(r_mc_mean_final, r_ref, inv_cov)

    lam_pred_0 = control_profile[0]
    lam_true_0 = y_bundle[(X_bundle[:, 0] == time_vals[0]) & (X_bundle[:, -1] == 0)][0]
    cos_sim = cosine_similarity(lam_pred_0, lam_true_0)

    r_mc_mean_full = np.mean(r_mc, axis=0)
    mc_vs_pred_mse = np.mean((r_pred_all - r_mc_mean_full) ** 2)

    control_true = []
    for t0 in time_vals[:-1]:
        lam_true = y_bundle[(X_bundle[:, 0] == t0) & (X_bundle[:, -1] == 0)][0]
        control_true.append(lam_true)
    control_true = np.array(control_true)
    control_mse = mean_squared_error(np.array(control_profile), control_true)

    metrics = {
        "time stride": stride,
        "model mse": mean_squared_error(y_test, rf.predict(X_test)),
        "cosine similarity": cos_sim,
        "final position deviation": np.linalg.norm(r_pred_all[-1] - r_ref),
        "mahalanobis_pred": mahalanobis_pred,
        "mahalanobis_mc_mean": mahalanobis_mc_mean,
        "trace diff (cov)": np.trace(np.abs(P_diag[-1] - np.diag(np.diag(np.outer(r_pred_all[-1] - mean_pred[-1], r_pred_all[-1] - mean_pred[-1]))))),
        "mc vs predicted trajectory mse": mc_vs_pred_mse,
        "control MSE": control_mse,
        "x mse": np.mean((r_ref[0] - r_pred_all[:, 0]) ** 2),
        "y mse": np.mean((r_ref[1] - r_pred_all[:, 1]) ** 2),
        "z mse": np.mean((r_ref[2] - r_pred_all[:, 2]) ** 2),
        "vx mse": np.mean((v_ref[0] - v_pred_all[:, 0]) ** 2),
        "vy mse": np.mean((v_ref[1] - v_pred_all[:, 1]) ** 2),
        "vz mse": np.mean((v_ref[2] - v_pred_all[:, 2]) ** 2),
    }

    count_gt3 = 0
    for i in range(r_pred_stack.shape[0]):
        dist = mahalanobis(r_pred_stack[i, -1], r_ref, inv_cov)
        metrics[f"mahalanobis_sigma_{i}"] = dist
        if dist > 3.0:
            count_gt3 += 1
    metrics["mahalanobis_count_gt3"] = count_gt3

    return metrics

# === Main Execution ===
results = []
W_bundle = joblib.load("sweep_stride_1_config_baseline_data.pkl")
Wm, Wc = W_bundle["Wm"], W_bundle["Wc"]

stride_files = sorted(glob.glob("sweep_stride_*_config_baseline_data.pkl"))
for file in stride_files:
    stride = int(file.split("_")[2])
    data = joblib.load(file)
    X, y = data["X"], data["y"]
    metrics = evaluate_model_with_sigma0_alignment(X, y, stride, Wm, Wc)
    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv("ml_stride_sigma0_aligned_metrics.csv", index=False)
print("Saved: ml_stride_sigma0_aligned_metrics.csv")
