import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import sys

# === Path Setup ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
bundle_idx = 32

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def final_position_deviation(r_pred_final, X_bundle):
    X_sigma0 = X_bundle[X_bundle[:, -1] == 0]
    times = np.sort(X_sigma0[:, 0])
    next_time = times[-1]
    next_row = X_sigma0[X_sigma0[:, 0] == next_time][0]
    r_sigma0_next = next_row[1:4]
    return np.linalg.norm(r_pred_final - r_sigma0_next)

def propagate_sigma0_truth(X_bundle, y_bundle):
    mask = X_bundle[:, -1] == 0
    X_sigma = X_bundle[mask]
    y_sigma = y_bundle[mask]
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma = y_sigma[sort_idx]

    r_true_all, v_true_all = [], []
    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        seg_times = np.linspace(t0, t1, 5)
        state = X_sigma[i, 1:8]
        lam_used = y_sigma[i]
        S = np.concatenate([state, lam_used])
        Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                       [t0, t1], S, t_eval=seg_times)
        r_xyz, v_xyz = mee2rv.mee2rv(*Sf.y.T[:, :6].T, mu)
        r_true_all.append(r_xyz)
        v_true_all.append(v_xyz)

    return np.vstack(r_true_all), np.vstack(v_true_all)

def monte_carlo_validation(X_init, control_profile, time_segments, num_samples=100, state_std=1e-4):
    r_mc_all = []

    for _ in range(num_samples):
        perturbed_state = X_init.copy()
        perturbed_state[:6] += np.random.normal(0, state_std, size=6)
        r_mc = []

        for i, (t0, t1) in enumerate(time_segments):
            lam_used = control_profile[i]
            S_mc = np.concatenate([perturbed_state, lam_used])
            seg_times = np.linspace(t0, t1, 5)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S_mc, t_eval=seg_times)
            r_xyz, _ = mee2rv.mee2rv(*Sf.y.T[:, :6].T, mu)
            r_mc.append(r_xyz)
            perturbed_state = Sf.y[:7, -1]

        r_mc_all.append(np.vstack(r_mc))

    return np.array(r_mc_all)

def evaluate_per_sigma_point(X, y, stride):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X[:, :-2])

    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = rf.predict(X_bundle[:, :-2])
    r_ref, v_ref = propagate_sigma0_truth(X_bundle, y_bundle)

    sigma_mask = X_bundle[:, -1] != 0
    X_filtered = X_bundle[sigma_mask]
    y_pred_filtered = y_pred_bundle[sigma_mask]
    y_true_filtered = y_bundle[sigma_mask]

    time_indices = np.unique(X_filtered[:, 0])
    time_indices.sort()
    num_segments = len(time_indices) - 1

    r_pred_all, control_errors, mc_control_profile = [], [], []
    initial_state = None

    for i in range(num_segments):
        t0, t1 = time_indices[i], time_indices[i + 1]
        segment_mask = (X_filtered[:, 0] == t0)
        state = X_filtered[segment_mask][0, 1:8]
        lam_pred = y_pred_filtered[segment_mask][0]
        lam_true = y_true_filtered[segment_mask][0]
        control_errors.append(np.mean((lam_pred - lam_true) ** 2))
        mc_control_profile.append(lam_pred)

        S = np.concatenate([state, lam_pred])
        seg_times = np.linspace(t0, t1, 5)
        Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                       [t0, t1], S, t_eval=seg_times)
        r_xyz, _ = mee2rv.mee2rv(*Sf.y.T[:, :6].T, mu)
        r_pred_all.append(r_xyz)
        if i == 0:
            initial_state = state

    r_pred_all = np.vstack(r_pred_all)

    # Monte Carlo propagation
    time_segments = list(zip(time_indices[:-1], time_indices[1:]))
    r_mc_pred = monte_carlo_validation(initial_state, mc_control_profile, time_segments)
    r_mc_mean = np.mean(r_mc_pred, axis=0)

    # === SolveTrajectories-style covariance comparison at final time step ===
    N = r_mc_pred.shape[0]
    W = np.ones(N) / N
    r_mc_final = r_mc_pred[:, -1, :]  # shape (N, 3)

    # Align r_pred_all with MC (e.g. last N points or repeat final)
    r_pred_final = np.repeat(r_pred_all[-1][None, :], N, axis=0)

    mean_mc = np.average(r_mc_final, axis=0, weights=W)
    mean_pred = np.average(r_pred_final, axis=0, weights=W)
    dev_mc = r_mc_final - mean_mc
    dev_pred = r_pred_final - mean_pred

    cov_mc = sum(W[i] * np.outer(dev_mc[i], dev_mc[i]) for i in range(N))
    cov_pred = sum(W[i] * np.outer(dev_pred[i], dev_pred[i]) for i in range(N))
    trace_diff = np.trace(np.abs(cov_mc - cov_pred))

    avg_traj_dev = np.mean(np.linalg.norm(r_pred_all - r_mc_mean, axis=1))
    final_dev = final_position_deviation(r_pred_all[-1], X_bundle)

    return [{
        "sigma": "ALL",
        "time stride": stride,
        "model mse": mean_squared_error(y_test, rf.predict(X_test)),
        "control mse (initial)": np.mean(control_errors),
        "control mse (full profile)": np.mean((y_pred_filtered - y_true_filtered) ** 2),
        "cosine similarity": cosine_similarity(y_pred_filtered[0], y_true_filtered[0]),
        "mc mean deviation (pred vs true control)": 0.0,
        "avg deviation from MC mean": avg_traj_dev,
        "avg 3sigma envelope": 3 * np.mean(np.std(r_mc_pred, axis=0)),
        "x mse": np.mean((r_ref[:len(r_pred_all), 0] - r_pred_all[:, 0]) ** 2),
        "y mse": np.mean((r_ref[:len(r_pred_all), 1] - r_pred_all[:, 1]) ** 2),
        "z mse": np.mean((r_ref[:len(r_pred_all), 2] - r_pred_all[:, 2]) ** 2),
        "vx mse": np.mean((v_ref[:len(r_pred_all), 0]) ** 2),  # placeholder
        "vy mse": np.mean((v_ref[:len(r_pred_all), 1]) ** 2),
        "vz mse": np.mean((v_ref[:len(r_pred_all), 2]) ** 2),
        "final position deviation": final_dev,
        "mc trace diff": trace_diff
    }], rf

def plot_predicted_vs_actual_trajectories(X_bundle, y_bundle, y_pred_bundle, stride):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sigma_indices = sorted(np.unique(X_bundle[:, -1]).astype(int))
    for sigma in sigma_indices:
        if sigma == 0:
            continue

        mask = X_bundle[:, -1] == sigma
        X_sigma = X_bundle[mask]
        y_sigma_pred = y_pred_bundle[mask]
        y_sigma_true = y_bundle[mask]

        sort_idx = np.argsort(X_sigma[:, 0])
        X_sigma = X_sigma[sort_idx]
        y_sigma_pred = y_sigma_pred[sort_idx]
        y_sigma_true = y_sigma_true[sort_idx]

        r_pred_all = []
        r_true_all = []

        for i in range(len(X_sigma) - 1):
            t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
            seg_times = np.linspace(t0, t1, 5)

            # Predicted
            state_pred = X_sigma[i, 1:8]
            lam_pred = y_sigma_pred[i]
            S_pred = np.concatenate([state_pred, lam_pred])
            Sf_pred = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                                [t0, t1], S_pred, t_eval=seg_times)
            r_pred, _ = mee2rv.mee2rv(*Sf_pred.y.T[:, :6].T, mu)
            r_pred_all.append(r_pred)

            # True
            state_true = X_sigma[i, 1:8]
            lam_true = y_sigma_true[i]
            S_true = np.concatenate([state_true, lam_true])
            Sf_true = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                                [t0, t1], S_true, t_eval=seg_times)
            r_true, _ = mee2rv.mee2rv(*Sf_true.y.T[:, :6].T, mu)
            r_true_all.append(r_true)

        r_pred_all = np.vstack(r_pred_all)
        r_true_all = np.vstack(r_true_all)

        ax.plot(r_true_all[:, 0], r_true_all[:, 1], r_true_all[:, 2], label=f"True σ={sigma}", linestyle='--')
        ax.plot(r_pred_all[:, 0], r_pred_all[:, 1], r_pred_all[:, 2], label=f"Pred σ={sigma}", linestyle='-', alpha=0.5)

    ax.set_title(f"Predicted vs. True Trajectories (Stride = {stride})")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend()
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig(f"trajectory_comparison_stride_{stride}.png", dpi=300)
    plt.show()

# === Main Loop ===
results = []
stride_files = sorted(glob.glob("data_bundle_32_stride_*.pkl"))

for file in stride_files:
    stride = int(file.split("_stride_")[1].split(".")[0])
    data = joblib.load(file)
    metrics, rf = evaluate_per_sigma_point(data['X'], data['y'], stride)
    results.extend(metrics)
    plot_predicted_vs_actual_trajectories(data['X'], data['y'], rf.predict(data['X'][:, :-2]), stride)

df = pd.DataFrame(results)
df = df[[ 
    "time stride", "sigma", "model mse",
    "control mse (initial)", "control mse (full profile)",
    "cosine similarity", "avg deviation from MC mean", "avg 3sigma envelope",
    "x mse", "y mse", "z mse", "vx mse", "vy mse", "vz mse",
    "final position deviation", "mc trace diff"
]]
df.sort_values(by=["time stride"], inplace=True)
df.to_csv("per_sigma_metrics_vs_stride.csv", index=False)
print("Saved: per_sigma_metrics_vs_stride.csv")
