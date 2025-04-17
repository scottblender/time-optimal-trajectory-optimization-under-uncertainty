import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import mee2rv, odefunc

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.05, scale=3.0):
    eigvals, eigvecs = eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    radii = scale * np.sqrt(eigvals)
    u, v = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], alpha=alpha, color=color)

# === Constants ===
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81

# === Load Data ===
data_sigma = joblib.load('data_bundle_32.pkl')
data_mc = joblib.load('monte_carlo_bundle_32.pkl')

def train_and_evaluate(label, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print(f"\n--- {label} Model ---")
    print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.6f}")
    print(f"Test MSE:  {mean_squared_error(y_test, y_pred_test):.6f}")
    print(f"Train R²:  {r2_score(y_train, y_pred_train):.6f}")
    print(f"Test R²:   {r2_score(y_test, y_pred_test):.6f}")

    mape = np.mean(np.abs((y_test - y_pred_test) / np.abs(y_test)))
    print("Mean Absolute Percentage Error (MAPE):", round(mape * 100, 2))
    print("Accuracy:", round(100 * (1 - mape), 2))

    return rf

rf_sigma = train_and_evaluate("Sigma Point", data_sigma['X'], data_sigma['y'])
rf_mc = train_and_evaluate("Monte Carlo", data_mc['X'], data_mc['y'])

def compare_prediction(data, rf, ax, label):
    X_full = data['X']
    y_full = data['y']
    trajectories = data['trajectories']
    P_combined_history = data['P_combined_history']
    bundle_idx = 32

    X_bundle = X_full[X_full[:, -2] == bundle_idx]
    y_bundle_pred = rf.predict(X_bundle[:, :-2])
    unique_ids = np.unique(X_bundle[:, -1].astype(int))

    for idx in unique_ids:
        mask = X_bundle[:, -1] == idx
        X_sample = X_bundle[mask]
        y_sample_pred = y_bundle_pred[mask]
        sort_idx = np.argsort(X_sample[:, 0])
        X_sample = X_sample[sort_idx]
        y_sample_pred = y_sample_pred[sort_idx]

        r_actual = trajectories[0][0][idx][:, :3]
        global_times = np.linspace(X_sample[0, 0], X_sample[-1, 0], r_actual.shape[0])

        r_pred_all = []
        for i in range(len(X_sample) - 1):
            t0, t1 = X_sample[i, 0], X_sample[i + 1, 0]
            seg_times = global_times[(global_times >= t0) & (global_times <= t1)]
            mee = X_sample[i, 1:8]
            ctrl_pred = y_sample_pred[i]
            S = np.concatenate([mee, ctrl_pred])
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S, t_eval=seg_times)
            r_xyz, _ = mee2rv.mee2rv(Sf.y.T[:, 0], Sf.y.T[:, 1], Sf.y.T[:, 2],
                                     Sf.y.T[:, 3], Sf.y.T[:, 4], Sf.y.T[:, 5], mu)
            r_pred_all.append(r_xyz)

        r_pred_all = np.vstack(r_pred_all)

        # === Plot actual and predicted trajectory ===
        if idx == 0:
            ax.plot(r_actual[:, 0], r_actual[:, 1], r_actual[:, 2], color='darkred', linewidth=2.5, label=f'{label} (Actual)')
            ax.plot(r_pred_all[:, 0], r_pred_all[:, 1], r_pred_all[:, 2], linestyle='--', color='blue', linewidth=2.0, alpha=0.9, label=f'{label} (Predicted)')
        else:
            ax.plot(r_actual[:, 0], r_actual[:, 1], r_actual[:, 2], color='darkred', alpha=0.6)
            ax.plot(r_pred_all[:, 0], r_pred_all[:, 1], r_pred_all[:, 2], linestyle='--', color='blue', alpha=0.4)

        # === Mark Start/End ===
        ax.scatter(*r_actual[0], color='green', marker='o', s=30)
        ax.scatter(*r_actual[-1], color='darkred', marker='X', s=30)

        # === Ellipsoids for nominal only ===
        if idx == 0:
            cov_list = P_combined_history[0][0]
            for i in np.linspace(0, len(cov_list) - 1, 5, dtype=int):
                plot_3sigma_ellipsoid(ax, r_actual[i], cov_list[i][:3, :3], color='orange', alpha=0.1)

# === Plot Setup ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

compare_prediction(data_sigma, rf_sigma, ax, "Sigma Point")

# === Final Formatting ===
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.view_init(elev=25, azim=135)
ax.set_box_aspect([1.25, 1, 0.75])
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
