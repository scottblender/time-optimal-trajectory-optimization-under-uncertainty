import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import odefunc
import mee2rv
import scipy.integrate
from scipy.linalg import eigh

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    eigvals, eigvecs = eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    radii = scale * np.sqrt(np.maximum(eigvals, 0))
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1)
    coords = coords @ eigvecs.T + mean
    ax.plot_surface(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2],
                    rstride=1, cstride=1, alpha=alpha, color=color)

# === Constants ===
mu = 27.899633640439433
F = 0.33
c = 4.4246246663455135
m0 = 4000
g0 = 9.81

# === Load Data ===
data = joblib.load('data.pkl')
X_full = data['X']
y_full = data['y']
trajectories = data['trajectories']
P_combined_history = data['P_combined_history']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_full[:, :-2], y_full, test_size=0.2, random_state=42
)

# === Train Model ===
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)

# === Evaluate Model ===
print("\nModel Evaluation:")
print(f"Train MSE: {mean_squared_error(y_train, rf.predict(X_train)):.6f}")
print(f"Test MSE:  {mean_squared_error(y_test, y_test_pred):.6f}")
print(f"Train R²:  {r2_score(y_train, rf.predict(X_train)):.6f}")
print(f"Test R²:   {r2_score(y_test, y_test_pred):.6f}")

# === Select Bundle ===
bundle_idx = 32
bundle_mask = (X_full[:, -2] == bundle_idx)
X_bundle = X_full[bundle_mask]
y_bundle = y_full[bundle_mask]
y_bundle_pred = rf.predict(X_bundle[:, :-2])

unique_sigma_points = np.unique(X_bundle[:, -1].astype(int))

# === 3D Trajectory Plot with Error Tubes and Start/End Ellipsoids ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for sigma_idx in unique_sigma_points:
    sigma_mask = (X_bundle[:, -1] == sigma_idx)
    X_sigma = X_bundle[sigma_mask]
    y_sigma_pred = y_bundle_pred[sigma_mask]

    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]

    r_pred_all = []
    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
        mee = X_sigma[i, 1:8]
        ctrl_pred = y_sigma_pred[i]
        S = np.concatenate([mee, ctrl_pred])
        tspan = np.linspace(t0, t1, 20)
        Sf = scipy.integrate.solve_ivp(lambda t, S: odefunc.odefunc(t, S, mu, F, c, m0, g0),
                                       [t0, t1], S, t_eval=tspan)
        r_xyz, _ = mee2rv.mee2rv(Sf.y.T[:, 0], Sf.y.T[:, 1], Sf.y.T[:, 2],
                                 Sf.y.T[:, 3], Sf.y.T[:, 4], Sf.y.T[:, 5], mu)
        r_pred_all.append(r_xyz)

    r_pred_all = np.vstack(r_pred_all)
    r_actual = trajectories[bundle_idx][0][sigma_idx][:, :3]

    print(f"\nSigma Point {sigma_idx}:")
    print("  Actual Start :", np.round(r_actual[0], 6))
    print("  Actual End   :", np.round(r_actual[-1], 6))
    print("  Predicted Start:", np.round(r_pred_all[0], 6))
    print("  Predicted End  :", np.round(r_pred_all[-1], 6))

    ax.plot(r_actual[:, 0], r_actual[:, 1], r_actual[:, 2], color='red', linestyle='-', alpha=0.4)
    ax.plot(r_pred_all[:, 0], r_pred_all[:, 1], r_pred_all[:, 2], color='blue', linestyle='--', alpha=0.7)

    ax.scatter(*r_actual[0], color='lime', marker='X', s=50, edgecolor='k')
    ax.scatter(*r_actual[-1], color='red', marker='X', s=50, edgecolor='k')
    ax.scatter(*r_pred_all[0], color='green', marker='^', s=50, edgecolor='k')
    ax.scatter(*r_pred_all[-1], color='blue', marker='^', s=50, edgecolor='k')

    if sigma_idx == 0:
        P = P_combined_history[bundle_idx][0]
        # Add ellipsoids throughout the trajectory (error tube)
        for k in range(0, len(P), max(1, len(P) // 5)):
            plot_3sigma_ellipsoid(ax, r_actual[k], P[k][:3, :3], color='orange', alpha=0.1)
        # Add ellipsoids at start and end
        plot_3sigma_ellipsoid(ax, r_actual[0], P[0][:3, :3], color='orange', alpha=0.3)
        plot_3sigma_ellipsoid(ax, r_actual[-1], P[-1][:3, :3], color='orange', alpha=0.3)

ax.set_title(f"Predicted vs Actual End-to-End Trajectories\nNominal Trajectory {bundle_idx}", fontsize=14)
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.view_init(elev=25, azim=120)
ax.set_box_aspect([1, 1, 0.5])
ax.legend(['Actual', 'Predicted'], loc='upper left')
plt.tight_layout()
plt.show()

