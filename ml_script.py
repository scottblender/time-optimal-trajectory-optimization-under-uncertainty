import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import odefunc
import mee2rv
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import pandas as pd

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    # Compute eigenvalues and eigenvectors of the covariance matrix.
    eigvals, eigvecs = eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Compute radii by scaling the standard deviations.
    radii = scale * np.sqrt(np.maximum(eigvals, 0))
    # Create a mesh grid for a sphere.
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1)
    # Rotate and translate the sphere to form an ellipsoid.
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

# Prepare a list to accumulate metrics per sigma point.
table_data = []
# Also prepare a list to store L2 norm error data for log-scale plotting.
sigma_error_data = []

# Create a 3D plot of trajectories (with ellipsoids).
fig_traj = plt.figure(figsize=(12, 10))
ax_traj = fig_traj.add_subplot(111, projection='3d')

for sigma_idx in unique_sigma_points:
    sigma_mask = (X_bundle[:, -1] == sigma_idx)
    X_sigma = X_bundle[sigma_mask]
    y_sigma_pred = y_bundle_pred[sigma_mask]

    # Sort by time (assumed to be in the first column).
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]

    # Get the actual trajectory (assumed to be defined on a fine time grid).
    r_actual = trajectories[bundle_idx][0][sigma_idx][:, :3]

    # Create a global time grid matching the actual trajectory.
    global_times = np.linspace(X_sigma[0, 0], X_sigma[-1, 0], r_actual.shape[0])

    # Build the predicted trajectory over the control segments.
    r_pred_all = []
    for i in range(len(X_sigma) - 1):
        t0, t1 = X_sigma[i, 0], X_sigma[i+1, 0]
        # Use all time points in this segment.
        seg_times = global_times[(global_times >= t0) & (global_times <= t1)]
        mee = X_sigma[i, 1:8]
        ctrl_pred = y_sigma_pred[i]
        S = np.concatenate([mee, ctrl_pred])
        Sf = solve_ivp(
            lambda t, S: odefunc.odefunc(t, S, mu, F, c, m0, g0),
            [t0, t1], S, t_eval=seg_times
        )
        r_xyz, _ = mee2rv.mee2rv(
            Sf.y.T[:, 0], Sf.y.T[:, 1], Sf.y.T[:, 2],
            Sf.y.T[:, 3], Sf.y.T[:, 4], Sf.y.T[:, 5],
            mu
        )
        r_pred_all.append(r_xyz)
    r_pred_all = np.vstack(r_pred_all)

    # Compute the per-coordinate MSE (mean squared error).
    mse_x = np.mean((r_pred_all[:, 0] - r_actual[:, 0])**2)
    mse_y = np.mean((r_pred_all[:, 1] - r_actual[:, 1])**2)
    mse_z = np.mean((r_pred_all[:, 2] - r_actual[:, 2])**2)
    # Compute the overall MSE (averaged across all coordinates).
    mse_overall = mean_squared_error(r_actual, r_pred_all)

    # Compute overall L2 error (Euclidean norm) over time.
    error_l2 = np.linalg.norm(r_pred_all - r_actual, axis=1)
    max_l2 = np.max(error_l2)
    
    # Compute maximum L2 error per coordinate.
    max_l2_x = np.max(np.abs(r_pred_all[:, 0] - r_actual[:, 0]))
    max_l2_y = np.max(np.abs(r_pred_all[:, 1] - r_actual[:, 1]))
    max_l2_z = np.max(np.abs(r_pred_all[:, 2] - r_actual[:, 2]))
    
    table_data.append({
        "Sigma Point": sigma_idx,
        "MSE (overall)": mse_overall,
        "MSE (X)": mse_x,
        "MSE (Y)": mse_y,
        "MSE (Z)": mse_z,
        "Max L2 Norm (overall)": max_l2,
        "Max L2 Norm (X)": max_l2_x,
        "Max L2 Norm (Y)": max_l2_y,
        "Max L2 Norm (Z)": max_l2_z
    })
    
    # Store L2 error and global_times for log-scale plot.
    sigma_error_data.append({
        "sigma": sigma_idx,
        "global_times": global_times,
        "error_l2": error_l2
    })

    # Plot the actual and predicted trajectories.
    ax_traj.plot(r_actual[:, 0], r_actual[:, 1], r_actual[:, 2],
                 color='red', linestyle='-', alpha=0.4)
    ax_traj.plot(r_pred_all[:, 0], r_pred_all[:, 1], r_pred_all[:, 2],
                 color='blue', linestyle='--', alpha=0.7)
    ax_traj.scatter(*r_actual[0], color='lime', marker='X', s=50, edgecolor='k')
    ax_traj.scatter(*r_actual[-1], color='red', marker='X', s=50, edgecolor='k')
    ax_traj.scatter(*r_pred_all[0], color='green', marker='^', s=50, edgecolor='k')
    ax_traj.scatter(*r_pred_all[-1], color='blue', marker='^', s=50, edgecolor='k')

    # Add ellipsoids for the first sigma point using its covariance history.
    if sigma_idx == unique_sigma_points[0]:
        P = P_combined_history[bundle_idx][0]
        for k in range(0, len(P), max(1, len(P)//5)):
            plot_3sigma_ellipsoid(ax_traj, r_actual[k], P[k][:3, :3], color='orange', alpha=0.1)
        plot_3sigma_ellipsoid(ax_traj, r_actual[0], P[0][:3, :3], color='orange', alpha=0.3)
        plot_3sigma_ellipsoid(ax_traj, r_actual[-1], P[-1][:3, :3], color='orange', alpha=0.3)

# Finalize the 3D trajectory plot.
ax_traj.set_title(f"Predicted vs Actual End-to-End Trajectories\nNominal Trajectory {bundle_idx}", fontsize=14)
ax_traj.set_xlabel('X [km]')
ax_traj.set_ylabel('Y [km]', labelpad=20)
ax_traj.set_zlabel('Z [km]')
ax_traj.view_init(elev=25, azim=120)
ax_traj.set_box_aspect([1, 1, 0.5])
ax_traj.legend(['Actual', 'Predicted'], loc='upper left')
plt.tight_layout()
plt.show()

# Format floating-point numbers to 6 decimal places.
pd.options.display.float_format = '{:.8f}'.format

# Create and print the table of metrics.
df = pd.DataFrame(table_data)
print("\nSummary of Per-Coordinate and Overall MSE and Max L2 Norm for Each Sigma Point:")
print(df.to_string(index=False))


