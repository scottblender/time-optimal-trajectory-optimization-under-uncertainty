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

# === Constants ===
mu = 27.899633640439433
F = 0.33
c = 4.4246246663455135
m0 = 4000
g0 = 9.81

# === Load Data ===
data = joblib.load('data.pkl')
X = data['X']
y = data['y']
trajectories = data['trajectories']

# === Train/Test Split ===
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange(len(X)), test_size=0.2, random_state=42
)

# === Train Model ===
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)

# # === Evaluate Model ===
# train_mse = mean_squared_error(y_train, rf.predict(X_train))
# test_mse = mean_squared_error(y_test, y_test_pred)
# train_r2 = r2_score(y_train, rf.predict(X_train))
# test_r2 = r2_score(y_test, y_test_pred)

# print("\nModel Evaluation:")
# print(f"Train MSE: {train_mse:.6f}")
# print(f"Test MSE:  {test_mse:.6f}")
# print(f"Train R²:  {train_r2:.6f}")
# print(f"Test R²:   {test_r2:.6f}")

# === Select Bundle ===
random_bundle_idx = 32  # Or use: int(np.random.choice(X[:, -2], size=1)[0])

# === Use full X for bundle filtering to retain full time coverage ===
bundle_mask = (X[:, -2] == random_bundle_idx)
X_bundle = X[bundle_mask]
y_bundle = y[bundle_mask]

# Predict controls using the trained model
y_bundle_pred = rf.predict(X_bundle)

# === Unique Sigma Points in this Bundle ===
unique_sigma_points = np.unique(X_bundle[:, -1].astype(int))

# === Open Log File ===
control_log = open("control_comparisons.txt", "w")

# === Time Coverage Check ===
print(f"\nChecking time coverage per sigma point for Bundle {random_bundle_idx}:")
tmin_global = np.min(X_bundle[:, 0])
tmax_global = np.max(X_bundle[:, 0])
print(f"Expected time range: [{tmin_global:.3f}, {tmax_global:.3f}]")
for sigma_idx in unique_sigma_points:
    sigma_times = X_bundle[X_bundle[:, -1] == sigma_idx][:, 0]
    tmin, tmax = np.min(sigma_times), np.max(sigma_times)
    if abs(tmin - tmin_global) > 1e-3 or abs(tmax - tmax_global) > 1e-3:
        print(f"⚠️  Sigma Point {sigma_idx} has time range [{tmin:.4f}, {tmax:.4f}]")

# === 3D Plot Setup ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# === Loop Through Sigma Points ===
for sigma_idx in unique_sigma_points:
    sigma_mask = (X_bundle[:, -1] == sigma_idx)
    X_sigma = X_bundle[sigma_mask]
    y_sigma_actual = y_bundle[sigma_mask]
    y_sigma_pred = y_bundle_pred[sigma_mask]

    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_actual = y_sigma_actual[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]
    time_grid = X_sigma[:, 0]

    r_eci_actual = []
    r_eci_pred = []

    for i in range(len(time_grid) - 1):
        tstart, tend = time_grid[i], time_grid[i + 1]
        mee_state = X_sigma[i, 1:8]
        ctrl_actual = y_sigma_actual[i]
        ctrl_pred = y_sigma_pred[i]

        actual_str = ", ".join([f"{val:.5f}" for val in ctrl_actual])
        predicted_str = ", ".join([f"{val:.5f}" for val in ctrl_pred])
        msg = (
            f"\nTime interval [{tstart:.3f}, {tend:.3f}] — Sigma Point {int(sigma_idx)}\n"
            f"Actual Control   : [{actual_str}]\n"
            f"Predicted Control: [{predicted_str}]\n"
        )
        print(msg)
        control_log.write(msg)

        S_actual = np.concatenate([mee_state, ctrl_actual])
        func = lambda t, S: odefunc.odefunc(t, S, mu, F, c, m0, g0)
        tspan = np.linspace(tstart, tend, 10)
        Sf_a = scipy.integrate.solve_ivp(func, [tstart, tend], S_actual, rtol=1e-6, atol=1e-8, t_eval=tspan)
        r_a, _ = mee2rv.mee2rv(Sf_a.y.T[:, 0], Sf_a.y.T[:, 1], Sf_a.y.T[:, 2],
                               Sf_a.y.T[:, 3], Sf_a.y.T[:, 4], Sf_a.y.T[:, 5], mu)
        r_eci_actual.append(r_a)

        S_pred = np.concatenate([mee_state, ctrl_pred])
        Sf_p = scipy.integrate.solve_ivp(func, [tstart, tend], S_pred, rtol=1e-6, atol=1e-8, t_eval=tspan)
        r_p, _ = mee2rv.mee2rv(Sf_p.y.T[:, 0], Sf_p.y.T[:, 1], Sf_p.y.T[:, 2],
                               Sf_p.y.T[:, 3], Sf_p.y.T[:, 4], Sf_p.y.T[:, 5], mu)
        r_eci_pred.append(r_p)

    r_eci_actual = np.vstack(r_eci_actual)
    r_eci_pred = np.vstack(r_eci_pred)

    # === Compare with stored trajectory ===
    traj_stored = trajectories[random_bundle_idx][0][sigma_idx]  # shape: (N, 7) — r, v, m
    r_xyz_stored = traj_stored[:, :3]

    print(f"\n--- Position Comparison for Bundle {random_bundle_idx}, Sigma Point {sigma_idx} ---")

    print("\nStored Trajectory:")
    print(f"Start (XYZ): {r_xyz_stored[0]}")
    print(f"End   (XYZ): {r_xyz_stored[-1]}")

    print("\nPropagated Trajectory:")
    print(f"Start (XYZ): {r_eci_actual[0]}")
    print(f"End   (XYZ): {r_eci_actual[-1]}")

    # === Plotting ===
    ax.plot(r_eci_actual[:, 0], r_eci_actual[:, 1], r_eci_actual[:, 2],
            color='red', linestyle='-', alpha=0.3 if sigma_idx != 0 else 1.0,
            label='Actual' if sigma_idx == 0 else None)

    ax.plot(r_eci_pred[:, 0], r_eci_pred[:, 1], r_eci_pred[:, 2],
            color='blue', linestyle='--', linewidth=2,
            alpha=0.3 if sigma_idx != 0 else 1.0,
            label='Predicted' if sigma_idx == 0 else None)

    ax.scatter(r_eci_actual[0, 0], r_eci_actual[0, 1], r_eci_actual[0, 2],
               color='lime', marker='X', s=60 if sigma_idx != 0 else 100, edgecolor='black',
               label='Start (Actual)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_actual[-1, 0], r_eci_actual[-1, 1], r_eci_actual[-1, 2],
               color='red', marker='X', s=60 if sigma_idx != 0 else 100, edgecolor='black',
               label='End (Actual)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_pred[0, 0], r_eci_pred[0, 1], r_eci_pred[0, 2],
               color='green', marker='^', s=60 if sigma_idx != 0 else 100, edgecolor='black',
               label='Start (Predicted)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_pred[-1, 0], r_eci_pred[-1, 1], r_eci_pred[-1, 2],
               color='blue', marker='^', s=60 if sigma_idx != 0 else 100, edgecolor='black',
               label='End (Predicted)' if sigma_idx == 0 else None)

# === Final Plot Touches ===
ax.set_title(f"Predicted vs Actual End-to-End Trajectories\nBundle {random_bundle_idx}", fontsize=14)
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.view_init(elev=25, azim=120)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === Close Log File ===
control_log.close()
