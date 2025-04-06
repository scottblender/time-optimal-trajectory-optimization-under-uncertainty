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

# Open a log file to write control comparisons
control_log = open("control_comparisons.txt", "w")

# Load data and constants
mu = 27.899633640439433
F = 0.33
c = 4.4246246663455135
m0 = 4000
g0 = 9.81

data = joblib.load('data.pkl')
X = data['X']
y = data['y']
time_match_indices = data['time_match_indices']

# Train/test split
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange(len(X)), test_size=0.2, random_state=42
)

# Train model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)

# Evaluate and print model performance
train_mse = mean_squared_error(y_train, rf.predict(X_train))
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, rf.predict(X_train))
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nModel Evaluation:")
print(f"Train MSE: {train_mse:.6f}")
print(f"Test MSE:  {test_mse:.6f}")
print(f"Train R²:  {train_r2:.6f}")
print(f"Test R²:   {test_r2:.6f}")

# Select a random bundle to visualize
random_bundle_idx = int(np.random.choice(X[:, -2], size=1)[0])  # -2 is bundle_idx column

# Extract rows from test set that match the bundle
test_bundle_mask = (X_test[:, -2] == random_bundle_idx)
X_test_bundle = X_test[test_bundle_mask]
y_test_bundle = y_test[test_bundle_mask]
y_test_pred_bundle = y_test_pred[test_bundle_mask]

# Get all sigma point indices in this bundle
unique_sigma_points = np.unique(X_test_bundle[:, -1].astype(int))  # -1 is sigma_point_idx

# Plot all sigma point trajectories for this bundle
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for sigma_idx in unique_sigma_points:
    sigma_mask = (X_test_bundle[:, -1] == sigma_idx)
    X_sigma = X_test_bundle[sigma_mask]
    y_sigma_actual = y_test_bundle[sigma_mask]
    y_sigma_pred = y_test_pred_bundle[sigma_mask]

    # Sort by time
    sort_idx = np.argsort(X_sigma[:, 0])
    X_sigma = X_sigma[sort_idx]
    y_sigma_actual = y_sigma_actual[sort_idx]
    y_sigma_pred = y_sigma_pred[sort_idx]

    # Initialize state
    r_eci_actual = []
    r_eci_pred = []

    for i in range(len(X_sigma) - 1):
        
        tstart = X_sigma[i, 0]
        tend = X_sigma[i + 1, 0]
        mee_state = X_sigma[i, 1:8]  # p, f, g, h, k, L, m
        ctrl_actual = y_sigma_actual[i]
        ctrl_pred = y_sigma_pred[i]

       # Format controls as strings with 5 decimal places
        actual_str = ", ".join([f"{val:.5f}" for val in ctrl_actual])
        predicted_str = ", ".join([f"{val:.5f}" for val in ctrl_pred])

        # Create message
        msg = (
            f"\nTime interval [{tstart:.3f}, {tend:.3f}] — Sigma Point {int(sigma_idx)}\n"
            f"Actual Control   : [{actual_str}]\n"
            f"Predicted Control: [{predicted_str}]\n"
        )

        # Print to console
        print(msg)

        # Write to log file
        control_log.write(msg)

        # Propagate actual
        S_actual = np.concatenate([mee_state, ctrl_actual])
        func = lambda t, S: odefunc.odefunc(t, S, mu, F, c, m0, g0)
        tspan = np.linspace(tstart, tend, 10)
        Sf_a = scipy.integrate.solve_ivp(func, [tstart, tend], S_actual, t_eval=tspan)
        r_a, _ = mee2rv.mee2rv(Sf_a.y[0], Sf_a.y[1], Sf_a.y[2], Sf_a.y[3], Sf_a.y[4], Sf_a.y[5], mu)
        r_eci_actual.append(r_a)

        # Propagate predicted
        S_pred = np.concatenate([mee_state, ctrl_pred])
        Sf_p = scipy.integrate.solve_ivp(func, [tstart, tend], S_pred, t_eval=tspan)
        r_p, _ = mee2rv.mee2rv(Sf_p.y[0], Sf_p.y[1], Sf_p.y[2], Sf_p.y[3], Sf_p.y[4], Sf_p.y[5], mu)
        r_eci_pred.append(r_p)

    r_eci_actual = np.vstack(r_eci_actual)
    r_eci_pred = np.vstack(r_eci_pred)

    # Plot actual trajectory
    ax.plot(r_eci_actual[:, 0], r_eci_actual[:, 1], r_eci_actual[:, 2],
            color='red', linestyle='-', alpha=0.3 if sigma_idx != 0 else 1.0,
            label='Actual' if sigma_idx == 0 else None)

    # Plot predicted trajectory
    ax.plot(r_eci_pred[:, 0], r_eci_pred[:, 1], r_eci_pred[:, 2],
            color='blue', linestyle='--', linewidth=2,
            alpha=0.3 if sigma_idx != 0 else 1.0,
            label='Predicted' if sigma_idx == 0 else None)

   # Mark start and end points for each sigma point
    ax.scatter(r_eci_actual[0, 0], r_eci_actual[0, 1], r_eci_actual[0, 2],
               color='lime', marker='X', s=60 if sigma_idx != 0 else 100,
               label='Start (Actual)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_actual[-1, 0], r_eci_actual[-1, 1], r_eci_actual[-1, 2],
               color='red', marker='X', s=60 if sigma_idx != 0 else 100,
               label='End (Actual)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_pred[0, 0], r_eci_pred[0, 1], r_eci_pred[0, 2],
               color='green', marker='^', s=60 if sigma_idx != 0 else 100,
               label='Start (Predicted)' if sigma_idx == 0 else None)

    ax.scatter(r_eci_pred[-1, 0], r_eci_pred[-1, 1], r_eci_pred[-1, 2],
               color='blue', marker='^', s=60 if sigma_idx != 0 else 100,
               label='End (Predicted)' if sigma_idx == 0 else None)

# Labels
ax.set_title(f"Predicted vs Actual End-to-End Trajectories\nBundle {random_bundle_idx}", fontsize=14)
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
