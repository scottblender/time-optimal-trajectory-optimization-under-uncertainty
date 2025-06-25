import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
import solve_trajectories
import generate_monte_carlo_trajectories

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    eigvals, eigvecs = eigh(cov)
    radii = scale * np.sqrt(np.maximum(eigvals, 0))
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2],
                    rstride=1, cstride=1, alpha=alpha, color=color)

# === Load bundle data ===
data = joblib.load("bundle_data.pkl")
r_tr = data["r_tr"]
v_tr = data["v_tr"]
mass_tr = data["mass_tr"]
S_bundles = data["S_bundles"]
r_bundles = data["r_bundles"]
v_bundles = data["v_bundles"]
new_lam_bundles = data["new_lam_bundles"]
mass_bundles = data["mass_bundles"]
backTspan = data["backTspan"]
mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]

# === Plot Bundle and Nominal Trajectories ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot all perturbed bundle trajectories
num_bundles = r_bundles.shape[2]
for i in range(num_bundles):
    x_r_bundle = r_bundles[::-1, 0, i]
    y_r_bundle = r_bundles[::-1, 1, i]
    z_r_bundle = r_bundles[::-1, 2, i]
    ax.plot(x_r_bundle, y_r_bundle, z_r_bundle, color='blue', alpha=0.15, linewidth=1)

# Plot the nominal trajectory
x_r = r_tr[:, 0]
y_r = r_tr[:, 1]
z_r = r_tr[:, 2]
ax.plot(x_r, y_r, z_r, color='black', linewidth=2.5, label='Nominal Trajectory')

# Mark start and end of nominal trajectory
ax.scatter(x_r[0], y_r[0], z_r[0], color='green', marker='o', s=50, label='Start')
ax.scatter(x_r[-1], y_r[-1], z_r[-1], color='red', marker='X', s=60, label='End')

# Formatting
ax.set_xlabel('X Position [km]')
ax.set_ylabel('Y Position [km]')
ax.set_zlabel('Z Position [km]')
ax.set_box_aspect([1.25, 1, 0.75])
ax.grid(True)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig("bundle_vs_nominal_trajectories.png", dpi=300)
plt.close()

# === Plot Nominal Trajectory Only ===
fig_nominal = plt.figure(figsize=(10, 8))
ax_nominal = fig_nominal.add_subplot(111, projection='3d')

# Plot nominal trajectory
ax_nominal.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='black', linewidth=3, label='Nominal Trajectory')

# Start and end points
ax_nominal.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='green', marker='o', s=30, label='Start')
ax_nominal.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='red', marker='X', s=30, label='End')

# Formatting
ax_nominal.set_xlabel('X Position [km]')
ax_nominal.set_ylabel('Y Position [km]')
ax_nominal.set_zlabel('Z Position [km]')
ax_nominal.set_box_aspect([1.25, 1, 0.75])
ax_nominal.grid(True)
ax_nominal.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig("nominal_only_trajectory.png", dpi=300)
plt.close()

# === Parameters ===
bundle_index = 32
r_b = r_bundles[::-1, :, bundle_index][:, :, np.newaxis]
v_b = v_bundles[::-1, :, bundle_index][:, :, np.newaxis]
m_b = mass_bundles[::-1, bundle_index][:, np.newaxis]
lam_b = new_lam_bundles[::-1, :, bundle_index][:, :, np.newaxis]
S_b = S_bundles[::-1, :, bundle_index][:, :, np.newaxis]

# === Sigma Point Uncertainty Sweep Configurations ===
uncertainty_configs = [
    {"name": "baseline",         "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
    {"name": "high_pos",         "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
    {"name": "high_vel",         "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.0001]])},
    {"name": "high_mass",        "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
    {"name": "high_pos_vel",     "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.0001]])},
    {"name": "high_pos_mass",    "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
    {"name": "high_vel_mass",    "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.01]])},
    {"name": "high_all",         "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.01]])},
]

# === Sigma Point Settings ===
nsd = 7
beta, kappa = 2, float(3 - nsd)
alpha = np.sqrt(9 / (nsd + kappa))

# === Sweep Loops ===
time_strides = np.arange(1, 9) #np.array([1])
for stride in time_strides:
    time_steps = np.arange(len(backTspan), step=stride)
    time_steps = time_steps[0:2]
    num_time_steps = len(time_steps)

    for config in uncertainty_configs:
        name = config["name"]
        print(f"\n=== Running stride {stride}, distribution {name} ===")

        sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
            nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
            P_pos=config["P_pos"], P_vel=config["P_vel"], P_mass=config["P_mass"],
            num_time_steps=num_time_steps, backTspan=backTspan,
            r_bundles=r_b, v_bundles=v_b, mass_bundles=m_b
        )

        trajectories, P_hist, means_hist, X, y = solve_trajectories.solve_trajectories_with_covariance(
            backTspan, time_steps, num_time_steps, 1, sigmas_combined,
            lam_b, m_b, mu, F, c, m0, g0, Wm, Wc
        )
        print("X shape:", X.shape)  # Input features
        print("y shape:", y.shape)  # Target values
        mc_traj, mc_P_hist, mc_means_hist, mc_X, mc_y = generate_monte_carlo_trajectories.generate_monte_carlo_trajectories(
            backTspan=backTspan, time_steps=time_steps, num_time_steps=num_time_steps,
            num_bundles=1, sigmas_combined=sigmas_combined,
            new_lam_bundles=lam_b, mu=mu, F=F, c=c, m0=m0, g0=g0,
            num_samples=1000
        )

        out_prefix = f"sweep_stride_{stride}_config_{name}"
        joblib.dump({"X": X, "y": y, "trajectories": trajectories,
                     "P_combined_history": P_hist, "means_history": means_hist,
                    "Wm": Wm, "Wc": Wc},
                    f"{out_prefix}_data.pkl")
        joblib.dump({"X": mc_X, "y": mc_y, "trajectories": mc_traj,
                     "P_combined_history": mc_P_hist, "means_history": mc_means_hist},
                    f"{out_prefix}_monte_carlo.pkl")

        # Mahalanobis Diagnostic
        unique_times = np.unique(np.round(X[:, 0], decimals=10))
        if len(unique_times) < 2:
            print(f"[Warning] Not enough time steps for stride={stride}, config={name}")
            continue

        t_k1 = unique_times[-1]
        X_tk1 = X[np.isclose(X[:, 0], t_k1) & (X[:, -2] == bundle_index)]
        sigma0_candidates = X_tk1[(X_tk1[:, -1] == 0)]
        appended_row = next((row for row in sigma0_candidates if np.allclose(row[8:15], 0.0)), None)
        if appended_row is None:
            print(f"[Warning] Appended sigma 0 not found for stride={stride}, config={name}")
            continue

        p, f, g, h, k, L = appended_row[1:7]
        r_ref, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                          np.array([h]), np.array([k]), np.array([L]), mu)
        r_ref = r_ref[0]

        P_final = P_hist[0][0][-1][:3, :3]
        inv_cov = inv(P_final)

        print(f"\n=== Mahalanobis Distances at t = {t_k1:.6f} for stride={stride}, config={name} ===")
        for sigma_idx in range(15):
            rows = X_tk1[(X_tk1[:, -1] == sigma_idx)]
            for row in rows:
                if not np.allclose(row[8:15], 0.0):
                    p, f, g, h, k, L = row[1:7]
                    r_eci, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                                      np.array([h]), np.array([k]), np.array([L]), mu)
                    d = mahalanobis(r_eci[0], r_ref, inv_cov)
                    print(f"Sigma {sigma_idx:2d}: Mahalanobis distance = {d:.6f}")
                    break
            else:
                print(f"Sigma {sigma_idx:2d}: [not found]")
