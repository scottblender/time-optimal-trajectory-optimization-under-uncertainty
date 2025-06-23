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

# === Plot nominal + bundles ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='black', linewidth=2.5, label='Nominal Trajectory')
ax.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='green', s=50, label='Start')
ax.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='red', s=60, label='End')
for i in range(r_bundles.shape[2]):
    ax.plot(r_bundles[::-1, 0, i], r_bundles[::-1, 1, i], r_bundles[::-1, 2, i],
            color='blue', alpha=0.15, linewidth=1)
ax.set_xlabel('X Position [km]')
ax.set_ylabel('Y Position [km]')
ax.set_zlabel('Z Position [km]')
ax.set_box_aspect([1.25, 1, 0.75])
ax.legend()
plt.tight_layout()
plt.show()

# === Plot Nominal Trajectory Only ===
fig_nominal = plt.figure(figsize=(10, 8))
ax_nominal = fig_nominal.add_subplot(111, projection='3d')

# Plot nominal trajectory
ax_nominal.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='black', linewidth=3, label='Nominal Trajectory')

# Start and end points
ax_nominal.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='green', marker='o', s=30, label='Start')
ax_nominal.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='red', marker='X', s=30, label='End')

# Axis labels and view formatting
ax_nominal.set_xlabel('X [km]')
ax_nominal.set_ylabel('Y [km]')
ax_nominal.set_zlabel('Z [km]')
ax_nominal.view_init(elev=25, azim=135)
ax_nominal.set_box_aspect([1.25, 1, 0.75])
ax_nominal.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === Propagation Setup ===
bundle_index = 32
r_b = r_bundles[::-1, :, bundle_index][:, :, np.newaxis]
v_b = v_bundles[::-1, :, bundle_index][:, :, np.newaxis]
m_b = mass_bundles[::-1, bundle_index][:, np.newaxis]
lam_b = new_lam_bundles[::-1, :, bundle_index][:, :, np.newaxis]
S_b = S_bundles[::-1, :, bundle_index][:, :, np.newaxis]

# === Sigma Point Configuration ===
nsd = 7
beta, kappa = 2, float(3 - nsd)
alpha = np.sqrt(9 / (nsd + kappa))
P_pos = np.eye(3) * 1e-2
P_vel = np.eye(3) * 1e-4
P_mass = np.array([[1e-4]])

# === Time stride settings ===
time_stride = 1
time_steps = np.arange(len(backTspan), step=time_stride)
tstart_index, tend_index = 0, 1
time_steps = time_steps[tstart_index:tend_index + 1]
num_time_steps = len(time_steps)

# === Generate sigma points ===
sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
    nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
    P_pos=P_pos, P_vel=P_vel, P_mass=P_mass,
    num_time_steps=num_time_steps, backTspan=backTspan,
    r_bundles=r_b, v_bundles=v_b, mass_bundles=m_b
)

# === Save sigma weights to CSV ===
weights_df = pd.DataFrame({
    "Wm": Wm,
    "Wc": Wc
})
weights_df.to_csv("sigma_weights.csv", index=False)
print("Saved sigma weights to sigma_weights.csv")

# === Sigma propagation ===
trajectories, P_hist, means_hist, X, y = solve_trajectories.solve_trajectories_with_covariance(
    backTspan, time_steps, num_time_steps, 1, sigmas_combined,
    lam_b, m_b, mu, F, c, m0, g0, Wm, Wc
)

# === Monte Carlo propagation ===
mc_traj, mc_P_hist, mc_means_hist, mc_X, mc_y = generate_monte_carlo_trajectories.generate_monte_carlo_trajectories(
    backTspan=backTspan, time_steps=time_steps, num_time_steps=num_time_steps,
    num_bundles=1, sigmas_combined=sigmas_combined,
    new_lam_bundles=lam_b, mu=mu, F=F, c=c, m0=m0, g0=g0,
    num_samples=1000
)

# === Save outputs ===
joblib.dump({"X": X, "y": y, "trajectories": trajectories,
             "P_combined_history": P_hist, "means_history": means_hist},
            "data_bundle_32.pkl")
joblib.dump({"X": mc_X, "y": mc_y, "trajectories": mc_traj,
             "P_combined_history": mc_P_hist, "means_history": mc_means_hist},
            "monte_carlo_bundle_32.pkl")

# === Print summary stats ===
def print_summary(name, means, covs):
    for label, mean, cov in zip(["Start", "End"], [means[0], means[-1]], [covs[0], covs[-1]]):
        print(f"\n--- {name} {label} ---")
        print("Mean:\n", np.array2string(mean, formatter={'float_kind': lambda x: f'{x: .8f}'}))
        print("Covariance:\n", np.array2string(cov, formatter={'float_kind': lambda x: f'{x: .8f}'}))

print_summary("Sigma Point", means_hist[0], P_hist[0][0])
print_summary("Monte Carlo", mc_means_hist[0], mc_P_hist[0][0])

# === Save expected CSV ===
expected_data = []
backTspan_rev = backTspan[::-1]
tstart, tend = backTspan_rev[tstart_index], backTspan_rev[tend_index]
for sigma_idx in range(len(trajectories[0][0])):
    full_trajectory = np.concatenate([segment[sigma_idx] for segment in trajectories[0]], axis=0)
    times = np.linspace(tstart, tend, full_trajectory.shape[0])
    for t_idx, row in enumerate(full_trajectory):
        expected_data.append([
            bundle_index, sigma_idx, *row[:7], times[t_idx]
        ])

df = pd.DataFrame(expected_data, columns=[
    "bundle", "sigma", "x", "y", "z", "vx", "vy", "vz", "mass", "time"
])
df.to_csv("expected_trajectories_bundle_32.csv", index=False)

# === Diagnostic printout at t_k and t_{k+1} ===
unique_times = np.unique(np.round(X[:, 0], decimals=10))
t_k = unique_times[1]
t_k1 = unique_times[-1]

X_tk = X[np.isclose(X[:, 0], t_k) & (X[:, -2] == 32)]
X_tk1 = X[np.isclose(X[:, 0], t_k1) & (X[:, -2] == 32)]

print(f"\n=== Sigma Points at t = {t_k:.6f} (Second Step) ===")
for sigma_idx in range(15):
    row = X_tk[X_tk[:, -1] == sigma_idx]
    if len(row) == 1:
        p, f, g, h, k, L = row[0, 1:7]
        r_eci, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                          np.array([h]), np.array([k]), np.array([L]), mu)
        print(f"Sigma {sigma_idx:2d}: Cartesian = {r_eci[0]}, MEE = {row[0, 1:4]}")
    else:
        print(f"Sigma {sigma_idx:2d}: [not found or multiple entries]")

sigma0_candidates = X_tk1[(X_tk1[:, -1] == 0)]
appended_row = next((row for row in sigma0_candidates if np.allclose(row[8:15], 0.0)), None)
if appended_row is None:
    raise ValueError("Appended sigma 0 not found")

p, f, g, h, k, L = appended_row[1:7]
r_ref, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                  np.array([h]), np.array([k]), np.array([L]), mu)
r_ref = r_ref[0]

print(f"\n=== Sigma Points at t = {t_k1:.6f} (Final Step) ===")
for sigma_idx in range(15):
    rows = X_tk1[X_tk1[:, -1] == sigma_idx]
    for row in rows:
        p, f, g, h, k, L = row[1:7]
        r_eci, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                          np.array([h]), np.array([k]), np.array([L]), mu)
        r_cart = r_eci[0]
        tag = "(Appended Reference)" if np.allclose(row[8:15], 0.0) else "(Propagated)"
        print(f"Sigma {sigma_idx:2d} {tag:>20}: Cartesian = {r_cart}, MEE = {row[1:4]}")

# === Mahalanobis distances ===
P_final = P_hist[0][0][-1][:3, :3]
inv_cov = inv(P_final)
print(f"\n=== Mahalanobis Distances to Appended Sigma 0 at t = {t_k1:.6f} ===")
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
        print(f"Sigma {sigma_idx:2d}: [propagated version not found]")

# === Plot: Sigma Point & Monte Carlo Trajectories with 3σ Ellipsoids ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
bundle_trajectories = trajectories[0]
P_sigma = P_hist[0][0]

for sigma_idx in range(len(bundle_trajectories[0])):
    full_traj = np.concatenate([segment[sigma_idx] for segment in bundle_trajectories], axis=0)
    r = full_traj[:, :3]
    if sigma_idx == 0:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', linewidth=2, label='Sigma Point Sample Trajectory')
        plot_3sigma_ellipsoid(ax, r[0], P_sigma[0][:3, :3], color='orange', alpha=0.2)
        plot_3sigma_ellipsoid(ax, r[-1], P_sigma[-1][:3, :3], color='orange', alpha=0.2)
    else:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', alpha=0.25)
    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='green', s=8)
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='darkred', s=8)

for sample_idx in range(len(mc_traj[0][0])):
    full_mc = np.concatenate([segment[sample_idx] for segment in mc_traj[0]], axis=0)
    r = full_mc[:, :3]
    if sample_idx == 0:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='blue', linewidth=1.0, label='MC Sample Trajectory')
    else:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='blue', alpha=0.05)
    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='cyan', s=5)
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='navy', s=5)

ellipsoid_patch = Patch(facecolor='orange', edgecolor='orange', alpha=0.2, label='Sigma Point 3σ Ellipsoid')
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.view_init(elev=25, azim=135)
ax.set_box_aspect([1.25, 1, 0.75])
handles, labels = ax.get_legend_handles_labels()
handles.append(ellipsoid_patch)
labels.append('Sigma Point 3σ Ellipsoid')
ax.legend(handles, labels, loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
