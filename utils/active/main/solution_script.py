import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from propagate_sensitivity import propagate_sensitivity_from_initial_lam_only
from sensitivity_metrics import compute_aggregate_metrics
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import generate_sigma_points
import solve_trajectories
import generate_monte_carlo_trajectories
from scipy.linalg import eigh
from matplotlib.patches import Patch

def create_initial_csv_for_multiple_bundles(
    backTspan, r_bundles, v_bundles, mass_bundles, new_lam_bundles,
    bundle_indices, output_filename="initial_bundles_32_33.csv"
):
    backTspan_reversed = backTspan[::-1]
    combined_data = []

    for bundle_index in bundle_indices:
        bundle_r = r_bundles[::-1, :, bundle_index]
        bundle_v = v_bundles[::-1, :, bundle_index]
        bundle_m = mass_bundles[::-1, bundle_index]
        bundle_lam = new_lam_bundles[::-1, :, bundle_index]

        for t_idx in range(len(backTspan_reversed)):
            row = [
                backTspan_reversed[t_idx],
                *bundle_r[t_idx], *bundle_v[t_idx],
                bundle_m[t_idx],
                *bundle_lam[t_idx],
                bundle_index
            ]
            combined_data.append(row)

    df = pd.DataFrame(combined_data, columns=[
        "time", "x", "y", "z", "vx", "vy", "vz", "mass",
        "lam0", "lam1", "lam2", "lam3", "lam4", "lam5", "lam6",
        "bundle_index"
    ])
    df.to_csv(output_filename, index=False)


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
    
# === Setup and Trajectory Bundle Generation ===
mu_s = 132712 * 10**6 * 1e9
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

num_bundles = 100
r_tr, v_tr, mass_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, mass_bundles, backTspan = compute_bundle_trajectory_params.compute_bundle_trajectory_params(
    p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles
)

create_initial_csv_for_multiple_bundles(
    backTspan=backTspan,
    r_bundles=r_bundles,
    v_bundles=v_bundles,
    mass_bundles=mass_bundles,
    new_lam_bundles=new_lam_bundles,
    bundle_indices=list(range(num_bundles-25)),
    output_filename="initial_bundles_all.csv"
)

# === Plot Bundle Trajectories  ===
# Create a 3D plot to visualize the nominal and perturbed position trajectories
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, z coordinates of the nominal position trajectory
x_r = r_tr[:,0]
y_r = r_tr[:,1]
z_r = r_tr[:,2]

# Plot all perturbed bundle trajectories
for i in range(num_bundles):
    x_r_bundle = r_bundles[::-1, 0, i]
    y_r_bundle = r_bundles[::-1, 1, i]
    z_r_bundle = r_bundles[::-1, 2, i]
    ax.plot(x_r_bundle, y_r_bundle, z_r_bundle, color='blue', alpha=0.15, linewidth=1)

# Plot the nominal trajectory
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
plt.show()

# === Plot Nominal Trajectory Only ===
fig_nominal = plt.figure(figsize=(10, 8))
ax_nominal = fig_nominal.add_subplot(111, projection='3d')

# Plot nominal trajectory
ax_nominal.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='black', linewidth=3, label='Nominal Trajectory')

# Start and end points
ax_nominal.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='green', marker='o', s=30, label='Start')
ax_nominal.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='red', marker='X', s=30, label='End')

# Axis labels
ax_nominal.set_xlabel('X [km]')
ax_nominal.set_ylabel('Y [km]')
ax_nominal.set_zlabel('Z [km]')
ax.view_init(elev=25, azim=135)
ax.set_box_aspect([1.25, 1, 0.75])
ax_nominal.legend(loc='upper left')
plt.tight_layout()
plt.show()

bundle_index = 32
S_bundles = S_bundles[::-1, :, bundle_index:bundle_index+1]
r_bundles = r_bundles[::-1, :, bundle_index:bundle_index+1]
v_bundles = v_bundles[::-1, :, bundle_index:bundle_index+1]
new_lam_bundles = new_lam_bundles[::-1, :, bundle_index:bundle_index+1]
mass_bundles = mass_bundles[::-1, bundle_index:bundle_index+1]

# === Sigma Point Generation for Different Time Strides ===
nsd = 7
P_pos = np.eye(3) * 0.01
P_vel = np.eye(3) * 0.0001
P_mass = np.array([[0.0001]])
alpha, beta, kappa = 1.7215, 2, float((3 - nsd))

# === Generate Data for Different Time Strides  ===
time_strides_to_test = [1, 2, 5, 10]  # Define the time strides you want to evaluate

for stride in time_strides_to_test:
    print(f"\n=== Evaluating Time Stride: {stride} ===")

    # Select strided time steps
    time_steps = np.arange(len(backTspan), step=stride)
    tstart_index, tend_index = 0, 1
    tstart, tend = backTspan[tstart_index], backTspan[tend_index]
    time_steps_strided = time_steps[tstart_index:tend_index + 1]
    num_time_steps = len(time_steps_strided)

    # Generate sigma points
    sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
        nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
        P_pos=P_pos, P_vel=P_vel, P_mass=P_mass,
        num_time_steps=num_time_steps, backTspan=backTspan,
        r_bundles=r_bundles, v_bundles=v_bundles, mass_bundles=mass_bundles
    )

    # Propagate sigma points
    trajectories, P_combined_history, means_history, X, y = solve_trajectories.solve_trajectories_with_covariance(
        backTspan, time_steps_strided, num_time_steps, 1, sigmas_combined,
        new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc
    )

    # Save result
    save_path = f"data_bundle_32_stride_{stride}.pkl"
    joblib.dump({
        "X": X,
        "y": y,
        "trajectories": trajectories,
        "P_combined_history": P_combined_history,
        "means_history": means_history
    }, save_path)

    print(f"Saved propagated sigma trajectories for stride {stride} to {save_path}")

# === Perform Propgation for Bundle 32 with MC Comparison  === 
time_stride = 10
time_steps = np.arange(len(backTspan),step=time_stride)
tstart_index, tend_index = 0, 1
tstart, tend = backTspan[tstart_index], backTspan[tend_index]
time_steps = time_steps[tstart_index:tend_index + 1]
num_time_steps = len(time_steps)

# === Sigma Point Generation  for Baseline Dataset ===
nsd = 7
P_pos = np.eye(3) * 0.01
P_vel = np.eye(3) * 0.0001
P_mass = np.array([[0.0001]])
alpha, beta, kappa = 1.7215, 2, float((3 - nsd))

sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
    nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
    P_pos=P_pos, P_vel=P_vel, P_mass=P_mass,
    num_time_steps=num_time_steps, backTspan=backTspan,
    r_bundles=r_bundles, v_bundles=v_bundles, mass_bundles=mass_bundles
)

# === Sigma Point Propagation ===
trajectories, P_combined_history, means_history, X, y = solve_trajectories.solve_trajectories_with_covariance(
    backTspan, time_steps, num_time_steps, 1, sigmas_combined,
    new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc
)

# === Monte Carlo Propagation ===
num_samples = 1000
mc_trajectories, mc_P_combined_history, mc_means_history, mc_X, mc_y = generate_monte_carlo_trajectories.generate_monte_carlo_trajectories(
    backTspan=backTspan,
    time_steps=time_steps,
    num_time_steps=num_time_steps,
    num_bundles=1,  
    sigmas_combined=sigmas_combined,
    new_lam_bundles=new_lam_bundles,
    mu=mu, F=F, c=c, m0=m0, g0=g0,
    num_samples=1000
)

# === Save Sigma Point Data ===
joblib.dump({
    "X": X,
    "y": y,
    "trajectories": trajectories,
    "P_combined_history": P_combined_history,
    "means_history": means_history
}, "data_bundle_32.pkl")

# === Save Monte Carlo Data ===
joblib.dump({
    "X": mc_X,
    "y": mc_y,
    "trajectories": mc_trajectories,
    "P_combined_history": mc_P_combined_history,
    "means_history": mc_means_history
}, "monte_carlo_bundle_32.pkl")

# === Print Start/End Means and Covariances for Sigma Points ===
mean_array_sigma = means_history[0]
cov_array_sigma_start = P_combined_history[0][0][0]
cov_array_sigma_end = P_combined_history[0][0][-1]

for mean, cov, name in zip(
    [mean_array_sigma[0], mean_array_sigma[-1]],
    [cov_array_sigma_start, cov_array_sigma_end],
    ["Start", "End"]
):
    print(f"\n--- Sigma Point {name} ---")
    print("Mean:")
    print(np.array2string(mean, formatter={'float_kind': lambda x: f'{x: .8f}'}))
    print("Covariance:")
    print(np.array2string(cov, formatter={'float_kind': lambda x: f'{x: .8f}'}))

# === Print Start/End Means and Covariances for Monte Carlo ===
mean_array_mc = mc_means_history[0]
cov_array_mc_start = mc_P_combined_history[0][0][0]
cov_array_mc_end = mc_P_combined_history[0][0][-1]

for mean, cov, name in zip(
    [mean_array_mc[0], mean_array_mc[-1]],
    [cov_array_mc_start, cov_array_mc_end],
    ["Start", "End"]
):
    print(f"\n--- Monte Carlo {name} ---")
    print("Mean:")
    print(np.array2string(mean, formatter={'float_kind': lambda x: f'{x: .8f}'}))
    print("Covariance:")
    print(np.array2string(cov, formatter={'float_kind': lambda x: f'{x: .8f}'}))

# === Plot: Sigma Point & Monte Carlo Trajectories with 3σ Ellipsoids (Only for Sigma Point) ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# --- Sigma Point Trajectories ---
bundle_trajectories = trajectories[0]
P_sigma = P_combined_history[0][0]

for sigma_idx in range(len(bundle_trajectories[0])):
    full_trajectory = np.concatenate([segment[sigma_idx] for segment in bundle_trajectories], axis=0)
    r = full_trajectory[:, 0:3]

    if sigma_idx == 0:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', linewidth=2, label='Sigma Point Sample Trajectory')
        plot_3sigma_ellipsoid(ax, r[0], P_sigma[0][:3, :3], color='orange', alpha=0.2)
        plot_3sigma_ellipsoid(ax, r[-1], P_sigma[-1][:3, :3], color='orange', alpha=0.2)
    else:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', alpha=0.25)

    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='green', marker='o', s=8, alpha=0.7)
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='darkred', marker='o', s=8, alpha=0.7)

# --- Monte Carlo Trajectories ---
bundle_mc_trajectories = mc_trajectories[0]

for sample_idx in range(len(bundle_mc_trajectories[0])):
    full_traj = np.concatenate([segment[sample_idx] for segment in bundle_mc_trajectories], axis=0)
    r = full_traj[:, 0:3]

    if sample_idx == 0:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='blue', linewidth=1.0, label='MC Sample Trajectory')
    else:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='blue', alpha=0.05)

    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='cyan', marker='o', s=5, alpha=0.5)
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='navy', marker='o', s=5, alpha=0.5)

# --- Legend Entries for Ellipsoids ---
ellipsoid_patch = Patch(facecolor='orange', edgecolor='orange', alpha=0.2, label='Sigma Point 3σ Ellipsoid')

# --- Final Formatting ---
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


# === Save Full Propagated Sigma Point Trajectories as CSV ===
backTspan_reversed = backTspan[::-1]
expected_data = []
tstart, tend = backTspan_reversed[tstart_index], backTspan_reversed[tend_index]
for sigma_idx in range(len(bundle_trajectories[0])):
    full_trajectory = np.concatenate([segment[sigma_idx] for segment in bundle_trajectories], axis=0)
    times = np.linspace(tstart, tend, full_trajectory.shape[0])
    for t_idx, row in enumerate(full_trajectory):
        expected_data.append([
            bundle_index,
            sigma_idx,
            *row[:7],  # x, y, z, vx, vy, vz, mass
            times[t_idx]
        ])

expected_df = pd.DataFrame(expected_data, columns=[
    "bundle", "sigma", "x", "y", "z", "vx", "vy", "vz", "mass", "time"
])
expected_df.to_csv("expected_trajectories_bundle_32.csv", index=False)


# === Prepare and Run Sensitivity Analysis ===
# Combine state and control history from solve_trajectories
control_dataset = np.hstack((X, y))

# Generate sensitivity trajectories using only initial lam ±3σ
sensitivity_df = propagate_sensitivity_from_initial_lam_only(
    X_sorted=X,
    y_sorted=y,
    trajectories=trajectories,
    backTspan=backTspan,
    stride=time_stride,
)

# === Plot Sensitivity Trajectories per Sigma ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sigma 0 in red
df = sensitivity_df[sensitivity_df["sigma_idx"] == 0].sort_values("time")
ax.plot(df["x"], df["y"], df["z"], color="red", linewidth=2, label="σ=0")
ax.scatter(df["x"].values[-1], df["y"].values[-1], df["z"].values[-1], color="red", marker="x", s=60)

# Others in blue
for sigma in sorted(sensitivity_df["sigma_idx"].unique()):
    if sigma == 0:
        continue
    df = sensitivity_df[sensitivity_df["sigma_idx"] == sigma].sort_values("time")
    ax.plot(df["x"], df["y"], df["z"], color="blue", alpha=0.3)
    ax.scatter(df["x"].values[-1], df["y"].values[-1], df["z"].values[-1], color="blue", s=20, marker="o")

ax.set_title("3D Trajectories: σ₀ in Red, Others in Blue")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
plt.tight_layout()
plt.show()

# === Compute MSE metrics ===
aggregate_results = compute_aggregate_metrics(sensitivity_df)

# === Print nicely ===
print("\n=== MSE (x/y/z) and Final Position Deviation Compared to Sigma 0 ===")
for _, row in aggregate_results.iterrows():
    lam_type = str(row.get("lam_type", "UNKNOWN")).upper()
    sigma_idx = row.get("sigma_idx", "UNKNOWN")
    x_mse = row.get("x mse", np.nan)
    y_mse = row.get("y mse", np.nan)
    z_mse = row.get("z mse", np.nan)
    final_dev = row.get("final position deviation", np.nan)

    print(f"[{lam_type}] Sigma {sigma_idx}")
    if any(pd.isna(val) for val in [x_mse, y_mse, z_mse, final_dev]):
        print("  Skipped due to missing data.\n")
    else:
        print(f"  x MSE (km²): {x_mse:.6f}")
        print(f"  y MSE (km²): {y_mse:.6f}")
        print(f"  z MSE (km²): {z_mse:.6f}")
        print(f"  Final Position Deviation (km): {final_dev:.6f}\n")