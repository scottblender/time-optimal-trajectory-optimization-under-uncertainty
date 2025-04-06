import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import generate_sigma_points
import evaluate_bundle_widths
import solve_trajectories
import mc_samples

# Gravitational parameter for the Sun
mu_s = 132712 * 10**6 * 1e9

# Compute the nominal trajectory parameters
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU,TU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

# Number of bundles to generate
num_bundles = 100

# Call function
r_tr, v_tr, mass_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, mass_bundles, backTspan = compute_bundle_trajectory_params.compute_bundle_trajectory_params(
    p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles
)

# Reverse the trajectory to adjust the order of bundles
r_bundles = r_bundles[::-1, :, :]
v_bundles = v_bundles[::-1, :, :]
new_lam_bundles = new_lam_bundles[::-1,:, :]
mass_bundles = mass_bundles[::-1,:]
# Print out new_lam_bundles nicely
num_steps, lam_size, num_bundles = new_lam_bundles.shape

# Get the nominal trajectory for position and velocity
r_nom = r_tr
v_nom = v_tr

# Bundle counts and maximum widths (not used here but prepared for further processing)
bundle_counts = np.array([num_bundles])
max_widths = evaluate_bundle_widths.evaluate_bundle_widths(bundle_counts, r_nom, r_bundles, tfound)

# Extract the x, y, z coordinates of the nominal position trajectory
x_r = r_nom[:,0]
y_r = r_nom[:,1]
z_r = r_nom[:,2]

# Extract the x, y, z coordinates of the perturbed position trajectory for the first bundle
x_r_bundle = r_bundles[:,0, 1]
y_r_bundle = r_bundles[:,1, 1]
z_r_bundle = r_bundles[:, 2,1]

# Define the initial covariance matrix for position and velocity (combined state)
nsd = 7
P_pos = np.eye(3) * 0.01  # 3x3 Position covariance
P_vel = np.eye(3) * 0.0001  # 3x3 Velocity covariance
P_mass = np.array([[0.0001]])  # Mass variance (scalar)

alpha, beta, kappa = 1.7215, 2, float((3-nsd))  # UKF parameters

num_time_steps = 1000

# Call function
sigmas_combined, P_combined, time_steps, num_time_steps, Wm, Wc = generate_sigma_points.generate_sigma_points(
    nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
    P_pos=P_pos, P_vel=P_vel, P_mass=P_mass,
    num_time_steps=num_time_steps, backTspan=backTspan,
    r_bundles=r_bundles, v_bundles=v_bundles, mass_bundles=mass_bundles
)

# Print sigma points for the first bundle (bundle 0) at all time steps
print("Sigma Points for the first bundle (bundle 0) at all time steps:")

# Loop through each time step for the first bundle
for j in range(1):  # sigmas_combined has shape (num_bundles, 2*nsd + 1, nsd, num_points)
    print(f"Time step {j}:")
    
    # Loop through each sigma point (2*nsd + 1 sigma points per time step)
    for i in range(sigmas_combined.shape[1]):
        # Format and print each sigma point (position, velocity, mass)
        print("  ".join(f"{val: .8f}" for val in sigmas_combined[0, i, :, j].flatten()))  # Flattening for easier printing
    print()  # Empty line between time steps

# Print the matrix
print("Covariance Matrix (P_combined):")
for row in P_combined:
    print("  ".join(f"{val: .8f}" for val in row))  # 8 decimal places for more precisi

# Define time_steps as all indices of backTspan
time_steps = np.arange(len(backTspan))  # [0, 1, 2, ..., 999]

# Choose specific indices for tstart and tend
tstart_index = 0  # Start at index 0
tend_index = 1    # End at index 1

# Extract the corresponding times from backTspan
tstart = backTspan[tstart_index]
tend = backTspan[tend_index]

# Extract the correct sub-array from time_steps
selected_time_steps = time_steps[tstart_index:tend_index + 1]  # Ensure inclusive selection
num_time_steps = len(selected_time_steps)

# Solve the trajectory with covariance and get X and y
trajectories, P_combined_history, means_history, X, y, time_match_indices = solve_trajectories.solve_trajectories_with_covariance(
    backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, 
    new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc
)

# Print results (for debugging)
print("Trajectories Shape:", trajectories.shape)  # Should now be (..., 7)
print("P_combined_history Shape:", P_combined_history.shape)
print("Means_history Shape:", means_history.shape)

# Save the data to a .pkl file
joblib.dump({
    'trajectories': trajectories,
    'P_combined_history': P_combined_history,
    'means_history': means_history,
    'X': X,
    'y': y,
    'time_match_indices':time_match_indices
}, 'data.pkl')

# Select the bundle index (e.g., bundle 0)
bundle_index = 0

# Select the time steps you want to plot (this will correspond to the indices of time steps)
time_indices = range(num_time_steps-1)

# Select the integration point index to inspect (e.g., first integration point)
integration_point_idx = 199  

# Extract the covariance history for the selected bundle and integration point
P_combined_bundle = P_combined_history[bundle_index, time_indices, integration_point_idx, :, :]

# Initialize lists to store the diagonal elements (variances) of the covariance matrix over time
position_variances = []
velocity_variances = []

# Loop over the selected time indices and extract the diagonal elements (variances)
for i in range(len(time_indices)):
    P = P_combined_bundle[i]  # Shape: (6, 6)

    # Print the covariance matrix P for the current time index in a formatted manner
    print(f"Covariance matrix for Bundle {bundle_index+1}, Time Step {time_indices[i]+1}, Integration Point {integration_point_idx+1}:")
    for row in P:
        print("  ".join(f"{val:.8f}" for val in row))  # Format each value to 8 decimal places
   
    # Extract the diagonal elements of the covariance matrix (position and velocity variances)
    position_variances.append([P[0, 0], P[1, 1], P[2, 2]])  # Variance of position (x, y, z)
    velocity_variances.append([P[3, 3], P[4, 4], P[5, 5]])  # Variance of velocity (vx, vy, vz)

    # --------------------------
# Plot the end-to-end trajectories for one bundle
# --------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set publication-style fonts and larger figure
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (10, 8),
    'figure.dpi': 150
})

# Plot all sigma point trajectories for one bundle
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

bundle_idx = np.random.randint(0,trajectories.shape[0])
bundle_trajectories = trajectories[bundle_idx]

for sigma_idx in range(len(bundle_trajectories[0])):
    full_trajectory = np.concatenate([segment[sigma_idx] for segment in bundle_trajectories], axis=0)
    r = full_trajectory[:, 0:3]  # Already in Cartesian

    # Plot each trajectory
    if sigma_idx == 0:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', linewidth=2, label='Sigma Point 0')
    else:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='red', alpha=0.25)

    # Add start and end point markers
    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='limegreen', marker='X', s=60, edgecolor='black', label='Start' if sigma_idx == 0 else "")
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='red', marker='X', s=60, edgecolor='black', label='End' if sigma_idx == 0 else "")

# Format the plot
ax.set_title(f"Sigma Point Trajectories for Bundle {bundle_idx}")
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.view_init(elev=30, azim=135)
ax.legend()
plt.tight_layout()
plt.show()