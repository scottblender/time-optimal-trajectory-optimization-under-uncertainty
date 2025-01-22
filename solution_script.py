import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from filterpy.kalman import MerweScaledSigmaPoints
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import evaluate_bundle_widths
import odefunc
import rv2mee
import mee2rv

# Gravitational parameter for the Sun
mu_s = 132712 * 10**6 * 1e9

# Compute the nominal trajectory parameters
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

# Number of bundles to generate
num_bundles = 100

# Compute bundle trajectory parameters
r_tr, v_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, backTspan = compute_bundle_trajectory_params.compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles)

# Reverse the trajectory to adjust the order of bundles
r_bundles = r_bundles[::-1, :, :]
v_bundles = v_bundles[::-1, :, :]
new_lam_bundles = new_lam_bundles[::-1,:]

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

# Create a 3D plot to visualize the nominal and perturbed position trajectories for all bundles
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')

# Plot the nominal trajectory with a larger line width
ax.plot(x_r, y_r, z_r, label='Nominal Position Trajectory', color='r', linewidth=3)

# Loop through all bundles and plot their respective trajectories with more transparency
for i in range(num_bundles):
    x_r_bundle = r_bundles[:, 0, i]
    y_r_bundle = r_bundles[:, 1, i]
    z_r_bundle = r_bundles[:, 2, i]
    ax.plot(x_r_bundle, y_r_bundle, z_r_bundle, color='b', alpha=0.2)  # Change alpha for more transparency

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title(f'3D Position Trajectory Plot for {num_bundles} Bundles')
plt.axis('equal')
ax.legend()
plt.show()

# Parameters for the sigma point generation process
nsd = 6  # Dimensionality of the state (3D position and 3D velocity combined)
beta = 2.  # UKF parameter
kappa = float(3 - nsd)  # UKF parameter
alpha = 0.5  # UKF parameter
lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter

# Create weights for the sigma points using the MerweScaledSigmaPoints class
weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)

# Define the initial covariance matrix for position and velocity (combined state)
P_combined = np.block([
    [np.eye(nsd//2) * 0.01, np.zeros((nsd//2, nsd//2))],  # Position covariance with zero velocity covariance
    [np.zeros((nsd//2, nsd//2)), np.eye(nsd//2) * 0.001]  # Velocity covariance
])

# Print the matrix
print("Covariance Matrix (P_combined):")
print(P_combined)

# Define the time steps for which sigma points will be generated
num_time_steps = 5
time_steps = np.linspace(0, len(backTspan) - 1, num_time_steps, dtype=int)
num_points = time_steps.shape[0]

# Create placeholders for storing the sigma points (7 for each bundle)
sigmas_combined = np.zeros((num_bundles, 2 * nsd + 1, nsd, num_points))  # For position and velocity combined

# Compute the Cholesky decomposition of the scaled covariance matrix for the combined state
U_combined = scipy.linalg.cholesky((nsd + lambda_) * P_combined)  # For combined state

# Loop over each bundle to generate sigma points for the combined state and time step
for i in range(num_bundles):
    for j in range(num_points):
        # Get the nominal combined state (position and velocity) for the current time step
        nominal_combined = np.concatenate([r_bundles[time_steps[j], :, i], v_bundles[time_steps[j], :, i]])
        
        # Set the center sigma point (nominal combined state)
        sigmas_combined[i, 0, :, j] = nominal_combined
        
        # Generate sigma points in the positive and negative directions for each dimension (x, y, z, vx, vy, vz)
        for k in range(nsd):
            sigmas_combined[i, k+1, :, j] = nominal_combined + U_combined[k]  # Perturb positive
            sigmas_combined[i, nsd+k+1, :, j] = nominal_combined - U_combined[k]  # Perturb negative

# Example: Print the sigma points for the first bundle (i = 0) at the first time step (j = 0)
bundle_idx = 0
time_step_idx = 0

print(f"Sigma points for Bundle {bundle_idx+1}, Time Step {time_step_idx+1}:")
print(sigmas_combined[bundle_idx, :, :, time_step_idx])
# Output the shape of the generated sigma points for the combined state
print("Sigma Points for Combined State (Position + Velocity):", sigmas_combined.shape)

# Convert the sigma points for the first bundle and first time step into a pandas DataFrame for position and velocity
df_combined = pd.DataFrame(sigmas_combined[0, :, :, 0].reshape(13, 6))  # 13 sigma points, 6 dimensions (3 position + 3 velocity)
print("Combined Sigma Points (first bundle, first time step):")
print(df_combined.head(7))

# Plotting sigma points for the combined state (position + velocity)
fig = plt.figure(figsize=(12, 8))

# Plot position part
ax1 = fig.add_subplot(121, projection='3d')
sigma_points_position = df_combined.iloc[:, :3].values
nominal_position = sigma_points_position[0]
ax1.scatter(sigma_points_position[:, 0], sigma_points_position[:, 1], sigma_points_position[:, 2], color='b', label='Position Sigma Points')
ax1.scatter(nominal_position[0], nominal_position[1], nominal_position[2], color='r', label='Nominal Position', s=100)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
ax1.set_title('Position Sigma Points in 3D Space')
ax1.legend()

# Plot velocity part
ax2 = fig.add_subplot(122, projection='3d')
sigma_points_velocity = df_combined.iloc[:, 3:].values
nominal_velocity = sigma_points_velocity[0]
ax2.scatter(sigma_points_velocity[:, 0], sigma_points_velocity[:, 1], sigma_points_velocity[:, 2], color='b', label='Velocity Sigma Points')
ax2.scatter(nominal_velocity[0], nominal_velocity[1], nominal_velocity[2], color='r', label='Nominal Velocity', s=100)
ax2.set_xlabel('X Velocity')
ax2.set_ylabel('Y Velocity')
ax2.set_zlabel('Z Velocity')
ax2.set_title('Velocity Sigma Points in 3D Space')
ax2.legend()

plt.tight_layout()
plt.show()

# Now select times based on time_steps
time = [backTspan[time_steps[i]] for i in range(num_time_steps)]
time = time[::-1]

# Initialize an empty list to store trajectories for all bundles
trajectories = []

# Loop over each bundle and each time step to solve the IVP
for i in range(num_bundles):  # Loop over each bundle (adjust based on your needs)
    # Extract the lamperture values from new_lam_bundles for the current bundle i
    new_lam = new_lam_bundles[:, i]  # 7 elements for lamperture

    # List to store the sigma points trajectories for the current bundle
    bundle_trajectories = []

    for j in range(num_time_steps - 1):  # Loop over time steps (ensure range is valid)
        # Define the start and end times for the integration
        tstart = time[j]
        tend = time[j + 1]

        # Extract the current sigma points for position and velocity (combined state)
        sigma_combined = sigmas_combined[i, :, :, j]  # Shape: (13, 6)

        # List to store the trajectories for all sigma points at the current time step
        sigma_point_trajectories = []

        # Loop over the sigma points
        for sigma_idx in range(sigma_combined.shape[0]):  # Loop through the 13 sigma points
            # Extract position and velocity from the combined sigma point
            r0 = sigma_combined[sigma_idx, :3]  # First 3 elements are position
            v0 = sigma_combined[sigma_idx, 3:]  # Last 3 elements are velocity

            # Convert the position and velocity to modified equinoctial elements
            initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
            initial_state = np.append(initial_state, np.array([1]), axis=0)  # Append the 1 element for perturbation
            S = np.append(initial_state, new_lam)  # Append the lamperture values

            # Define the ODE function for integration
            func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

            # Define the time span for the ODE solver
            tspan = np.linspace(tstart, tend, 1000)

            try:
                # Solve the ODE using RK45 method
                Sf = scipy.integrate.solve_ivp(func, [tstart, tend], S, method='RK45', rtol=1e-3, atol=1e-6, t_eval=tspan)

                # If the solution is successful, convert the MEE back to RV
                if Sf.success:
                    r_new, v_new = mee2rv.mee2rv(Sf.y[0, :], Sf.y[1, :], Sf.y[2, :], Sf.y[3, :], Sf.y[4, :], Sf.y[5, :], mu)

                    # Store both position and velocity for the current sigma point at each time step
                    trajectory = np.hstack((r_new, v_new))  # Combine position (r_new) and velocity (v_new)
                    sigma_point_trajectories.append(trajectory)

            except Exception as e:
                continue  # In case of error, continue with the next sigma point

        # Append the trajectories of all sigma points at this time step to the bundle
        bundle_trajectories.append(sigma_point_trajectories)

    # Append the trajectories for the entire bundle
    trajectories.append(bundle_trajectories)

# Convert trajectories to a numpy array
trajectories = np.array(trajectories)

# Ensure that the shape of the trajectories is (4, 2, 13, 1000, 6)
print("Shape of trajectories:", trajectories.shape)  # Should be (4, 2, 13, 1000, 6)

# Ensure that the random indices do not exceed the bounds
random_indices = random.sample(range(trajectories.shape[0]), 4)

# Create a subplot figure (2x2 grid for 4 random original trajectories)
fig = plt.figure(figsize=(12, 12))

# Loop through the randomly selected indices
for idx in random_indices:
    ax = fig.add_subplot(2, 2, random_indices.index(idx) + 1, projection='3d')
    ax.set_title(f"Original Trajectory {idx} - Uncertainty Propagation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    # Loop through all sigma point trajectories for the selected original trajectory
    for sigma_idx in range(trajectories.shape[2]):  # Loop through the 13 sigma points
        # Extract the sub-trajectory for the current sigma point from time step t_k to t_{k+1}
        r_new_and_v_new = trajectories[idx, 0, sigma_idx, :, :]  # (1000, 6) for the 1st time interval

        # Extract position and velocity separately
        r_new = r_new_and_v_new[:, :3]  # Position (X, Y, Z)
        v_new = r_new_and_v_new[:, 3:]  # Velocity (Vx, Vy, Vz)

        # Plot the position trajectory in the current subplot (3D space)
        if sigma_idx == 0:
            # Thicker line and label for sigma point 0 (Initial State)
            ax.plot(r_new[:, 0], r_new[:, 1], r_new[:, 2], label="Initial State", color='b', linewidth=2)
        else:
            # Thinner line and lower transparency for all other sigma points
            ax.plot(r_new[:, 0], r_new[:, 1], r_new[:, 2], color='b', alpha=0.3, linewidth=1, label=f"Sigma Point {sigma_idx}")

        # Add markers for the start and end points of the trajectory (using X markers)
        start_pos = r_new[0, :]  # Start position
        end_pos = r_new[-1, :]   # End position
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='g', marker='x', s=100, label='Start Point' if sigma_idx == 0 else "")
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='r', marker='x', s=100, label='End Point' if sigma_idx == 0 else "")

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid()

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.show()

# Choose a random bundle (trajectory set) from the available ones (assuming 4 bundles here)
random_bundle_idx = random.randint(0, 3)  # Choose a random index between 0 and 3

# Choose a random bundle (trajectory set) from the available ones (assuming 4 bundles here)
random_bundle_idx = random.randint(0, 3)  # Choose a random index between 0 and 3

# Extract the selected random trajectory for the chosen bundle
# Assume that for each bundle, there are multiple time steps and sigma points
random_trajectory = trajectories[random_bundle_idx]  # Shape: (time_steps, sigma_points, trajectory_length, 6)

# Loop through each time step and sigma point to print the start and end positions and velocities
num_time_steps = random_trajectory.shape[0]
num_sigma_points = random_trajectory.shape[1]

# Loop through each time step and sigma point to print the start and end positions and velocities
for t_idx in range(num_time_steps):
    print(f"Time Step {t_idx}:")
    for sigma_idx in range(num_sigma_points):
        # Extract position and velocity for the current sigma point
        r_new = random_trajectory[t_idx, sigma_idx, :, :3]  # Position (X, Y, Z)
        v_new = random_trajectory[t_idx, sigma_idx, :, 3:]  # Velocity (Vx, Vy, Vz)
        
        # Start and end positions and velocities
        start_pos = r_new[0, :]  # Start position
        end_pos = r_new[-1, :]   # End position
        start_vel = v_new[0, :]  # Start velocity
        end_vel = v_new[-1, :]   # End velocity

        # Print the info for the current sigma point at the current time step
        print(f"  Sigma Point {sigma_idx}:")
        print(f"    Start Position: {start_pos}")
        print(f"    End Position: {end_pos}")
        print(f"    Start Velocity: {start_vel}")
        print(f"    End Velocity: {end_vel}")

# Create a subplot for each time step (3D subplots for each time step)
fig, axes = plt.subplots(1, num_time_steps, figsize=(15, 5), subplot_kw={'projection': '3d'})

# If there's only one time step, axes will not be a list, so we make it iterable
if num_time_steps == 1:
    axes = [axes]

# Loop through each time step and plot
for t_idx, ax in enumerate(axes):
    ax.set_title(f"Time Step {t_idx}")
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Plot the original trajectory (sigma point 0 for that time step)
    original_trajectory = random_trajectory[t_idx, 0, :, :]  # First sigma point is the original trajectory
    r_new = original_trajectory[:, :3]  # Position (X, Y, Z)
    v_new = original_trajectory[:, 3:]  # Velocity (Vx, Vy, Vz)
    ax.plot(r_new[:, 0], r_new[:, 1], r_new[:, 2], 
            label="Original Trajectory", color='b', linewidth=2)

    # Add markers for the start and end points of the original trajectory (using X markers)
    start_pos = r_new[0, :]  # Start position
    end_pos = r_new[-1, :]   # End position
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='g', marker='x', s=100, label='Start Point')
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='r', marker='x', s=100, label='End Point')

    # Loop through each sigma point (perturbed trajectories) and plot
    for sigma_idx in range(1, num_sigma_points):  # Start from 1 to skip the original
        perturbed_trajectory = random_trajectory[t_idx, sigma_idx, :, :]
        r_perturbed = perturbed_trajectory[:, :3]  # Position (X, Y, Z)
        ax.plot(r_perturbed[:, 0], r_perturbed[:, 1], r_perturbed[:, 2], 
                label=f"Sigma Point {sigma_idx}", color='r', alpha=0.4)

        # Add markers for the start and end points of the perturbed trajectory (using X markers)
        start_pos_perturbed = r_perturbed[0, :]  # Start position for perturbed trajectory
        end_pos_perturbed = r_perturbed[-1, :]   # End position for perturbed trajectory
        ax.scatter(start_pos_perturbed[0], start_pos_perturbed[1], start_pos_perturbed[2], 
                   color='g', marker='x', s=100, alpha=0.5)
        ax.scatter(end_pos_perturbed[0], end_pos_perturbed[1], end_pos_perturbed[2], 
                   color='r', marker='x', s=100, alpha=0.5)

    # Add grid and legend
    ax.grid(True)

# Adjust layout and add legend outside the plot for clarity
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

