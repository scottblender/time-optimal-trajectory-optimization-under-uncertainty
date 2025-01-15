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

# Create a 3D plot to visualize both the nominal and perturbed position trajectories
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.plot(x_r, y_r, z_r, label='Nominal Position Trajectory', color='r')
ax.plot(x_r_bundle, y_r_bundle, z_r_bundle, label='Perturbed Position Trajectory', color='b')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Position Trajectory Plot')
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
    [np.eye(nsd//2) * 0.1, np.zeros((nsd//2, nsd//2))],  # Position covariance with zero velocity covariance
    [np.zeros((nsd//2, nsd//2)), np.eye(nsd//2) * 0.001]  # Velocity covariance
])

# Define the time steps for which sigma points will be generated
time_steps = np.linspace(0, 999, num=3, dtype=int)
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
            sigmas_combined[i, 2*k+1, :, j] = nominal_combined + U_combined[k, :]  # Perturb positive
            sigmas_combined[i, 2*k+2, :, j] = nominal_combined - U_combined[k, :]  # Perturb negative

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

# Assuming the relevant variables (p_sol, tfound, s0, mu, F, c, m0, g0, new_lam_bundles) are already defined
num_time_steps = len(time_steps)  # or specify manually
time = [backTspan[time_steps[2]], backTspan[time_steps[1]], backTspan[time_steps[0]]]

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
                    print(r_new)

                    # Store the trajectory for the current sigma point
                    sigma_point_trajectories.append(r_new)

            except Exception as e:
                continue  # In case of error, continue with the next sigma point

        # Append the trajectories of all sigma points at this time step to the bundle
        bundle_trajectories.append(sigma_point_trajectories)

    # Append the trajectories for the entire bundle
    trajectories.append(bundle_trajectories)

# Convert trajectories to a numpy array
trajectories = np.array(trajectories)

# Ensure that the shape of the trajectories is (4, 2, 13, 1000, 3)
print("Shape of trajectories:", trajectories.shape)  # Should be (4, 2, 13, 1000, 3)

# Ensure that the random indices do not exceed the bounds
random_indices = random.sample(range(trajectories.shape[0]), 4)

# Create a subplot figure (2x2 grid for 4 random original trajectories)
fig = plt.figure(figsize=(12, 12))

# Loop through the randomly selected indices
for idx in random_indices:
    ax = fig.add_subplot(2, 2, random_indices.index(idx) + 1, projection='3d')
    ax.set_title(f"Original Trajectory {idx} - All Sigma Points")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    # Loop through all sigma point trajectories for the selected original trajectory
    for sigma_idx in range(trajectories.shape[2]):  # Loop through the 13 sigma points
        # Extract the sub-trajectory for the current sigma point from time step t_k to t_{k+1}
        # We're plotting the segment from the first to the second time step (i.e., t_0 to t_1)
        r_new = trajectories[idx, 0, sigma_idx, :, :]  # (1000, 3) for the 1st time interval

        # Plot the sigma point trajectory in the current subplot (3D space)
        if sigma_idx == 0:
            # Thicker line and label for sigma point 0 (Initial State)
            ax.plot(r_new[:, 0], r_new[:, 1], r_new[:, 2], label="Initial State", color='b', linewidth=2)
        else:
            # Thinner line and lower transparency for all other sigma points
            ax.plot(r_new[:, 0], r_new[:, 1], r_new[:, 2], color='b', alpha=0.3, linewidth=1, label=f"Sigma Point {sigma_idx}")

    # Automatically place the legend in the best spot
    ax.legend(loc='best')
    ax.grid()

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.show()
