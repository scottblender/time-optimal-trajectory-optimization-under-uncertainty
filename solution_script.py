import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import generate_sigma_points
import evaluate_bundle_widths
import solve_trajectories

# Gravitational parameter for the Sun
mu_s = 132712 * 10**6 * 1e9

# Compute the nominal trajectory parameters
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU,TU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

# Number of bundles to generate
num_bundles = 100

# Compute bundle trajectory parameters
r_tr, v_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, backTspan = compute_bundle_trajectory_params.compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles)

# Reverse the trajectory to adjust the order of bundles
r_bundles = r_bundles[::-1, :, :]
v_bundles = v_bundles[::-1, :, :]
#new_lam_bundles = new_lam_bundles[::-1,:, :]
# Print out new_lam_bundles nicely
num_steps, lam_size, num_bundles = new_lam_bundles.shape

# Loop over the bundles and time steps to print out new_lam_bundles
#for t in range(num_steps):
#    print(f"\nTime Step {t+1} (backTspan time {backTspan[t]}):")
#    
#    for b in range(num_bundles):
#        print(f"  Bundle {b+1}: {new_lam_bundles[t, :, b]}")

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


# Define the initial covariance matrix for position and velocity (combined state)
nsd = 6
P_pos = [np.eye(nsd//2) * 0.01, np.zeros((nsd//2, nsd//2))]
P_vel =  [np.zeros((nsd//2, nsd//2)), np.eye(nsd//2) * 0.0001]


# Run external function to generate sigma points
sigmas_combined, P_combined, time_steps, num_time_steps_ = generate_sigma_points.generate_sigma_points(nsd=nsd, alpha=1.7215, beta=2., kappa=float(3-nsd), P_pos = P_pos, P_vel = P_vel, num_time_steps=1000, backTspan=backTspan, r_bundles=r_bundles, v_bundles=v_bundles)

# Print the matrix
print("Covariance Matrix (P_combined):")
for row in P_combined:
    print("  ".join(f"{val: .8f}" for val in row))  # 8 decimal places for more precisi

# Example: Print the sigma points for the first bundle (i = 0) at the first time step (j = 0)
bundle_idx = 0
time_step_idx = 0

print(f"Sigma points for Bundle {bundle_idx+1}, Time Step {time_step_idx+1}:")
# Format the sigma points output nicely
for row in sigmas_combined[bundle_idx, :, :, time_step_idx].reshape((2 * nsd + 1, -1)):
    print("  ".join(f"{val: .8f}" for val in row))  # 8 decimal places
# Output the shape of the generated sigma points for the combined state
print("Sigma Points for Combined State (Position + Velocity):", sigmas_combined.shape)

# Convert the sigma points for the first bundle and first time step into a pandas DataFrame for position and velocity
df_combined = pd.DataFrame(sigmas_combined[0, :, :, 0].reshape(13, 6))  # 13 sigma points, 6 dimensions (3 position + 3 velocity)

# Plotting sigma points for the combined state (position + velocity)
fig = plt.figure(figsize=(12, 8))

# Plot position part
ax1 = fig.add_subplot(121, projection='3d')
sigma_points_position = df_combined.iloc[:, :3].values
nominal_position = sigma_points_position[0]
ax1.scatter(sigma_points_position[:, 0], sigma_points_position[:, 1], sigma_points_position[:, 2], color='b', label='Position Sigma Points')
ax1.scatter(nominal_position[0], nominal_position[1], nominal_position[2], color='r', label='Nominal Position', s=100)

# Plot 3-sigma ellipsoid around the nominal position
covariance_position = P_combined[:3, :3]  # Extract the position part of the covariance matrix
eigenvalues_position, eigenvectors_position = np.linalg.eigh(covariance_position)  # Eigen decomposition
radii_position = 3 * np.sqrt(eigenvalues_position)  # 3-sigma scaling

# Create a grid of points in 3D spherical coordinates
phi, theta = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = radii_position[0] * np.sin(theta) * np.cos(phi)
y = radii_position[1] * np.sin(theta) * np.sin(phi)
z = radii_position[2] * np.cos(theta)

# Reshape and transform into 3D space
points_position = np.vstack([x.ravel(), y.ravel(), z.ravel()])
ellipsoid_position = eigenvectors_position @ points_position + nominal_position[:, np.newaxis]

# Plot the ellipsoid
ax1.plot_wireframe(ellipsoid_position[0, :].reshape(x.shape), 
                   ellipsoid_position[1, :].reshape(y.shape),
                   ellipsoid_position[2, :].reshape(z.shape), color='g', alpha=0.3)

ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
ax1.set_title('Position Sigma Points in 3D Space with 3-Sigma Ellipsoid')
ax1.legend()

# Plot velocity part
ax2 = fig.add_subplot(122, projection='3d')
sigma_points_velocity = df_combined.iloc[:, 3:].values
nominal_velocity = sigma_points_velocity[0]
ax2.scatter(sigma_points_velocity[:, 0], sigma_points_velocity[:, 1], sigma_points_velocity[:, 2], color='b', label='Velocity Sigma Points')
ax2.scatter(nominal_velocity[0], nominal_velocity[1], nominal_velocity[2], color='r', label='Nominal Velocity', s=100)

# Plot 3-sigma ellipsoid around the nominal velocity
covariance_velocity = P_combined[3:, 3:]  # Extract the velocity part of the covariance matrix
eigenvalues_velocity, eigenvectors_velocity = np.linalg.eigh(covariance_velocity)  # Eigen decomposition
radii_velocity = 3 * np.sqrt(eigenvalues_velocity)  # 3-sigma scaling

# Create a grid of points in 3D spherical coordinates
x_vel = radii_velocity[0] * np.sin(theta) * np.cos(phi)
y_vel = radii_velocity[1] * np.sin(theta) * np.sin(phi)
z_vel = radii_velocity[2] * np.cos(theta)

# Reshape and transform into 3D space
points_velocity = np.vstack([x_vel.ravel(), y_vel.ravel(), z_vel.ravel()])
ellipsoid_velocity = eigenvectors_velocity @ points_velocity + nominal_velocity[:, np.newaxis]

# Plot the ellipsoid
ax2.plot_wireframe(ellipsoid_velocity[0, :].reshape(x_vel.shape), 
                   ellipsoid_velocity[1, :].reshape(y_vel.shape),
                   ellipsoid_velocity[2, :].reshape(z_vel.shape), color='g', alpha=0.3)

ax2.set_xlabel('X Velocity')
ax2.set_ylabel('Y Velocity')
ax2.set_zlabel('Z Velocity')
ax2.set_title('Velocity Sigma Points in 3D Space with 3-Sigma Ellipsoid')
ax2.legend()

plt.tight_layout()
plt.show()

# --- Calculate Mahalanobis Distances ---
# Get the position part of the covariance matrix and nominal point
covariance_position = P_combined[:3, :3]  # Position part of covariance
nominal_position = sigmas_combined[0, 0, :, 0][:3]  # First sigma point, nominal position

# Get the perturbed sigma points for position (all sigma points except the center one)
position_sigma_points = sigmas_combined[0, 1:, :, 0][:, :3]  # Remove the nominal point from the set

# Inverse of the covariance matrix
covariance_inv_position = np.linalg.inv(covariance_position)

# Calculate Mahalanobis distances for position sigma points
mahalanobis_distances_position = []
for point in position_sigma_points:
    diff = point - nominal_position
    dist = np.sqrt(diff.T @ covariance_inv_position @ diff)  # Mahalanobis distance
    mahalanobis_distances_position.append(dist)

# Get the velocity part of the covariance matrix and nominal point
covariance_velocity = P_combined[3:, 3:]  # Velocity part of covariance
nominal_velocity = sigmas_combined[0, 0, :, 0][3:]  # First sigma point, nominal velocity

# Get the perturbed sigma points for velocity
velocity_sigma_points = sigmas_combined[0, 1:, :, 0][:, 3:]  # Remove the nominal point for velocity

# Inverse of the covariance matrix for velocity
covariance_inv_velocity = np.linalg.inv(covariance_velocity)

# Calculate Mahalanobis distances for velocity sigma points
mahalanobis_distances_velocity = []
for point in velocity_sigma_points:
    diff = point - nominal_velocity
    dist = np.sqrt(diff.T @ covariance_inv_velocity @ diff)  # Mahalanobis distance
    mahalanobis_distances_velocity.append(dist)

# --- Plotting ---
# Plot the Mahalanobis distances for position sigma points
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(mahalanobis_distances_position, 'bo-', label='Mahalanobis Distance (Position)')
plt.title('Mahalanobis Distances for Position Sigma Points')
plt.xlabel('Sigma Point Index')
plt.ylabel('Mahalanobis Distance')
plt.grid(True)
plt.legend()

# Plot the Mahalanobis distances for velocity sigma points
plt.subplot(122)
plt.plot(mahalanobis_distances_velocity, 'ro-', label='Mahalanobis Distance (Velocity)')
plt.title('Mahalanobis Distances for Velocity Sigma Points')
plt.xlabel('Sigma Point Index')
plt.ylabel('Mahalanobis Distance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# define dummy value for num_time_steps
num_time_steps = 6

# Call external function to solve new IVPs
trajectories, P_combined_history = solve_trajectories.solve_trajectories_with_covariance(
    backTspan=backTspan, 
    time_steps=time_steps, 
    num_time_steps=num_time_steps, # fix later, but 5 time steps computed
    num_bundles=num_bundles, 
    sigmas_combined=sigmas_combined, 
    new_lam_bundles=new_lam_bundles, 
    mu=mu, 
    F=F, 
    c=c, 
    m0=m0, 
    g0=g0,
    P_combined_initial=P_combined
)

# Select the bundle index (e.g., bundle 0)
bundle_index = 0

# Select the time steps you want to plot (this will correspond to the indices of time steps)
# For example, plot for the first 5 time steps
time_indices = range(5)

# Extract the covariance history for the selected bundle
P_combined_bundle = P_combined_history[bundle_index, time_indices, :, :]

# Initialize lists to store the diagonal elements (variances) of the covariance matrix over time
position_variances = []
velocity_variances = []

# Loop over the selected time indices and extract the diagonal elements (variances)
for i in range(len(time_indices)):
    P = P_combined_bundle[i, :, :]

    # Print the covariance matrix P for the current time index in a formatted manner
    print(f"Covariance matrix for Bundle {bundle_index+1}, Time Step {time_indices[i]+1}:")
    for row in P:
        print("  ".join(f"{val: .8f}" for val in row))  # Format each value to 8 decimal places
   
    # Extract the diagonal elements of the covariance matrix (position and velocity variances)
    position_variances.append([P[0, 0], P[1, 1], P[2, 2]])  # Variance of position (x, y, z)
    velocity_variances.append([P[3, 3], P[4, 4], P[5, 5]])  # Variance of velocity (vx, vy, vz)

# Convert lists to numpy arrays for easier plotting
position_variances = np.array(position_variances)
velocity_variances = np.array(velocity_variances)

# Plot the variances over time
plt.figure(figsize=(10, 6))

# Plot the position variances (x, y, z)
plt.subplot(2, 1, 1)
plt.plot(time_indices, position_variances[:, 0], label="Variance in X Position", color='b')
plt.plot(time_indices, position_variances[:, 1], label="Variance in Y Position", color='g')
plt.plot(time_indices, position_variances[:, 2], label="Variance in Z Position", color='r')
plt.xlabel('Time Step')
plt.ylabel('Position Variance')
plt.title(f'Evolution of Position Variance for Bundle {bundle_index}')
plt.legend()

# Plot the velocity variances (vx, vy, vz)
plt.subplot(2, 1, 2)
plt.plot(time_indices, velocity_variances[:, 0], label="Variance in X Velocity", color='b')
plt.plot(time_indices, velocity_variances[:, 1], label="Variance in Y Velocity", color='g')
plt.plot(time_indices, velocity_variances[:, 2], label="Variance in Z Velocity", color='r')
plt.xlabel('Time Step')
plt.ylabel('Velocity Variance')
plt.title(f'Evolution of Velocity Variance for Bundle {bundle_index}')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Print the shape of trajectories
print("Shape of trajectories:", trajectories.shape)  

# # Initialize a list to store the Mahalanobis distances for all bundles and time steps
# mahalanobis_distances = []

# # Loop over the bundles and time steps to calculate the Mahalanobis distance
# for i in range(num_bundles):  # Loop over each bundle
#     for j in range(5):  # Loop over each time step
        
#         # Extract the nominal trajectory (we assume the first sigma point of the first bundle is the nominal)
#         nominal_trajectory = trajectories[i,j,0,:,:]  # First sigma point (unperturbed) for this bundle and time step

#         # Loop over each perturbed trajectory (sigma points)
#         for sigma_idx in range(trajectories.shape[2]):  # Loop through the 13 sigma points at time step j
#             perturbed_trajectory = trajectories[i,j,sigma_idx,:,:]  # Shape: (6,) [position, velocity]

#             # Compute the difference between the perturbed trajectory and the nominal trajectory
#             diff = perturbed_trajectory - nominal_trajectory  # Shape: (6,)
            
#             # Loop over each row
#             for row in range(trajectories.shape[3]):
#                 # Calculate the Mahalanobis distance for this perturbed trajectory
#                 mahalanobis_dist = np.sqrt(np.dot(diff[row,:].T, np.linalg.inv(P_combined).dot(diff[row,:])))
                
#                 # Store the Mahalanobis distance
#                 mahalanobis_distances.append(mahalanobis_dist)

# # Convert the list of Mahalanobis distances to a NumPy array
# mahalanobis_distances = np.array(mahalanobis_distances)

#  Ensure that the random indices do not exceed the bounds
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

        # Compare the current trajectory with all previous trajectories for this index
        for prev_sigma_idx in range(sigma_idx):
            prev_r_new_and_v_new = trajectories[idx, 0, prev_sigma_idx, :, :]
            prev_r_new = prev_r_new_and_v_new[:, :3]
            prev_v_new = prev_r_new_and_v_new[:, 3:]

            # Compare the current position and velocity matrices with the previous ones
            if np.allclose(r_new, prev_r_new, atol=1e-2) and np.allclose(v_new, prev_v_new, atol=1e-2):
                print(f"Trajectory {idx}, Sigma Point {sigma_idx} is the same as Sigma Point {prev_sigma_idx}")
                break

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
random_time_steps = np.random.randint(0,(random_trajectory.shape[0]-1),size=5)
print(random_time_steps)
num_sigma_points = random_trajectory.shape[1]

# Loop through each time step and sigma point to print the start and end positions and velocities
for t_idx in random_time_steps:
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
fig, axes = plt.subplots(1, random_time_steps.shape[0], figsize=(15, 5), subplot_kw={'projection': '3d'})

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

