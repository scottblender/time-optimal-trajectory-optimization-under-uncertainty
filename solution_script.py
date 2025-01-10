import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
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
nsd = 3  # Dimensionality of the state (3D position and velocity)
beta = 2.  # UKF parameter
kappa = float(3 - nsd)  # UKF parameter
alpha = 0.5  # UKF parameter
lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter

# Create weights for the sigma points using the MerweScaledSigmaPoints class
weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)

# Define the initial covariance matrix for position (identity matrix for position)
P_r = np.eye(nsd)*0.1

# Define the initial covariance matrix for velocity (identity matrix scaled by 0.1 for velocity)
P_v = np.eye(nsd) * 0.001  # Covariance matrix for velocity (0.1 km/s)

# Define the time steps for which sigma points will be generated
time_steps = np.linspace(0, 999, num=3, dtype=int)
num_points = time_steps.shape[0]

# Create placeholders for storing the sigma points (7 for each bundle)
sigmas_r = np.zeros((num_bundles, 2*nsd+1, nsd, num_points))  # For position
sigmas_v = np.zeros((num_bundles, 2*nsd+1, nsd, num_points))  # For velocity

# Compute the Cholesky decomposition of the scaled covariance matrices
U_r = scipy.linalg.cholesky((nsd + lambda_) * P_r)  # For position
U_v = scipy.linalg.cholesky((nsd + lambda_) * P_v)  # For velocity

# Loop over each bundle to generate sigma points for each bundle and time step
for i in range(num_bundles):
    for j in range(num_points):
        # Get the nominal position state (center sigma point) for the current time step
        nominal_r = r_bundles[time_steps[j], :, i]
        # Set the center sigma point (nominal position state)
        sigmas_r[i, 0, :, j] = nominal_r
        
        # Generate sigma points in the positive and negative directions for each dimension (x, y, z)
        for k in range(nsd):
            sigmas_r[i, 2*k+1, :, j] = nominal_r + U_r[k, :]  # Perturb positive
            sigmas_r[i, 2*k+2, :, j] = nominal_r - U_r[k, :]  # Perturb negative

        # Get the nominal velocity state (center sigma point) for the current time step
        nominal_v = v_bundles[time_steps[j], :, i]
        # Set the center sigma point (nominal velocity state)
        sigmas_v[i, 0, :, j] = nominal_v
        
        # Generate sigma points in the positive and negative directions for each velocity dimension (vx, vy, vz)
        for k in range(nsd):
            sigmas_v[i, 2*k+1, :, j] = nominal_v + U_v[k, :]  # Perturb positive
            sigmas_v[i, 2*k+2, :, j] = nominal_v - U_v[k, :]  # Perturb negative

# Output the shape of the generated sigma points for both position and velocity
print("Sigma Points for Position:", sigmas_r.shape)
print("Sigma Points for Velocity:", sigmas_v.shape)

# Convert the sigma points for the first bundle and first time step into a pandas DataFrame for position
df_r = pd.DataFrame(sigmas_r[0, :, :, 0].reshape(7, 3))
print("Position Sigma Points (first bundle, first time step):")
print(df_r.head(7))

# Convert the sigma points for the first bundle and first time step into a pandas DataFrame for velocity
df_v = pd.DataFrame(sigmas_v[0, :, :, 0].reshape(7, 3))
print("Velocity Sigma Points (first bundle, first time step):")
print(df_v.head(7))

# Plotting sigma points for position
sigmas_r_df = pd.DataFrame(sigmas_r[0, :, :, 0].reshape(7, 3))
nominal_r = sigmas_r_df.iloc[0].values
sigma_points_r = sigmas_r_df.values

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sigma_points_r[:, 0], sigma_points_r[:, 1], sigma_points_r[:, 2], color='b', label='Position Sigma Points')
ax.scatter(nominal_r[0], nominal_r[1], nominal_r[2], color='r', label='Nominal Position', s=100)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Position Sigma Points in 3D Space')
ax.legend()
plt.show()

# Plotting sigma points for velocity
sigmas_v_df = pd.DataFrame(sigmas_v[0, :, :, 0].reshape(7, 3))
nominal_v = sigmas_v_df.iloc[0].values
sigma_points_v = sigmas_v_df.values

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sigma_points_v[:, 0], sigma_points_v[:, 1], sigma_points_v[:, 2], color='b', label='Velocity Sigma Points')
ax.scatter(nominal_v[0], nominal_v[1], nominal_v[2], color='r', label='Nominal Velocity', s=100)
ax.set_xlabel('X Velocity')
ax.set_ylabel('Y Velocity')
ax.set_zlabel('Z Velocity')
ax.set_title('Velocity Sigma Points in 3D Space')
ax.legend()
plt.show()

# Assuming the relevant variables (p_sol, tfound, s0, mu, F, c, m0, g0, new_lam_bundles) are already defined
num_time_steps = len(time_steps)  # or specify manually
time = [backTspan[time_steps[2]], backTspan[time_steps[1]], backTspan[time_steps[0]]]

# Loop over each bundle and each time step to solve the IVP
for i in range(1):
    # Extract the lamperture values from new_lam_bundles for the current bundle i
    new_lam = new_lam_bundles[:, i]  # 7 elements for lamperture

    for j in range(1):
        # Define the start and end times for the integration
        tstart = time[j]
        tend = time[j + 1]

        # Extract the current sigma points for position and velocity
        sigma_r = sigmas_r[i, :, :, j]
        sigma_v = sigmas_v[i, :, :, j]

        # Loop over the sigma points
        for sigma_idx in range(sigma_r.shape[0]):
            r0 = sigma_r[sigma_idx, :]
            v0 = sigma_v[sigma_idx, :]

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
                if not Sf.success:
                    continue
                r_new, v_new = mee2rv.mee2rv(Sf.y[0, :], Sf.y[1, :], Sf.y[2, :], Sf.y[3, :], Sf.y[4, :], Sf.y[5, :], mu)
                print(r_new)

                # Plot the new trajectory from the perturbed state
                plt.plot(r_new[:,0], r_new[:,1], label=f"Sigma Point {sigma_idx}")

            except Exception as e:
                continue

# Add labels, title, and grid to the plot for better clarity
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Trajectories in the X-Y Plane")
plt.legend()
plt.grid()
plt.show()
