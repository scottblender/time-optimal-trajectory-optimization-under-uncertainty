import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import evaluate_bundle_widths

# Compute the nominal trajectory parameters
p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

# Number of bundles to generate
num_bundles = 100

# Compute bundle trajectory parameters
r_tr, v_tr, S_bundles, r_bundles, v_bundles = compute_bundle_trajectory_params.compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles)

# Reverse the trajectory to get nominal trajectory
r_nom = r_tr[::-1]

# Bundle counts and max widths (for further processing if needed)
bundle_counts = np.array([num_bundles])
max_widths = evaluate_bundle_widths.evaluate_bundle_widths(bundle_counts, r_nom, r_bundles, tfound)

# Extract the x, y, z coordinates of the nominal trajectory
x = r_nom[:,0]
y = r_nom[:,1]
z = r_nom[:,2]

# Extract the x, y, z coordinates of the perturbed trajectory (for bundle 1)
x_bundle = r_bundles[:,0, 1]
y_bundle = r_bundles[:,1, 1]
z_bundle = r_bundles[:, 2,1]

# Create a 3D plot
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')

# Plot the nominal and perturbed trajectories
ax.plot(x, y, z, label='Nominal Trajectory', color='r')
ax.plot(x_bundle, y_bundle, z_bundle, label='Perturbed Trajectory', color='b')

# Set axis labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory Plot')

# Add a legend
ax.legend()

# Display the plot
plt.show()

# Re-import necessary modules for generating sigma points
import numpy as np
import scipy.linalg

# Constants for the sigma point generation process
nsd = 3  # Dimensionality of the state (3 for 3D position)
beta = 2.  # UKF parameter
kappa = float(3 - nsd)  # UKF parameter
alpha = 0.5  # UKF parameter
lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter

# Create weights for the sigma points
weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)

# Define the initial covariance matrix (identity matrix for 1 km uncertainty)
P = np.eye(nsd)

# Define the time steps for which sigma points are generated (3 time steps)
time_steps = np.linspace(0, 999, num=3, dtype=int)
num_points = time_steps.shape[0]  # Number of time steps
sigmas = np.zeros((num_points*(2 * nsd + 1), nsd, num_bundles))  # Shape for storing sigma points

# Compute the Cholesky decomposition of the scaled covariance matrix
U = scipy.linalg.cholesky((nsd + lambda_) * P)
print(U)

# Loop over each bundle
for i in range(num_bundles):
    # Loop over each time step (3 time steps in total)
    for j in range(num_points):
        # Set the nominal state (center sigma point) for the current time step
        sigmas[(2*nsd+1)*j, :, i] = r_bundles[j, :, i]
        
        # Generate sigma points for each dimension (x, y, z)
        for k in range(nsd):
            # Sigma points in the positive direction (adding the Cholesky matrix factor)
            sigmas[((2*nsd+1)*j)+(k+1), :, i] = r_bundles[time_steps[j], :, i] + U[k, :]
            
            # Sigma points in the negative direction (subtracting the Cholesky matrix factor)
            sigmas[((2*nsd+1)*j)+(nsd+k+1), :, i] = r_bundles[time_steps[j], :, i] - U[k, :]

# Output the shape of the generated sigma points
print(sigmas.shape)

# Convert the sigma points for the first bundle into a pandas DataFrame for easier viewing
df = pd.DataFrame(sigmas[:,:,1])
print(df.head(21))
