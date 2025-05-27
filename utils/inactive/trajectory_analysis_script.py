import joblib

#varialbles
num_time_steps = 2

# Load the data
data = joblib.load('data.pkl')
trajectories = data['trajectories']
P_combined_history = data['P_combined_history']
# mc_data = joblib.load('mc_data.pkl')
# mc_trajectories = mc_data['mc_trajectories']
# mc_covariances_history = mc_data['mc_covariances_history']

# Sigma Points
# Select the bundle index (e.g., bundle 0)
bundle_index = 0

# Select the time steps you want to plot (this will correspond to the indices of time steps)
time_indices = range(num_time_steps-1)

# Select the integration point index to inspect (e.g., first integration point)
integration_point_idx = 999 

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

# # Monte Carlo
# # Select the bundle index (e.g., bundle 0)
# bundle_index = 0

# # Select the time steps to print (e.g., first 5 time steps)
# time_indices = range(num_time_steps-1)

# # Select the integration point index to inspect (e.g., first integration point)
# integration_point_idx = 999

# # Extract the covariance history for the selected bundle and integration point
# P_mc_bundle = mc_covariances_history[bundle_index, time_indices, integration_point_idx, :, :]

# # Initialize lists to store diagonal elements (variances) of the covariance matrix over time
# position_variances = []
# velocity_variances = []

# # Loop over the selected time indices and extract the diagonal elements (variances)
# for i in range(len(time_indices)):
#     P = P_mc_bundle[i]  # Shape: (6,6)

#     # Print the covariance matrix P for the current time index in a formatted manner
#     print(f"Monte Carlo Covariance matrix for Bundle {bundle_index+1}, Time Step {time_indices[i]+1}, Integration Point {integration_point_idx+1}:")
#     for row in P:
#         print("  ".join(f"{val:.8f}" for val in row))  # Format each value to 8 decimal places
   
#     # Extract the diagonal elements of the covariance matrix (position and velocity variances)
#     position_variances.append([P[0, 0], P[1, 1], P[2, 2]])  # Variance of position (x, y, z)
#     velocity_variances.append([P[3, 3], P[4, 4], P[5, 5]])  # Variance of velocity (vx, vy, vz)