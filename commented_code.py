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