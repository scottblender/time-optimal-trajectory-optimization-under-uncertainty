import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Function to compute the deviations from the initial state (nominal trajectory) for each sigma point
def compute_deviations(trajectories):
    # Initialize an empty array to store the deviations (same shape as trajectories)
    deviations = np.zeros_like(trajectories)
    
    # Loop over bundles and time steps
    for i in range(trajectories.shape[0]):  # Loop over bundles
        for j in range(trajectories.shape[1]):  # Loop over time steps
            # Get the nominal state (first sigma point) for the current bundle and time step
            nominal_trajectory = trajectories[i, j, 0]  # Shape: (num_integration_points, num_states)
            
            # Loop over the sigma points (starting from 1 as the first sigma point is the nominal state)
            for k in range(1, trajectories.shape[2]):  # Loop over sigma points
                # Extract the trajectory for the current sigma point at the current time step
                sigma_point_trajectory = trajectories[i, j, k]  # Shape: (num_integration_points, num_states)
                
                # Compute the deviation from the nominal state
                deviation = sigma_point_trajectory - nominal_trajectory  # Element-wise subtraction
                
                # Store the deviation in the deviations array
                deviations[i, j, k] = deviation  # Store deviations for this bundle, time step, and sigma point
    
    return deviations

# Function to compute and plot PDF using Kernel Density Estimation (KDE)
def compute_pdf(deviations):
    # Reshape deviations to a 2D array: (num_samples, num_states)
    # Flatten the deviations across bundles, time steps, and sigma points
    flattened_deviations = deviations.reshape(-1, deviations.shape[-1])  # Shape: (num_samples, num_states)
    
    # KDE to estimate the PDF of the deviations
    kde = gaussian_kde(flattened_deviations.T)  # Transpose to have each state as a separate dimension
    return kde

# Function to compute and plot the KDE of the deviations for one bundle and one time step
def plot_pdf_with_kde(trajectories, bundle_idx, time_step_idx):
    # Extract the trajectories for the given bundle and time step
    bundle_trajectories = trajectories[bundle_idx, time_step_idx]  # Shape: (num_sigma_points, num_integration_points, num_states)
    
    # Extract the nominal state (the first sigma point) for reference
    nominal_trajectory = bundle_trajectories[0]  # Shape: (num_integration_points, num_states)
    
    # Set up the plot with 6 subplots (one for each state)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns of subplots
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # State labels for plotting
    state_labels = ['x (Position)', 'y (Position)', 'z (Position)', 'vx (Velocity)', 'vy (Velocity)', 'vz (Velocity)']
    
    # Loop over each of the 6 states (position and velocity components)
    for i in range(6):
        ax = axes[i]  # Get the current axis
        
        # Get the deviations for the current state (i-th state) for all sigma points
        deviations = []
        for sigma_idx in range(bundle_trajectories.shape[0]):  # Loop over sigma points
            state_trajectory = bundle_trajectories[sigma_idx, :, i]  # Extract the i-th state for this sigma point (shape: num_integration_points)
            deviation = state_trajectory - nominal_trajectory[:, i]  # Compute deviation from the nominal state (shape: num_integration_points)
            deviations.append(deviation)
        
        # Convert deviations to a numpy array (shape: num_sigma_points, num_integration_points)
        deviations = np.array(deviations)
        
        # Flatten the deviations to 1D for KDE
        flattened_deviations = deviations.flatten()

        # Compute the KDE for the flattened deviations using gaussian_kde
        kde = gaussian_kde(flattened_deviations)
        
        # Create an array of x values for plotting the KDE
        x_range = np.linspace(min(flattened_deviations), max(flattened_deviations), 1000)
        
        # Evaluate the KDE at the points in x_range
        pdf_vals = kde(x_range)
        
        # Plot the KDE on the current axis
        ax.plot(x_range, pdf_vals, label='KDE', color='blue')
        
        # Plot individual sigma points as vertical lines for reference
        for sigma_idx in range(bundle_trajectories.shape[0]):  # Loop over sigma points
            ax.axvline(deviations[sigma_idx, 0], linestyle='--', color='gray', alpha=0.7, label=f'Sigma Point {sigma_idx + 1}' if sigma_idx == 0 else "")
        
        # Highlight the nominal state (the first sigma point)
        ax.axvline(0, linestyle='-', color='black', label='Nominal', linewidth=2)
        
        # Set labels and title
        ax.set_title(f'{state_labels[i]} (Bundle {bundle_idx+1}, Time Step {time_step_idx+1})')
        ax.set_xlabel(f'{state_labels[i]} Deviation')
        ax.set_ylabel('PDF')
        ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
