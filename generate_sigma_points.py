import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints

def generate_sigma_points(nsd=None, alpha=None, beta=None, kappa=None, P_pos=None, P_vel=None, num_time_steps = None, backTspan=None, r_bundles=None, v_bundles=None):
    """
    Function to generate sigma points for a combined state (position and velocity).
    
    Args:
    nsd (int): Dimensionality of the state (e.g., 6 for 3D position and 3D velocity).
    alpha (float): UKF parameter that controls the spread of the sigma points.
    beta (float): UKF parameter that adjusts the weight distribution for sigma points.
    kappa (float): UKF parameter that further adjusts the spread. Default is None (computed automatically).
    P_combined (ndarray): Initial covariance matrix for position and velocity.
    num_time_steps (int): Number of time steps to generate sigma points for.
    backTspan (array): The time span array for indexing, not defined in the code.
    r_bundles (ndarray): Position data array.
    v_bundles (ndarray): Velocity data array.
    
    Returns:
    sigmas_combined (ndarray): Generated sigma points (position and velocity combined).
    """
    
    if kappa is None:
        kappa = float(3 - nsd)
        
    lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter
    
    # Create weights for the sigma points using the MerweScaledSigmaPoints class
    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    
    # If P_combined is not provided, use a default block covariance matrix
    P_combined = np.block([
        P_pos,  # Position covariance with zero velocity covariance
        P_vel  # Velocity covariance
    ])
    
    # Intialize Time Steps
    time_steps = np.linspace(0, len(backTspan) - 1, num_time_steps, dtype=int)
    num_points = time_steps.shape[0]
    
    # Initialize the sigma points array
    num_bundles = r_bundles.shape[2]  # Assuming r_bundles are 3D (bundles, time_steps, dim)
    sigmas_combined = np.zeros((num_bundles, 2 * nsd + 1, nsd, num_points))
    
    # Loop over each bundle to generate sigma points for the combined state and time step
    for i in range(num_bundles):
        for j in range(num_points):
            # Get the nominal combined state (position and velocity) for the current time step
            nominal_combined = np.concatenate([r_bundles[time_steps[j], :, i], v_bundles[time_steps[j], :, i]])
            
            # Set the center sigma point (nominal combined state)
            sigmas_combined[i, 0, :, j] = nominal_combined
            
            # Generate sigma points using the weights and the covariance matrix
            sigmas_combined[i, :, :, j] = weights.sigma_points(nominal_combined, P_combined)
    
    return sigmas_combined, P_combined, time_steps, num_time_steps