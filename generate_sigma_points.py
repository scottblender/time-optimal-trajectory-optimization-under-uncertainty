import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints

def generate_sigma_points(nsd=None, alpha=None, beta=None, kappa=None, P_pos=None, P_vel=None, P_mass=None, 
                          num_time_steps=None, backTspan=None, r_bundles=None, v_bundles=None, mass_bundles=None):
    """
    Function to generate sigma points for a combined state (position, velocity, and mass).
    
    Args:
    nsd (int): Dimensionality of the state (e.g., 6 for 3D position and 3D velocity, now 7 with mass).
    alpha (float): UKF parameter that controls the spread of the sigma points.
    beta (float): UKF parameter that adjusts the weight distribution for sigma points.
    kappa (float): UKF parameter that further adjusts the spread. Default is None (computed automatically).
    P_pos (ndarray): Covariance matrix for position.
    P_vel (ndarray): Covariance matrix for velocity.
    P_mass (float): Variance of mass (scalar, since mass is 1D).
    num_time_steps (int): Number of time steps to generate sigma points for.
    backTspan (array): The time span array for indexing.
    r_bundles (ndarray): Position data array.
    v_bundles (ndarray): Velocity data array.
    mass_bundles (ndarray): Mass history array.

    Returns:
    sigmas_combined (ndarray): Generated sigma points (position, velocity, mass combined).
    """
    
    if kappa is None:
        kappa = float(3 - nsd)
        
    lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter
    
    # Create weights for the sigma points using the MerweScaledSigmaPoints class
    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    
    # Construct full covariance matrix including mass uncertainty
    # Construct a 7x7 block matrix
    P_combined = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],  # Position block
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],  # Velocity block
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]  # Mass variance (1x1)
    ])
    
    # Initialize time steps
    time_steps = np.linspace(0, len(backTspan) - 1, num_time_steps, dtype=int)
    num_points = time_steps.shape[0]
    
    # Initialize the sigma points array
    num_bundles = r_bundles.shape[2]  # Assuming r_bundles are 3D (time_steps, 3, bundles)
    sigmas_combined = np.zeros((num_bundles, 2 * nsd + 1, nsd, num_points))
    
    # Loop over each bundle to generate sigma points for the combined state and time step
    for i in range(num_bundles):
        for j in range(num_points):
            # Get the nominal combined state (position, velocity, mass) for the current time step
            nominal_combined = np.concatenate([
                r_bundles[time_steps[j], :, i],  # Position (3)
                v_bundles[time_steps[j], :, i],  # Velocity (3)
                [mass_bundles[time_steps[j], i]]  # Mass (1)
            ])
            
            # Generate sigma points using the weights and the covariance matrix
            sigmas_combined[i, :, :, j] = weights.sigma_points(nominal_combined, P_combined)
    
    return sigmas_combined, P_combined, time_steps, num_time_steps, weights.Wm, weights.Wc
