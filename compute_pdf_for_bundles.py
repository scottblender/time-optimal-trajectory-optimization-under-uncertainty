import numpy as np
from scipy.stats import multivariate_normal

def compute_pdf_for_bundles(trajectories, P_combined_history, num_bundles, num_time_steps):
    """
    Compute the PDF for all bundles over all time steps, considering the entire 6D state vector.

    Parameters:
    - trajectories: The trajectories for all bundles and time steps.
    - P_combined_history: The covariance history for all bundles and time steps.
    - num_bundles: The number of bundles.
    - num_time_steps: The number of time steps.

    Returns:
    - pdf_history: A numpy array of PDFs for all bundles and time steps.
    """
    pdf_history = []  # List to store the PDFs for each bundle and time step

    # Loop over each bundle
    for i in range(num_bundles):
        bundle_pdf = []  # List to store the PDFs for each time step in the current bundle

        # Loop over each time step
        for j in range(num_time_steps-1):
            # Extract the sigma points for this time step and bundle
            sigma_points = trajectories[i][j]  # Shape: (13, 6) for bundle i, time step j

            # Extract the nominal trajectory (the first sigma point, which is the nominal)
            nominal_trajectory = sigma_points[0]  # This is the zeroth sigma point

            # Compute the covariance matrix for the current time step and bundle
            P_combined = P_combined_history[i, j]  # Covariance matrix at this time step and bundle

            # List to store PDF values for all sigma points
            pdf_values = []

            # Loop over each sigma point and compute the PDF using the multivariate Gaussian distribution
            for sigma_point in sigma_points:
                # Compute the deviation from the nominal trajectory (zero mean)
                deviation = sigma_point - nominal_trajectory

                # Compute the multivariate Gaussian PDF value for the deviation
                pdf = multivariate_normal.pdf(deviation, mean=np.zeros(6), cov=P_combined)
                pdf_values.append(pdf)

            # Store the PDFs for all sigma points at this time step
            bundle_pdf.append(np.array(pdf_values))

        # Append the PDFs for the entire bundle
        pdf_history.append(bundle_pdf)

    return np.array(pdf_history)  # Return the PDFs for all bundles and time steps
