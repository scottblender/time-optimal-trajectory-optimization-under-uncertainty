import numpy as np
import compute_bundle_trajectory_params

def evaluate_bundle_widths(bundle_counts, num_steps, final_lam, max_angle, final_pos, alpha, mu, F, c, m0, g0, tfound, backTspan, r_nom):
    """
    This function evaluates the width of spacecraft bundles for varying numbers of bundles, based on the 
    L2 norm of the positions. It returns the maximum width for each bundle count, calculated after
    removing outliers using the IQR method.

    Parameters:
    - bundle_counts: Array of integers specifying the number of bundles to evaluate.
    - num_steps: Number of steps for the backpropagation integration.
    - final_lam: The final lambda parameters (control parameters for the trajectory).
    - max_angle: The maximum allowed angular difference for bundle evaluation.
    - final_pos: The final position of the spacecraft.
    - alpha: A scaling factor used in perturbation of lambda values.
    - mu: The gravitational parameter.
    - F: The thrust force for the spacecraft.
    - c: Specific impulse.
    - m0: Initial mass of the spacecraft.
    - g0: Standard gravitational acceleration.
    - tfound: The final time.
    - backTspan: Time span for backward integration.
    - r_nom: The nominal position trajectory (for comparison with bundles).

    Returns:
    - max_widths: A list containing the maximum bundle widths (L2 norm) for each bundle count.
    """
    max_widths = []  # List to store the maximum width for each bundle count

    # Loop through each specified bundle count
    for num_bundles in bundle_counts:
        num_bundles = int(num_bundles)  # Ensure num_bundles is an integer

        # Call the function to evaluate bundles for the current number of bundles
        _, _, r_bundles, v_bundles, S_bundles = compute_bundle_trajectory_params(
            num_bundles, num_steps, final_lam, max_angle, final_pos, 
            alpha, mu, F, c, m0, g0, tfound, backTspan
        )
        
        max_width = 0  # Variable to store the maximum width for the current bundle count
        max_time = None  # Variable to store the time at which max width occurs
        
        # Iterate over each bundle to calculate the bundle width (L2 norm)
        for i in range(num_bundles):
            r_bundle = r_bundles[:, :, i]  # Get the current bundle's position trajectory
            
            # Calculate the L2 norm of the bundle's position differences from the nominal trajectory
            bundle_width = np.linalg.norm(r_bundle - r_nom, axis=0)  # Shape: (num_steps,)
            bundle_width = bundle_width[::-1]  # Reverse to match time direction
            
            # Perform outlier removal using Interquartile Range (IQR) method
            Q1 = np.percentile(bundle_width, 25)  # First quartile
            Q3 = np.percentile(bundle_width, 75)  # Third quartile
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
            upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
            
            # Filter out values outside the IQR bounds
            filtered_width = bundle_width[(bundle_width >= lower_bound) & (bundle_width <= upper_bound)]
            
            # Find the maximum width within the filtered values
            current_max_width = np.max(filtered_width)
            max_width = max(current_max_width, max_width)  # Update max width
            max_index = np.argmax(filtered_width)  # Get index of the max width
            max_time = np.linspace(0, tfound, len(filtered_width))[max_index]  # Get time at max width

        # Append the maximum width found for this number of bundles
        max_widths.append(max_width)

        # Print the result for each number of bundles (optional, for debugging)
        print(f"Maximum bundle width for {num_bundles} bundles: {max_width} at time {max_time}")

    return max_widths
