import numpy as np

def evaluate_bundle_widths(bundle_counts, r_nom, r_bundles, tfound):
    """
    This function evaluates the width of spacecraft bundles for varying numbers of bundles, based on the 
    L2 norm of the positions. It returns the maximum width for each bundle count, calculated after
    removing outliers using the IQR method. The time of maximum bundle width is also calculated 
    and returned.

    Parameters:
    - bundle_counts: Array of integers specifying the number of bundles to evaluate.
    - r_nom: The nominal position trajectory (Shape: [num_steps, 3]), used for comparison.
    - r_bundles: Position trajectories of the bundles (Shape: [num_steps, 3, num_bundles]).
    - tfound: The final time corresponding to the end of the simulation. This is used to scale the 
      time of maximum bundle width to the correct timespan.

    Returns:
    - max_widths: A list containing the maximum bundle widths (L2 norm) for each bundle count.
    - max_times: A list containing the times at which the maximum widths occur for each bundle count.
    """
    max_widths = []  # List to store the maximum width for each bundle count
    max_times = []   # List to store the times at which maximum width occurs for each bundle count

    # Loop through each specified bundle count
    for num_bundles in bundle_counts:
        num_bundles = int(num_bundles)  # Ensure num_bundles is an integer

        max_width = 0  # Variable to store the maximum width for the current bundle count
        max_time = None  # Variable to store the time at which max width occurs
        
        # Iterate over each bundle to calculate the bundle width (L2 norm)
        for i in range(num_bundles):
            r_bundle = r_bundles[:, :, i]  # Get the current bundle's position trajectory
            
            # Calculate the L2 norm of the bundle's position differences from the nominal trajectory
            bundle_width = np.linalg.norm(r_bundle - r_nom, axis=1)  # Shape: (num_steps,)
            bundle_width = bundle_width[::-1]  # Reverse to match time direction (so that time = 0 is first)
            
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
            
            # Calculate the time corresponding to the max width
            max_time = np.linspace(0, tfound, len(filtered_width))[max_index]  # Time at max width

        # Append the maximum width and time found for this number of bundles
        max_widths.append(max_width)
        max_times.append(max_time)

        # Print the result for each number of bundles (optional, for debugging)
        print(f"Maximum bundle width for {num_bundles} bundles: {max_width} at time {max_time}")

    return max_widths, max_times
