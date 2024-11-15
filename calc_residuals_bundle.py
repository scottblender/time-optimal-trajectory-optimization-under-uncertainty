import numpy as np

def calc_residuals_bundle(delta_values, final_lam, max_angle):
    """
    This function calculates the residuals for a perturbed bundle trajectory. The residuals are
    used in the optimization process to adjust the trajectory parameters and ensure that the
    perturbed trajectory satisfies certain constraints.

    Parameters:
    - delta_values: A 5-element array representing the perturbation values for the first 5 parameters
      of the `final_lam` vector. These perturbations are applied to the corresponding parameters.
    - final_lam: A 7-element array containing the final nominal values of the lambda parameters
      (related to the spacecraft's dynamics and control).
    - max_angle: The maximum allowable angle between the perturbed and final lambda vectors, used as
      a constraint during optimization.

    Returns:
    - res: A 7-element array of residuals that indicate how well the perturbed lambda vector satisfies
      the optimization criteria.
      The residuals consist of:
        1. The difference between the angle between the perturbed and final `lam` vectors and `max_angle`.
        2. The difference between the norm of the `lam` matrix and 1 (which constrains the magnitude of `lam`).
        3. Three zero values added as placeholders for other residuals (could be expanded in future).
    """

    # Initialize a new lambda array with the same size as `final_lam`
    new_lam = np.zeros(7)

    # Apply the perturbations (delta_values) to the first five elements of `final_lam`
    new_lam[0:5] = final_lam[0:5] + delta_values[0:5]

    # Keep the last two elements of `final_lam` unchanged
    new_lam[5:] = final_lam[5:]

    # Extract the individual lambda values for clarity
    lam_p = new_lam[0]
    lam_f = new_lam[1]
    lam_g = new_lam[2]
    lam_h = new_lam[3]
    lam_k = new_lam[4]
    lam_L = new_lam[5]

    # Stack the individual lambda values into a matrix for easier handling
    lam_matrix = np.vstack((lam_p, lam_f, lam_g, lam_h, lam_k, lam_L))

    # Calculate the norm (magnitude) of the lambda matrix and subtract 1 to check if it is constrained to 1
    H = np.linalg.norm(lam_matrix) - 1

    # Calculate the angle between the perturbed `new_lam` vector and the final `final_lam` vector using dot product
    # This is a measure of the angular difference between the two vectors
    angle = np.arccos(np.divide(np.dot(new_lam, final_lam), (np.linalg.norm(new_lam) * np.linalg.norm(final_lam))))

    # Build the residual array:
    # The first residual is the difference between the angle and `max_angle` (constraint on the angular difference)
    # The second residual is the difference between the norm of the `lam_matrix` and 1 (constraint on magnitude)
    res = np.append(angle - max_angle, H - 1)

    # Append three zero values as placeholders for future expansion of residuals if needed
    res = np.append(res, np.zeros(3))

    # Return the residual array
    return res
